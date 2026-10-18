"""Offline learners for the controlled learning test: BC, IQL and CQL in plain PyTorch.

These are compact re-implementations of exactly the configuration the paper's learning
test uses (the original research code supported many more options). All three learners
act directly in weight space: the action is the portfolio weight vector, bounded by
``action_clip = 2`` (the same cap as the reference rule). Training data are transitions
``(obs, action, reward, next_obs, done)`` collected by a noisy copy of the reference.

* **BC** regresses the behaviour action with an MLP (two hidden ReLU layers), mean
  squared error, Adam(lr=1e-3). The output is clamped to the action box.
* **IQL** (Kostrikov et al., 2022) fits ``V`` by expectile regression (tau=0.7) on target
  ``Q``, fits twin ``Q`` to ``r + gamma (1 - done) V(s')``, and extracts the policy by
  advantage-weighted log-likelihood (temperature 3, advantage clip 10, weight cap 100).
  The actor is a tanh-squashed Gaussian scaled to the action box; acting uses its mean.
* **CQL** (Kumar et al., 2020) is SAC-style: twin ``Q`` with Huber Bellman loss on
  ``min Q_target(s', a') - alpha log pi(a'|s')`` plus the conservative penalty
  ``logsumexp_a Q(s, a) - Q(s, a_data)`` over uniform and policy action samples (2 per state for
  roughly the first third of training, then 1: the original progress counter is
  advanced twice per step, a quirk kept for exactness), fixed entropy weight alpha = 0.2, CQL weight 1.

Reproducibility: networks are created in a fixed order after :func:`set_global_seed`, and
minibatches are drawn exactly as ``torch.utils.data.DataLoader(shuffle=True)`` draws
them, so a fixed seed reproduces the original research runs bit for bit on CPU.
"""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


def set_global_seed(seed: int) -> None:
    """Seed every RNG the training loop can touch (Python, NumPy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@dataclass(frozen=True)
class OfflineDataset:
    """Offline transitions as float32 arrays (one row per decision time)."""

    obs: np.ndarray  # [B, obs_dim]
    action: np.ndarray  # [B, action_dim]
    reward: np.ndarray  # [B]
    next_obs: np.ndarray  # [B, obs_dim]
    done: np.ndarray  # [B]

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def shuffled_minibatches(dataset: OfflineDataset, batch_size: int) -> Iterator[dict[str, Tensor]]:
    """One epoch of shuffled minibatches, drawn like ``DataLoader(shuffle=True)``.

    A DataLoader iterator first draws a base seed from the global torch RNG, then its
    ``RandomSampler`` draws a second seed that drives a private ``randperm``. Mirroring
    both draws keeps the global RNG stream (and hence every later sample) identical.
    The final minibatch may be smaller than ``batch_size``.
    """
    n = len(dataset)
    torch.empty((), dtype=torch.int64).random_()  # DataLoader's per-iterator base seed
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator()
    generator.manual_seed(sampler_seed)
    order = torch.randperm(n, generator=generator)
    tensors = {
        "obs": torch.from_numpy(dataset.obs),
        "action": torch.from_numpy(dataset.action),
        "reward": torch.from_numpy(dataset.reward),
        "next_obs": torch.from_numpy(dataset.next_obs),
        "done": torch.from_numpy(dataset.done),
    }
    for start in range(0, n, batch_size):
        idx = order[start : start + batch_size]
        yield {key: value[idx] for key, value in tensors.items()}


def _mlp_trunk(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())


def _soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Polyak averaging ``target <- tau * source + (1 - tau) * target``."""
    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters(), strict=True):
            tgt.data.copy_(tau * src.data + (1.0 - tau) * tgt.data)


class QNet(nn.Module):
    """State-action value ``Q(s, a)``: MLP on the concatenation ``[s, a]``."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(*_mlp_trunk(obs_dim + action_dim, hidden), nn.Linear(hidden, 1))

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


class ValueNet(nn.Module):
    """State value ``V(s)``."""

    def __init__(self, obs_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(*_mlp_trunk(obs_dim, hidden), nn.Linear(hidden, 1))

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)


class SquashedGaussianActor(nn.Module):
    """Gaussian policy squashed by ``tanh`` into ``(-1, 1)^d`` (log-std clamped to [-5, 2])."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = _mlp_trunk(obs_dim, hidden)
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        h = self.net(obs)
        return self.mu(h), torch.clamp(self.log_std(h), -5.0, 2.0)

    def mean_action(self, obs: Tensor) -> Tensor:
        return torch.tanh(self.forward(obs)[0])

    def sample(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        """Reparameterised sample ``a = tanh(z)`` and its log-density (shape ``[B, 1]``)."""
        mu, log_std = self.forward(obs)
        normal = torch.distributions.Normal(mu, log_std.exp())
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        """Log-density of a squashed action in ``(-1, 1)^d`` (shape ``[B, 1]``)."""
        mu, log_std = self.forward(obs)
        clipped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))
        normal = torch.distributions.Normal(mu, log_std.exp())
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - clipped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)


class BCLearner:
    """Behaviour cloning: MSE regression of the behaviour weights."""

    def __init__(
        self, obs_dim: int, action_dim: int, *, hidden: int, action_clip: float, lr: float = 1e-3
    ) -> None:
        self.action_clip = float(action_clip)
        self.policy = nn.Sequential(*_mlp_trunk(obs_dim, hidden), nn.Linear(hidden, action_dim))
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def _forward(self, obs: Tensor) -> Tensor:
        return torch.clamp(self.policy(obs), -self.action_clip, self.action_clip)

    def fit(self, dataset: OfflineDataset, *, epochs: int, batch_size: int) -> None:
        for _ in range(epochs):
            for batch in shuffled_minibatches(dataset, batch_size):
                pred = self._forward(batch["obs"])
                n, d = batch["action"].shape
                # Mean squared error over rows and assets (the +1e-8 mirrors the original).
                loss = (pred - batch["action"]).pow(2).sum() / (n * d + 1e-8)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                self.opt.step()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0)
        out = self._forward(x).squeeze(0).numpy()
        return np.clip(out, -self.action_clip, self.action_clip).astype(np.float32)


class IQLLearner:
    """Implicit Q-learning with advantage-weighted policy extraction."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: int,
        action_clip: float,
        gamma: float,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        tau: float = 0.005,
        expectile: float = 0.7,
        awr_temperature: float = 3.0,
        exp_adv_max: float = 100.0,
        adv_clip: float = 10.0,
    ) -> None:
        self.action_clip = float(action_clip)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.expectile = float(expectile)
        self.awr_temperature = float(awr_temperature)
        self.exp_adv_max = float(exp_adv_max)
        self.adv_clip = float(adv_clip)
        # Creation order fixes the initial weights under a given seed.
        self.actor = SquashedGaussianActor(obs_dim, action_dim, hidden)
        self.value = ValueNet(obs_dim, hidden)
        self.q1 = QNet(obs_dim, action_dim, hidden)
        self.q2 = QNet(obs_dim, action_dim, hidden)
        self.q1_target = QNet(obs_dim, action_dim, hidden)
        self.q2_target = QNet(obs_dim, action_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=lr_critic)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )

    def _update(self, batch: dict[str, Tensor]) -> None:
        obs, act, next_obs = batch["obs"], batch["action"], batch["next_obs"]
        rew, done = batch["reward"].unsqueeze(-1), batch["done"].unsqueeze(-1)

        # 1) V: expectile regression towards min of the target Qs.
        with torch.no_grad():
            q_target = torch.minimum(self.q1_target(obs, act), self.q2_target(obs, act))
        diff = q_target - self.value(obs)
        weight = torch.where(diff > 0.0, self.expectile, 1.0 - self.expectile)
        value_loss = (weight * diff.pow(2)).mean()
        self.opt_value.zero_grad(set_to_none=True)
        value_loss.backward()
        self.opt_value.step()

        # 2) Q: one-step TD target through the (just updated) value network.
        with torch.no_grad():
            td_target = rew + self.gamma * (1.0 - done) * self.value(next_obs)
        q_loss = F.mse_loss(self.q1(obs, act), td_target) + F.mse_loss(self.q2(obs, act), td_target)
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        # 3) Actor: advantage-weighted log-likelihood of the behaviour action.
        with torch.no_grad():
            adv = (q_target - self.value(obs)).clamp(min=-self.adv_clip, max=self.adv_clip)
            awr_weight = torch.exp(self.awr_temperature * adv).clamp(max=self.exp_adv_max)
        unit_action = torch.clamp(act / self.action_clip, -1.0, 1.0)
        actor_loss = -(awr_weight * self.actor.log_prob(obs, unit_action)).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        _soft_update(self.q1, self.q1_target, self.tau)
        _soft_update(self.q2, self.q2_target, self.tau)

    def fit(self, dataset: OfflineDataset, *, epochs: int, batch_size: int) -> None:
        for _ in range(epochs):
            for batch in shuffled_minibatches(dataset, batch_size):
                self._update(batch)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0)
        return (self.actor.mean_action(x) * self.action_clip).squeeze(0).numpy()


class CQLLearner:
    """Conservative Q-learning (SAC actor, fixed entropy weight, Huber Bellman loss)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        hidden: int,
        action_clip: float,
        gamma: float,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        tau: float = 0.005,
        entropy_alpha: float = 0.2,
        cql_weight: float = 1.0,
        n_samples_schedule: tuple[int, int, int] = (2, 2, 1),
    ) -> None:
        self.action_clip = float(action_clip)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.entropy_alpha = float(entropy_alpha)
        self.cql_weight = float(cql_weight)
        self.n_samples_schedule = n_samples_schedule
        # Log-density of the uniform proposal, as used by the original implementation:
        # d * log(1/2) for actions on [-1, 1]^d.
        self.uniform_log_prob = float(-math.log(2.0) * action_dim)
        self.actor = SquashedGaussianActor(obs_dim, action_dim, hidden)
        self.q1 = QNet(obs_dim, action_dim, hidden)
        self.q2 = QNet(obs_dim, action_dim, hidden)
        self.q1_target = QNet(obs_dim, action_dim, hidden)
        self.q2_target = QNet(obs_dim, action_dim, hidden)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.opt_q = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_critic
        )

    def _policy_sample(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        unit_action, log_prob = self.actor.sample(obs)
        return unit_action * self.action_clip, log_prob

    def _conservative_penalty(
        self, obs: Tensor, q1_data: Tensor, q2_data: Tensor, n_samples: int
    ) -> Tensor:
        """``mean_s [logsumexp_a Q(s, a) - Q(s, a_data)]`` summed over both critics.

        The negatives are ``n_samples`` uniform actions on ``[-1, 1]^d`` and ``n_samples``
        current-policy actions per state, each importance-corrected by its log-density.
        """
        bsz, obs_dim = obs.shape
        action_dim = self.actor.mu.out_features
        obs_rep = obs.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, obs_dim)
        # Faithful to the original: the uniform proposal covers [-1, 1]^d, i.e. only the
        # inner half of the [-2, 2]^d action box the critics are trained on.
        uniform = torch.empty((bsz * n_samples, action_dim)).uniform_(0.0, 1.0) * 2.0 - 1.0
        policy_action, policy_log_prob = self._policy_sample(obs_rep)
        policy_log_prob = policy_log_prob.view(bsz, n_samples).detach()

        all_obs = torch.cat([obs_rep, obs_rep], dim=0)
        all_act = torch.cat([uniform, policy_action], dim=0)
        split = bsz * n_samples

        def critic_penalty(q_net: QNet, q_data: Tensor) -> Tensor:
            q_all = q_net(all_obs, all_act)
            q_uniform = q_all[:split].view(bsz, n_samples)
            q_policy = q_all[split : 2 * split].view(bsz, n_samples)
            negatives = torch.cat(
                [q_uniform - self.uniform_log_prob, q_policy - policy_log_prob], dim=1
            )
            return torch.logsumexp(negatives, dim=1).mean() - q_data.mean()

        # Both critics see the same negatives (q1 is evaluated before q2, as originally).
        penalty_q1 = critic_penalty(self.q1, q1_data)
        penalty_q2 = critic_penalty(self.q2, q2_data)
        return penalty_q1 + penalty_q2

    def _n_samples(self, progress: float) -> int:
        if progress < 1.0 / 3.0:
            return self.n_samples_schedule[0]
        if progress < 2.0 / 3.0:
            return self.n_samples_schedule[1]
        return self.n_samples_schedule[2]

    def fit(self, dataset: OfflineDataset, *, epochs: int, batch_size: int) -> None:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        total_steps = epochs * steps_per_epoch
        step = 0
        for epoch in range(epochs):
            for batch in shuffled_minibatches(dataset, batch_size):
                # NB: the original schedule adds the epoch offset to a step counter that is
                # already cumulative, so "progress" runs at twice the nominal rate.
                progress = (epoch * steps_per_epoch + step) / total_steps
                self._update(batch, self._n_samples(progress))
                step += 1

    def _update(self, batch: dict[str, Tensor], n_samples: int) -> None:
        obs, act, next_obs = batch["obs"], batch["action"], batch["next_obs"]
        rew, done = batch["reward"].unsqueeze(-1), batch["done"].unsqueeze(-1)
        alpha = torch.tensor(self.entropy_alpha, dtype=obs.dtype)

        # 1) Critics: soft Bellman target + conservative penalty.
        with torch.no_grad():
            next_action, next_log_prob = self._policy_sample(next_obs)
            q_next = torch.minimum(
                self.q1_target(next_obs, next_action), self.q2_target(next_obs, next_action)
            )
            td_target = rew + self.gamma * (1.0 - done) * (q_next - alpha * next_log_prob)
        q1_data = self.q1(obs, act)
        q2_data = self.q2(obs, act)
        bellman = F.smooth_l1_loss(q1_data, td_target) + F.smooth_l1_loss(q2_data, td_target)
        penalty = self._conservative_penalty(obs, q1_data, q2_data, n_samples)
        q_loss = bellman + self.cql_weight * penalty
        self.opt_q.zero_grad(set_to_none=True)
        q_loss.backward()
        self.opt_q.step()

        # 2) Actor: maximise min Q minus the entropy cost.
        new_action, log_prob = self._policy_sample(obs)
        q_pi = torch.minimum(self.q1(obs, new_action), self.q2(obs, new_action))
        actor_loss = (alpha * log_prob - q_pi).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.opt_actor.step()

        _soft_update(self.q1, self.q1_target, self.tau)
        _soft_update(self.q2, self.q2_target, self.tau)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        x = torch.tensor(np.asarray(obs, dtype=np.float32).reshape(-1)).unsqueeze(0)
        return (self.actor.mean_action(x) * self.action_clip).squeeze(0).numpy()


Learner = BCLearner | IQLLearner | CQLLearner


def make_learner(
    algorithm: str,
    obs_dim: int,
    action_dim: int,
    *,
    hidden: int,
    action_clip: float,
    gamma: float,
) -> Learner:
    """Build the E240 configuration of ``"bc"``, ``"iql"`` or ``"cql"``.

    ``gamma`` is the Bellman discount (ignored by BC). Call :func:`set_global_seed` first.
    """
    if algorithm == "bc":
        return BCLearner(obs_dim, action_dim, hidden=hidden, action_clip=action_clip)
    if algorithm == "iql":
        return IQLLearner(obs_dim, action_dim, hidden=hidden, action_clip=action_clip, gamma=gamma)
    if algorithm == "cql":
        return CQLLearner(obs_dim, action_dim, hidden=hidden, action_clip=action_clip, gamma=gamma)
    raise ValueError(f"unknown algorithm {algorithm!r}")
