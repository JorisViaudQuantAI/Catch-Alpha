"""An append-only record of every evaluation ("count every attempt").

The deflated Sharpe ratio needs N, the number of strategies that were tried. The
honest N is not the number of final candidates: it is every signal screen and every
gated evaluation ever run on the dataset. The ledger stores one line per evaluation
(dataset, split, metric family, experiment id, code version) and never deletes one.

The paper's real-data ledger is in ``results/real_data/trial_ledger_public_v3.json``:
353 records = 344 signal screens + 9 portfolio evaluations, exactly one of which
touched the final ("test") period. N counts the 352 selection (train/valid) records:
a final-period run tests an already frozen claim and is not a selection attempt.

>>> ledger = TrialLedger("my_ledger.json")              # doctest: +SKIP
>>> ledger.record("my_panel", "valid", "sharpe", "momentum_24h", "v1")  # doctest: +SKIP
>>> ledger.n_trials("my_panel")                          # doctest: +SKIP
1
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SPLITS = ("train", "valid", "test")
SELECTION_SPLITS = ("train", "valid")


def _git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""
    return out


class TrialLedger:
    """A JSON file holding ``{"version": 1, "records": [...]}``.

    Writes are serialized with a lock file and land through an atomic replace, so
    several processes can append to the same ledger without losing a record.
    """

    def __init__(self, path: str | Path, test_access_budget: int = 3) -> None:
        self.path = Path(path)
        self.test_access_budget = int(test_access_budget)

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return list(json.loads(self.path.read_text(encoding="utf-8"))["records"])

    def n_trials(self, dataset_id: str) -> int:
        """N for the deflated Sharpe ratio: every selection (train/valid) record on the dataset."""
        return sum(
            r["dataset_id"] == dataset_id and r["split"] in SELECTION_SPLITS for r in self.records()
        )

    def test_accesses(self, dataset_id: str) -> int:
        return sum(r["dataset_id"] == dataset_id and r["split"] == "test" for r in self.records())

    def summary(self, dataset_id: str) -> Counter:
        """Counts by (split, metric family), e.g. how many signal screens vs portfolios."""
        return Counter(
            (r["split"], r["metric_family"])
            for r in self.records()
            if r["dataset_id"] == dataset_id
        )

    def record(
        self,
        dataset_id: str,
        split: str,
        metric_family: str,
        experiment_id: str,
        code_version: str | None = None,
    ) -> dict:
        """Append one evaluation. Refuses a final-period access beyond the budget.

        ``code_version`` identifies the code that ran (default: the current git commit);
        an evaluation that cannot be traced to a code version is refused.
        """
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        version = _git_head() if code_version is None else str(code_version).strip()
        if not version:
            raise ValueError("code_version is required outside a git checkout")
        entry = {
            "timestamp_iso": datetime.now(UTC).isoformat(),
            "dataset_id": dataset_id,
            "split": split,
            "metric_family": metric_family,
            "experiment_id": experiment_id,
            "git_sha": version,
        }
        with self._write_lock():  # read, check and write as one step
            if split == "test" and self.test_accesses(dataset_id) >= self.test_access_budget:
                raise RuntimeError(
                    f"final-period budget of {self.test_access_budget} already spent"
                )
            payload = {"version": 1, "records": self.records() + [entry]}
            tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            try:
                tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
                os.replace(tmp, self.path)
            finally:
                tmp.unlink(missing_ok=True)
        return entry

    @contextmanager
    def _write_lock(self, timeout: float = 30.0) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(f"{self.path.name}.lock")
        deadline = time.monotonic() + timeout
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"ledger lock {lock} held for over {timeout} s") from None
                time.sleep(0.05)
        try:
            yield
        finally:
            os.close(fd)
            lock.unlink(missing_ok=True)
