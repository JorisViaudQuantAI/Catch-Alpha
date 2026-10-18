"""Every headline number of the paper must be reproduced by the files in results/."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verify_paper_numbers",
    Path(__file__).resolve().parents[1] / "scripts" / "verify_paper_numbers.py",
)
verify = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verify  # dataclasses look the module up while it loads
_SPEC.loader.exec_module(verify)

CHECKS = verify.all_checks()


@pytest.mark.parametrize("check", CHECKS, ids=[c.claim for c in CHECKS])
def test_paper_number(check):
    value, ok = check.run()
    assert ok, f"{check.claim}: paper {check.paper}, results/{check.source} gives {value}"
