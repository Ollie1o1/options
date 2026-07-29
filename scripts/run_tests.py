#!/usr/bin/env python3
"""Canonical local test runner for the options venv, which has no pytest.

CI runs `pytest tests/ -q --ignore=tests/_phase3_stress.py -m "not network"`.
Locally that is not available, and the obvious substitutes lie: globbing
`tests/test_*.py` collects only the top level and silently skips whole packages
(tests/outlook, tests/squeeze) while still reporting a confident green, and
`unittest discover` refuses to start because tests/ is not an importable
package.

So this runner loads every test module by path, runs the unittest.TestCase
tests in them, and — the point of the exercise — reports how many modules it
could NOT cover because they use pytest-only constructs (bare test functions,
fixtures like tmp_path). An under-collecting run is visible here rather than
reassuring.

    scripts/test.sh                # everything
    scripts/test.sh capital_risk   # modules matching a substring
"""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"

# Modules deliberately not run locally, with the reason. Anything here is still
# collected by CI unless CI ignores it too.
SKIP = {
    # CI ignores this one explicitly; it is a long-running stress harness.
    "_phase3_stress.py",
}


def _module_name(path: Path) -> str:
    """Dotted name from the repo-relative path, so same-named files in
    different packages cannot shadow each other in sys.modules."""
    return ".".join(path.relative_to(ROOT).with_suffix("").parts)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(_module_name(path), path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _has_testcases(module) -> bool:
    return any(
        isinstance(obj, type) and issubclass(obj, unittest.TestCase)
        and obj is not unittest.TestCase
        for obj in vars(module).values()
    )


def is_missing_pytest(exc: BaseException) -> bool:
    """True if an import failed only because pytest (or a plugin) is absent.

    Expected locally — the options venv deliberately has no pytest — so it must
    not turn the run red. Every other import failure is a real breakage.
    """
    name = getattr(exc, "name", None)
    if not isinstance(exc, ModuleNotFoundError) or not name:
        return False
    root = name.split(".")[0]
    return root == "pytest" or root.startswith("pytest_")


def _has_bare_test_functions(module) -> bool:
    return any(
        callable(obj) and name.startswith("test_")
        for name, obj in vars(module).items()
    )


def main(argv: list[str]) -> int:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    pattern = argv[1] if len(argv) > 1 else ""
    paths = sorted(
        p for p in TESTS.rglob("test_*.py")
        if p.name not in SKIP and "__pycache__" not in p.parts and pattern in str(p)
    )

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    ran, pytest_only, unloadable = [], [], []

    for path in paths:
        try:
            module = _load(path)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            if is_missing_pytest(exc):
                pytest_only.append(path)
            else:
                unloadable.append((path, exc))
            continue
        if _has_testcases(module):
            suite.addTests(loader.loadTestsFromModule(module))
            ran.append(path)
        elif _has_bare_test_functions(module):
            pytest_only.append(path)

    print(f"collected {suite.countTestCases()} tests from {len(ran)} modules")
    if pytest_only:
        print(
            f"NOT RUN: {len(pytest_only)} pytest-only modules "
            f"(bare test functions / fixtures) — covered by CI, not by this runner"
        )
    if unloadable:
        print(f"NOT RUN: {len(unloadable)} modules failed to import:")
        for path, exc in unloadable:
            print(f"  {path.relative_to(ROOT)}: {type(exc).__name__}: {exc}")
    print("-" * 70)

    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() and not unloadable else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
