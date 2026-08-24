"""Module-level annotations must resolve on the Python CI actually runs.

Python 3.14 (the local venv) defers annotation evaluation. Python 3.11 and
3.12 (CI) evaluate module-level variable annotations EAGERLY at import. So

    _CAL_CACHE: Dict[str, Any] = {}

imports fine locally and raises `NameError: name 'Any' is not defined` on CI —
which is exactly what happened to `src/cli_display.py:594` in PR #63, taking
down BOTH test jobs at collection time, in every test file that transitively
imports it.

The local suite genuinely could not catch this by running: on 3.14 the
annotation is never evaluated. So this test does not run the code, it
reproduces 3.11's SEMANTICS — parse each module, find its module-level
annotated assignments, and evaluate each annotation against that module's own
namespace, which is precisely what 3.11 does at import.

Scoped to modules WITHOUT `from __future__ import annotations`: that import
makes every annotation a string on every version, so those files cannot have
the defect.

ON THE `eval` BELOW. It is deliberate and it is the only thing that works.
The input is not untrusted: it is annotation text parsed out of this repo's
OWN `src/` files, evaluated against the namespace of the module it came from —
byte for byte what CPython 3.11 does to those same lines at import. The test
suite already imports and executes every one of these modules. `literal_eval`
cannot substitute, because `Dict[str, Any]` is a subscript expression and not
a literal, which is exactly the construct that failed.
"""
from __future__ import annotations

import ast
import importlib
import os
import unittest

from src.paths import repo_path

SRC = repo_path("src")


def _modules_without_future_annotations():
    for root, _dirs, files in os.walk(SRC):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    source = fh.read()
            except OSError:
                continue
            if "from __future__ import annotations" in source:
                continue
            rel = os.path.relpath(path, os.path.dirname(SRC))
            dotted = rel[:-3].replace(os.sep, ".")
            if dotted.endswith(".__init__"):
                dotted = dotted[: -len(".__init__")]
            yield dotted, path, source


def _module_level_annotations(source):
    """(lineno, annotation source) for each top-level annotated assignment."""
    tree = ast.parse(source)
    out = []
    for node in tree.body:                      # top level ONLY
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            out.append((node.lineno, ast.unparse(node.annotation)))
    return out


class TestTheDetectorWorks(unittest.TestCase):
    """A guard nobody has pointed at a known failure is a guard nobody has
    tested."""

    def test_it_finds_an_unimported_name(self):
        src = "from typing import Dict\nX: Dict[str, Any] = {}\n"
        found = _module_level_annotations(src)
        self.assertEqual(found, [(2, "Dict[str, Any]")])
        ns = {}
        exec("from typing import Dict", ns)
        with self.assertRaises(NameError):
            eval(found[0][1], ns)

    def test_it_passes_when_the_name_is_imported(self):
        ns = {}
        exec("from typing import Dict, Any", ns)
        self.assertIsNotNone(eval("Dict[str, Any]", ns))


class TestEveryModuleLevelAnnotationResolves(unittest.TestCase):

    def test_no_module_level_annotation_would_raise_on_python_311(self):
        failures = []
        for dotted, path, source in _modules_without_future_annotations():
            annotations = _module_level_annotations(source)
            if not annotations:
                continue
            try:
                module = importlib.import_module(dotted)
            except Exception:
                # Import failures are another test's business; a module that
                # cannot import at all is not an annotation problem.
                continue
            ns = vars(module)
            for lineno, text in annotations:
                try:
                    eval(text, dict(ns))
                except NameError as exc:
                    failures.append(
                        f"{os.path.relpath(path, repo_path('.'))}:{lineno}  "
                        f"{text}  -> {exc}")
                except Exception:
                    # Only NameError reproduces the CI failure; anything else
                    # (a genuinely exotic annotation) is out of scope.
                    pass

        self.assertEqual(
            failures, [],
            "these annotations resolve on Python 3.14 but raise NameError on "
            "the 3.11/3.12 CI runs:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    unittest.main()
