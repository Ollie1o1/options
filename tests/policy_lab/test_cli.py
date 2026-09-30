"""The `--grid` flag is the only CLI surface for choosing between the
750-cell exploratory grids and the preregistered 6-cell `NOSTOP_GRID`. These
tests pin the flag's choices and default without touching any database —
`main()` itself needs real corpus data, so it is exercised manually, not
here.
"""
import unittest

from src.policy_lab.cli import build_parser


class TestGridFlag(unittest.TestCase):
    def test_default_grid_is_full(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.grid, "full")

    def test_nostop_is_a_valid_choice(self):
        args = build_parser().parse_args(["--grid", "nostop"])
        self.assertEqual(args.grid, "nostop")

    def test_invalid_grid_choice_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--grid", "bogus"])


if __name__ == "__main__":
    unittest.main()
