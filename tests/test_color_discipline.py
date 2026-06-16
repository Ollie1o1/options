"""Color discipline: green/red mean directional sign only."""
import re
import unittest

from src import formatting as fmt

ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def codes(s):
    return ANSI_RE.findall(s)


class StyleSignTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(True)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_positive_uses_good(self):
        pos = fmt.style_sign("+$131", 131)
        self.assertEqual(codes(pos), codes(fmt.style("+$131", "good")))

    def test_negative_uses_bad(self):
        neg = fmt.style_sign("-$12", -12)
        self.assertEqual(codes(neg), codes(fmt.style("-$12", "bad")))

    def test_zero_is_neutral_value(self):
        z = fmt.style_sign("$0", 0)
        self.assertEqual(codes(z), codes(fmt.style("$0", "value")))


if __name__ == "__main__":
    unittest.main()
