import unittest

from src.crypto.volbacktest.validate import mark_rmse_vol_pts


class TestValidate(unittest.TestCase):
    def test_zero_rmse_when_marks_match(self):
        model = [50.0, 52.0, 48.0]
        real = [50.0, 52.0, 48.0]
        self.assertAlmostEqual(mark_rmse_vol_pts(model_iv=model, real_iv=real), 0.0)

    def test_rmse_positive_and_correct(self):
        self.assertAlmostEqual(
            mark_rmse_vol_pts(model_iv=[50.0, 50.0], real_iv=[52.0, 48.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
