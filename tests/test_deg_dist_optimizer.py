import pytest
import numpy as np
from powergrid_synth.transmission.deg_dist_optimizer import DegreeDistributionOptimizer


class TestDegreeDistributionOptimizer:

    @pytest.fixture()
    def optimizer(self):
        return DegreeDistributionOptimizer(verbose=False)

    def test_dgln_pdf_sums_to_one(self, optimizer):
        pdf = optimizer._dgln_pdf(100, alpha=1.5, beta=2.0)
        assert pdf.shape == (100,)
        assert abs(pdf.sum() - 1.0) < 1e-10

    def test_dgln_pdf_all_nonneg(self, optimizer):
        pdf = optimizer._dgln_pdf(50, alpha=1.0, beta=1.5)
        assert np.all(pdf >= 0)

    def test_dgln_pdf_invalid_params_returns_zeros(self, optimizer):
        pdf = optimizer._dgln_pdf(10, alpha=-1.0, beta=2.0)
        assert np.all(pdf == 0)

    def test_dpl_pdf_sums_to_one(self, optimizer):
        pdf = optimizer._dpl_pdf(100, gamma=2.5)
        assert pdf.shape == (100,)
        assert abs(pdf.sum() - 1.0) < 1e-10

    def test_dpl_pdf_all_nonneg(self, optimizer):
        pdf = optimizer._dpl_pdf(50, gamma=3.0)
        assert np.all(pdf >= 0)

    def test_optimize_dgln_returns_correct_structure(self, optimizer):
        params, pdf = optimizer.optimize(
            target_avg=3.5, max_deg=30, dist_type="dgln"
        )
        assert len(params) == 2  # alpha, beta
        assert isinstance(pdf, np.ndarray)
        assert abs(pdf.sum() - 1.0) < 1e-6

    def test_optimize_dpl_returns_correct_structure(self, optimizer):
        params, pdf = optimizer.optimize(
            target_avg=2.5, max_deg=20, dist_type="dpl"
        )
        assert len(params) == 1  # gamma
        assert isinstance(pdf, np.ndarray)

    def test_optimize_achieves_target_avg(self, optimizer):
        target = 3.0
        _, pdf = optimizer.optimize(
            target_avg=target, max_deg=40, dist_type="dgln"
        )
        degrees = np.arange(1, len(pdf) + 1)
        achieved_avg = np.dot(degrees, pdf)
        assert abs(achieved_avg - target) < 0.5  # generous tolerance

    def test_dgln_pdf_extreme_params_fallback(self, optimizer):
        """Extreme params that produce total=0 should fall back to p[0]=1."""
        pdf = optimizer._dgln_pdf(20, 1e-20, 1e20)
        assert pdf[0] == 1.0
        assert abs(np.sum(pdf) - 1.0) < 1e-6

    def test_objective_func_unknown_type_raises(self, optimizer):
        """_objective_func raises ValueError for an unknown dist_type."""
        with pytest.raises(ValueError, match="Unknown type"):
            optimizer._objective_func([2.0, 2.0], "unknown", 20, 3.0, 1e-10)

    def test_optimize_unknown_type_raises(self, optimizer):
        """optimize raises ValueError for an unsupported dist_type."""
        with pytest.raises(ValueError, match="must be"):
            optimizer.optimize(target_avg=3.0, max_deg=20, dist_type="unknown")

    def test_verbose_optimizer(self, capsys):
        """Verbose mode prints optimization progress and results."""
        opt = DegreeDistributionOptimizer(verbose=True)
        params, pdf = opt.optimize(target_avg=3.0, max_deg=20, dist_type="dgln")
        captured = capsys.readouterr()
        assert "Params:" in captured.out
        assert "Optimization Success:" in captured.out
        assert "Found Params:" in captured.out

    def test_dpl_pdf_decreasing(self, optimizer):
        """Power-law PDF is monotonically decreasing."""
        pdf = optimizer._dpl_pdf(20, 2.0)
        for i in range(len(pdf) - 1):
            assert pdf[i] >= pdf[i + 1]

    def test_objective_func_penalty_high_pmax(self, optimizer):
        """When P(d_max) exceeds prob_bound, penalty term y1 is positive."""
        score = optimizer._objective_func([0.5, 0.5], "dgln", 5, 2.0, 1e-10)
        score_no_bound = optimizer._objective_func([0.5, 0.5], "dgln", 5, 2.0, 1.0)
        assert score > score_no_bound
