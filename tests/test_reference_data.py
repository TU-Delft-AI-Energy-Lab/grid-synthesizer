import pytest
import numpy as np
from powergrid_synth.core.reference_data import get_reference_stats


class TestReferenceData:

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_returns_dict_with_required_keys(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        assert isinstance(stats, dict)
        for key in [
            "system_name",
            "Tab_2D_Pgmax",
            "Tab_2D_load",
            "Tab_2D_Pg",
            "Tab_2D_FlBeta",
            "stab",
            "Overload_b",
            "mu_beta",
            "Zpr_pars",
        ]:
            assert key in stats, f"Missing key {key!r} for ref_sys_id={ref_sys_id}"

    @pytest.mark.xfail(reason="ref_sys_id=3 has inhomogeneous Tab_2D_load rows")
    def test_ref_sys_3_loads(self):
        get_reference_stats(3)

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_tab_2d_pgmax_is_2d_array(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        tab = stats["Tab_2D_Pgmax"]
        assert isinstance(tab, np.ndarray)
        assert tab.ndim == 2

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_stab_has_four_params(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        assert len(stats["stab"]) == 4

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_scalar_params_are_numeric(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        assert isinstance(stats["Overload_b"], (int, float))
        assert isinstance(stats["mu_beta"], (int, float))

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_zpr_pars_has_three_values(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        assert len(stats["Zpr_pars"]) == 3

    def test_invalid_ref_sys_raises(self):
        with pytest.raises(ValueError, match="Invalid Reference System ID"):
            get_reference_stats(99)

    def test_invalid_ref_sys_id_zero_raises(self):
        with pytest.raises(ValueError, match="Invalid Reference System ID"):
            get_reference_stats(0)

    def test_ref_sys_1_system_name(self):
        stats = get_reference_stats(1)
        assert stats["system_name"] == "Reference System 1"

    def test_ref_sys_2_system_name(self):
        stats = get_reference_stats(2)
        assert stats["system_name"] == "Reference System 2"

    @pytest.mark.parametrize("ref_sys_id", [1, 2])
    def test_tab_2d_pgmax_sums_close_to_one(self, ref_sys_id):
        stats = get_reference_stats(ref_sys_id)
        tab = stats["Tab_2D_Pgmax"]
        assert abs(tab.sum() - 1.0) < 0.05, (
            f"Tab_2D_Pgmax for ref_sys_id={ref_sys_id} should be a valid PMF"
        )
