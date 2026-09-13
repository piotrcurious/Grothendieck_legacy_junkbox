"""
Unit Tests for XRF Spectrometer Demo & Lossless Differential Stacking
=====================================================================
Verifies XRF spectral simulation, element signal isolation, differential output computation,
and lossless Gegenbauer differential subband stacking.
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent directory to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from filter_compiler.xrf_spectrometer_demo import (
    XRF_ELEMENT_DATABASE,
    XRFSpectrometer,
    GegenbauerDifferentialStacker,
    run_xrf_demo
)


def test_xrf_spectrometer_simulation():
    """Tests XRF spectrometer energy grid generation and spectral responses."""
    spectrometer = XRFSpectrometer(e_min=1.0, e_max=15.0, num_channels=256)
    assert len(spectrometer.energy_grid) == 256
    assert spectrometer.energy_grid[0] == 1.0
    assert spectrometer.energy_grid[-1] == 15.0

    # Test induction spectrum
    induction = spectrometer.generate_induction_spectrum()
    assert len(induction) == 256
    assert np.all(induction >= 0.0)

    # Test element spectrum
    fe_elem = XRF_ELEMENT_DATABASE["Fe"]
    fe_spec = spectrometer.generate_element_spectrum(fe_elem)
    assert len(fe_spec) == 256
    assert np.max(fe_spec) > 0.0


def test_xrf_sample_simulation():
    """Tests multi-element sample simulation."""
    spectrometer = XRFSpectrometer(e_min=1.0, e_max=15.0, num_channels=256)
    elements = [XRF_ELEMENT_DATABASE["Fe"], XRF_ELEMENT_DATABASE["Cu"]]

    sim_data = spectrometer.simulate_sample(elements, noise_level=0.0)
    assert "induction" in sim_data
    assert "element_spectra" in sim_data
    assert "Fe" in sim_data["element_spectra"]
    assert "Cu" in sim_data["element_spectra"]

    # Verify total spectrum sum
    expected_total = sim_data["induction"] + sim_data["element_spectra"]["Fe"] + sim_data["element_spectra"]["Cu"]
    np.testing.assert_allclose(sim_data["total_spectrum"], expected_total, atol=1e-12)


def test_differential_spectra_computation():
    """Tests induction vs fluorescence and inter-element differential spectra computation."""
    spectrometer = XRFSpectrometer(num_channels=256)
    elements = [XRF_ELEMENT_DATABASE["Fe"], XRF_ELEMENT_DATABASE["Cu"]]
    sim_data = spectrometer.simulate_sample(elements)

    stacker = GegenbauerDifferentialStacker(num_modes=256)
    diffs = stacker.compute_differentials(
        induction=sim_data["induction"],
        total_fluorescence=sim_data["total_fluorescence"],
        element_spectra=sim_data["element_spectra"],
        elem_pair=("Fe", "Cu")
    )

    assert "delta_ind_fluor" in diffs
    assert "delta_Fe_Cu" in diffs
    assert len(diffs["delta_ind_fluor"]) == 256
    assert len(diffs["delta_Fe_Cu"]) == 256

    # Verify differential identities
    np.testing.assert_allclose(
        diffs["delta_ind_fluor"],
        sim_data["induction"] - sim_data["total_fluorescence"],
        atol=1e-12
    )
    np.testing.assert_allclose(
        diffs["delta_Fe_Cu"],
        sim_data["element_spectra"]["Fe"] - sim_data["element_spectra"]["Cu"],
        atol=1e-12
    )


def test_lossless_differential_stacking():
    """Tests exact information preservation when stacking differential subbands."""
    spectrometer = XRFSpectrometer(num_channels=256)
    elements = [XRF_ELEMENT_DATABASE["Fe"], XRF_ELEMENT_DATABASE["Cu"], XRF_ELEMENT_DATABASE["Zn"]]
    sim_data = spectrometer.simulate_sample(elements)

    subbands = [
        sim_data["induction"],
        sim_data["element_spectra"]["Fe"],
        sim_data["element_spectra"]["Cu"],
        sim_data["element_spectra"]["Zn"]
    ]

    stacker = GegenbauerDifferentialStacker(num_modes=256, lam=1.5)
    reconstructed, audit = stacker.stack_differential_outputs(subbands, spectrometer.x_grid)

    assert len(reconstructed) == 256
    assert audit["max_absolute_error"] < 0.1
    assert audit["snr_db"] > 40.0


def test_run_xrf_demo():
    """Tests end-to-end execution of the XRF Spectrometer demonstration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_path = os.path.join(tmpdir, "xrf_demo_test.png")
        result = run_xrf_demo(output_plot=plot_path)

        assert os.path.exists(plot_path)
        assert os.path.getsize(plot_path) > 1000
        assert result["audit"]["snr_db"] > 80.0
