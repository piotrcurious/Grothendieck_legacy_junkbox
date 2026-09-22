#!/usr/bin/env python3
import subprocess
import json
import os
import numpy as np

def test_cpp_cli_integration():
    cli_path = os.path.join(os.path.dirname(__file__), "lfsr_wiener_cli")
    out_json = os.path.join(os.path.dirname(__file__), "test_out.json")

    cmd = [cli_path, "-L", "4", "-p", "19", "-b", "1", "-o", out_json]
    subprocess.run(cmd, check=True)

    with open(out_json, 'r') as f:
        data = json.load(f)

    assert data["L"] == 4
    assert data["N"] == 15
    assert data["is_flat"] == True
    assert data["is_autocorr_two_valued"] == True

    bipolar = np.array(data["bipolar"])
    power_spectrum = np.array(data["power_spectrum"])
    autocorr = np.array(data["autocorrelation"])
    wiener_energy = np.array(data["wiener_energy_distribution"])

    assert len(bipolar) == 15
    assert len(power_spectrum) == 15
    assert len(autocorr) == 15
    assert len(wiener_energy) == 5

    np.testing.assert_allclose(power_spectrum[1:], 16.0, rtol=1e-6)
    assert autocorr[0] == 15.0
    np.testing.assert_allclose(autocorr[1:], -1.0, rtol=1e-6)
    assert wiener_energy[1] == 1.0

    # Test pair synthesis via CLI
    cmd_synth = [cli_path, "-L", "3", "-p", "11", "-b", "1", "--synthesize", "-LB", "4", "-pB", "19", "-bB", "1", "-m", "multiplicative", "-o", out_json]
    subprocess.run(cmd_synth, check=True)

    with open(out_json, 'r') as f:
        synth_data = json.load(f)

    assert synth_data["L_A"] == 3
    assert synth_data["L_B"] == 4
    assert synth_data["N_joint"] == 105
    assert len(synth_data["synthesized_bipolar"]) == 105
    assert len(synth_data["joint_power_spectrum"]) == 105
    assert len(synth_data["joint_autocorrelation"]) == 105

    print("Python Integration Test: PASS")

    if os.path.exists(out_json):
        os.remove(out_json)

if __name__ == "__main__":
    test_cpp_cli_integration()
