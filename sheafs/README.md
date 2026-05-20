# Sheaf Polynomial Matching Suite

This directory contains tools for collecting data from an Arduino and matching it to feedback polynomials using sheaf aggregation and derivative-based techniques.

## Components

### 1. Arduino Firmware
- **`sheafs3_tune.ino`**: The main Arduino sketch. It collects analog data, converts it to binary, and performs on-device sheaf construction and polynomial matching.
  - Features: Derivative-based approximation, Monte Carlo search, and binary search for best-fit feedback polynomials.
  - Usage: Upload to an Arduino (tested on Uno/Mega).

### 2. Python Visualizers & Analyzers
- **`sheafs3_live_improved.py`**: Live visualization of data and polynomial fitting.
  - Supports real serial connection or a `--port mock` mode for testing without hardware.
  - Requirements: `pyserial`, `numpy`, `matplotlib`.
- **`sheafs3_cli_analyzer.py`**: A lightweight, CLI-only version of the analyzer for environments without a display.
  - Requirements: `numpy`.

### 3. Testing & Simulation
- **`mock_arduino.py`**: Simulates the serial output of an Arduino running `sheafs3_tune.ino`. Useful for testing Python scripts.
- **`test_integration.py`**: A simple script to verify the integration between mock output and parsing logic.

## Usage Instructions

### Running with Hardware
1. Upload `sheafs3_tune.ino` to your Arduino.
2. Run the live visualizer:
   ```bash
   python3 sheafs/sheafs3_live_improved.py --port /dev/ttyACM0
   ```
   (Replace `/dev/ttyACM0` with your actual serial port, e.g., `COM3` on Windows).

### Testing without Hardware
1. Run the live visualizer in mock mode:
   ```bash
   python3 sheafs/sheafs3_live_improved.py --port mock
   ```
2. Or use the CLI analyzer:
   ```bash
   python3 sheafs/sheafs3_cli_analyzer.py
   ```

## Mathematical Approach
The system uses **Sheaf Theory** to group local data points ("stalks") into consistent global sections. We approximate derivatives via finite differences in GF(2) to propose candidate polynomials, then refine these candidates using least squares fitting and exhaustive/Monte Carlo searches.
