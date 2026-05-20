# Sheaf Polynomial Matching Suite

This directory contains tools for collecting data from an Arduino and matching it to feedback polynomials using sheaf aggregation and derivative-based techniques.

## Components

### 1. Arduino Firmware
- **`sheafs3_tune.ino`**: The main Arduino sketch. It collects analog data, converts it to binary, and performs on-device sheaf construction and polynomial matching.
  - Features: Tiered search strategy (Monte Carlo followed by Brute Force) and pattern-based candidate generation.
  - Usage: Upload to an Arduino (tested on Uno/Mega).

### 2. Python Visualizers & Analyzers
- **`sheafs3_visualizer.py`**: Portable live visualization of data and polynomial fitting.
  - Supports real serial connection or a `--port mock` mode for testing without hardware.
  - No external dependencies like `scipy` or `sklearn`.
  - Requirements: `pyserial`, `numpy`, `matplotlib`.
- **`sheafs3_cli_analyzer.py`**: A lightweight, CLI-only version of the analyzer for environments without a display.
  - Requirements: `numpy`.

### 3. Testing & Utilities
- **`sheaf_utils.py`**: Shared utilities for serial parsing and mock hardware simulation.
- **`mock_arduino.py`**: Standalone mock hardware simulator using an LFSR generator.
- **`test_integration.py`**: A script to verify the full data collection and matching cycle.

## Usage Instructions

### Running with Hardware
1. Upload `sheafs3_tune.ino` to your Arduino.
2. Run the visualizer:
   ```bash
   python3 sheafs/sheafs3_visualizer.py --port /dev/ttyACM0
   ```
   (Replace `/dev/ttyACM0` with your actual serial port).

### Testing without Hardware
1. Run the visualizer in mock mode:
   ```bash
   python3 sheafs/sheafs3_visualizer.py --port mock
   ```
2. Or use the CLI analyzer:
   ```bash
   python3 sheafs/sheafs3_cli_analyzer.py --port mock
   ```

## Mathematical Approach
The system uses principles of **Sheaf Theory** to group local data points into consistent global sections. We generate candidate polynomials based on local bit patterns and transitions, then refine these candidates using least squares fitting and exhaustive/Monte Carlo searches.
