# ESP32 Numerically Stable Exponential Moving Average (EMA) with Kahan Summation

This project provides Arduino code for implementing an Exponential Moving Average (EMA) filter on the ESP32, enhanced with the Kahan summation algorithm for improved numerical stability.

## What is Exponential Moving Average (EMA)?

Exponential Moving Average is a type of moving average that places a greater weight and significance on the most recent data points. It is commonly used to smooth out noisy data from sensors or other sources, making trends and patterns more apparent. The basic formula for EMA is:

$$EMA_{new} = \alpha \cdot \text{reading} + (1 - \alpha) \cdot EMA_{old}$$

where $$\alpha$$ (alpha) is the smoothing factor, a value between 0 and 1. A higher alpha value means the EMA is more responsive to recent changes, while a lower alpha value results in smoother data but slower response.

## Why use Kahan Summation?

Standard floating-point arithmetic on computers and microcontrollers can introduce small rounding errors in each calculation. When performing repeated additions or accumulations over time, these small errors can build up and lead to a significant loss of precision, especially when adding a small number to a large accumulated sum.

Kahan summation (also known as compensated summation) is a technique that significantly reduces this error accumulation during a sequence of additions. By keeping track of the lost lower-order bits in a separate "compensation" variable, it helps to restore the precision in subsequent additions.

In the context of the EMA, the update step involves adding a scaled difference (`alpha * (reading - EMA_old)`) to the previous EMA (`EMA_old`). When the EMA is large and the scaled difference is small, a standard addition can lose precision. This implementation uses Kahan summation to make this specific addition more accurate, leading to a more numerically stable EMA over long periods.

## Features

* Implements a standard Exponential Moving Average (EMA) filter.
* Integrates the Kahan summation algorithm into the EMA update for enhanced numerical stability.
* Configurable smoothing factor (`ALPHA`).
* Designed for use on the ESP32, compatible with the Arduino IDE.
* Supports both `float` and `double` precision for the EMA calculation.

## How it Works

The code defines a `KahanEMA` (or `KahanEMA_Double`) class that holds the current EMA value and the Kahan compensation term. The `update()` method takes a new reading and calculates the new EMA using a modified formula derived from the standard EMA, applying the Kahan summation principle to the addition:

$$ \text{difference\_term} = \alpha \cdot (\text{reading} - EMA_{old}) $$
$$ y = \text{difference\_term} - \text{compensation} $$
$$ t = EMA_{old} + y $$
$$ \text{compensation} = (t - EMA_{old}) - y $$
$$ EMA_{new} = t $$

This ensures that the error from adding `y` to `EMA_old` is captured in `compensation` and used to correct future calculations.

## Getting Started

1.  **Install Arduino IDE:** If you don't have it, download and install the Arduino IDE from [https://www.arduino.cc/software](https://www.arduino.cc/software).
2.  **Install ESP32 Board Definitions:** Follow the instructions on the Espressif website or various online guides to add ESP32 board support to your Arduino IDE.
3.  **Open the Code:** Copy the provided C++ code into a new sketch in the Arduino IDE.
4.  **Select ESP32 Board:** Go to `Tools > Board` and select your specific ESP32 board (e.g., "ESP32 Dev Module").
5.  **Select Port:** Go to `Tools > Port` and select the serial port connected to your ESP32.
6.  **Modify for Your Sensor:**
    * Locate the line in the `loop()` function that simulates a sensor reading:
        ```cpp
        double sensorReading = (double)analogRead(GPIO_NUM_34) + random(-100, 100) * 0.01; // Example with noise on GPIO 34
        ```
        Replace `analogRead(GPIO_NUM_34)` with the actual code to read your sensor. Make sure the sensor reading is converted to a `double` (or `float` if using that version) before being passed to `sensorEMA.update()`. Adjust or remove the noise simulation as needed.
    * Ensure the GPIO pin used (`GPIO_NUM_34` in the example) matches the physical pin your sensor is connected to for analog input (if applicable). Refer to your ESP32 board's pinout.
7.  **Upload:** Click the Upload button in the Arduino IDE.
8.  **Open Serial Monitor:** Once uploaded, open the Serial Monitor (`Tools > Serial Monitor`). Set the baud rate to 115200 to see the raw and filtered sensor values.

## Code Structure

* **`KahanEMA_Double` Class (or `KahanEMA` for float):**
    * Manages the state of the EMA (`ema`) and the compensation term (`compensation`).
    * `KahanEMA_Double(double alpha)`: Constructor to initialize the filter with a smoothing factor.
    * `update(double reading)`: Processes a new sensor reading and updates the EMA using Kahan summation. Returns the new filtered value.
    * `getValue()`: Returns the current filtered value.
    * `reset()`: Resets the filter state to zero.
* **`ALPHA` Constant:** Defines the smoothing factor. Adjust this value between 0.0 and 1.0 based on your desired smoothing level.
* **`setup()`:** Initializes serial communication.
* **`loop()`:** Reads sensor data (simulated), updates the EMA using the `update()` method, and prints the raw and filtered values.

## Configuration

* **Smoothing Factor (`ALPHA`):** Modify the `const double ALPHA = 0.1;` line (or `const float ALPHA = 0.1;` for the float version) to change how aggressively the EMA smooths the data. Values closer to 0.0 provide more smoothing, while values closer to 1.0 make the filter more responsive.
* **Precision (`float` vs `double`):** The provided code uses `double` precision. If you need to conserve memory or find `float` sufficient for your application's accuracy requirements, you can switch back to `float` by changing all occurrences of `double` in the `KahanEMA_Double` class definition and instantiation to `float`, and potentially renaming the class back to `KahanEMA`.

## Hardware

This code is specifically written for the **ESP32** microcontroller due to its floating-point capabilities and compatibility with the Arduino environment. The core Kahan EMA logic is portable, but the analog reading part (`analogRead`) is specific to Arduino-compatible boards.

## Contributing

If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request on the code's repository (if hosted on a platform like GitHub).

## License

This code is typically released under an open-source license like the MIT License, allowing you to use, modify, and distribute it freely. (Add the specific license information here if applicable).
