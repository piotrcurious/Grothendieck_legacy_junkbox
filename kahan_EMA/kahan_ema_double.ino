#include <Arduino.h>

class KahanEMA_Double {
public:
    // Constructor
    KahanEMA_Double(double alpha) : alpha(alpha), ema(0.0), compensation(0.0) {}

    // Update the EMA with a new reading (using double precision)
    double update(double reading) {
        double y = alpha * (reading - ema) - compensation;
        double t = ema + y;
        compensation = (t - ema) - y;
        ema = t;
        return ema;
    }

    // Get the current EMA value
    double getValue() const {
        return ema;
    }

    // Reset the EMA and compensation
    void reset() {
        ema = 0.0;
        compensation = 0.0;
    }

private:
    double alpha;        // Smoothing factor (0.0 to 1.0)
    double ema;          // Current exponential moving average
    double compensation; // Kahan summation compensation term
};

// Define the smoothing factor (adjust as needed, using double)
const double ALPHA = 0.1;

// Create a KahanEMA_Double object
KahanEMA_Double sensorEMA(ALPHA);

void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial monitor time to connect
    Serial.println("ESP32 Kahan Exponential Averaging Example (Double Precision)");
}

void loop() {
    // Simulate reading a sensor value as a double
    // Replace this with your actual sensor reading code that returns a double
    // If your sensor provides float or int, cast it to double: (double)analogRead(GPIO_NUM_34)
    double sensorReading = (double)analogRead(GPIO_NUM_34) + random(-100, 100) * 0.01; // Example with noise on GPIO 34

    // Update the EMA with the new reading
    double filteredValue = sensorEMA.update(sensorReading);

    // Print the raw and filtered values (using printf for better double formatting)
    // Note: %lf is the format specifier for double with printf
    Serial.printf("Raw: %.4f, Filtered: %.4f\n", sensorReading, filteredValue);

    delay(100); // Adjust the delay based on your sampling rate
}
