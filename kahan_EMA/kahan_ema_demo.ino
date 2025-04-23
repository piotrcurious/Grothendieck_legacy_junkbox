#include <Arduino.h>

class KahanEMA {
public:
    // Constructor
    KahanEMA(float alpha) : alpha(alpha), ema(0.0), compensation(0.0) {}

    // Update the EMA with a new reading
    float update(float reading) {
        float y = alpha * (reading - ema) - compensation;
        float t = ema + y;
        compensation = (t - ema) - y;
        ema = t;
        return ema;
    }

    // Get the current EMA value
    float getValue() const {
        return ema;
    }

    // Reset the EMA and compensation
    void reset() {
        ema = 0.0;
        compensation = 0.0;
    }

private:
    float alpha;        // Smoothing factor (0.0 to 1.0)
    float ema;          // Current exponential moving average
    float compensation; // Kahan summation compensation term
};

// Define the smoothing factor (adjust as needed)
const float ALPHA = 0.1;

// Create a KahanEMA object
KahanEMA sensorEMA(ALPHA);

void setup() {
    Serial.begin(115200);
    delay(1000); // Give serial monitor time to connect
    Serial.println("ESP32 Kahan Exponential Averaging Example");
}

void loop() {
    // Simulate reading a sensor value
    // Replace this with your actual sensor reading code
    float sensorReading = analogRead(GPIO_NUM_34) + random(-10, 10) * 0.1; // Example with noise on GPIO 34

    // Update the EMA with the new reading
    float filteredValue = sensorEMA.update(sensorReading);

    // Print the raw and filtered values
    Serial.print("Raw: ");
    Serial.print(sensorReading);
    Serial.print(", Filtered: ");
    Serial.println(filteredValue);

    delay(100); // Adjust the delay based on your sampling rate
}
