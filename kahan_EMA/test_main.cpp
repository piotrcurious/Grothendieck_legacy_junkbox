#include "KahanEMA.hpp"
#include "Arduino.h"
#include <vector>
#include <cmath>
#include <cassert>

// Standard EMA implementation for comparison
template <typename T>
class StandardEMA {
public:
    StandardEMA(T alpha) : alpha(alpha), ema(0.0) {}
    T update(T reading) {
        ema = alpha * reading + (1.0 - alpha) * ema;
        return ema;
    }
    T getValue() const { return ema; }
private:
    T alpha;
    T ema;
};

void test_basic_convergence() {
    Serial.println("Testing basic convergence...");
    KahanEMA_Double ema(0.1);
    for (int i = 0; i < 100; ++i) {
        ema.update(10.0);
    }
    Serial.printf("Final value: %f\n", ema.getValue());
    assert(std::abs(ema.getValue() - 10.0) < 0.01);
    Serial.println("Basic convergence test passed.");
}

void test_numerical_stability() {
    Serial.println("Testing numerical stability...");
    // Using float to make precision issues more apparent
    float alpha = 0.0001f;
    KahanEMA kahanEma(alpha);
    StandardEMA<float> standardEma(alpha);

    float largeValue = 1e6f;
    float smallValue = 1.0f;

    // Initialize with large value
    for(int i=0; i<100; ++i) {
        kahanEma.update(largeValue);
        standardEma.update(largeValue);
    }

    Serial.printf("Initial Kahan EMA: %f\n", kahanEma.getValue());
    Serial.printf("Initial Standard EMA: %f\n", standardEma.getValue());

    // Update many times with small value
    // We expect it to slowly converge to smallValue (1.0)
    for (int i = 0; i < 1000000; ++i) {
        kahanEma.update(smallValue);
        standardEma.update(smallValue);
    }

    Serial.printf("Kahan EMA after 1M updates: %f\n", kahanEma.getValue());
    Serial.printf("Standard EMA after 1M updates: %f\n", standardEma.getValue());

    // In Kahan EMA, it should be closer to 1.0
    // Standard EMA might be stuck or drift due to (1-alpha)*ema + alpha*reading
    // with small alpha.

    // Let's try an even more extreme case where standard EMA completely fails to update
    kahanEma.reset();
    standardEma = StandardEMA<float>(alpha);

    float baseline = 10000.0f;
    for(int i=0; i<10000; ++i) {
        kahanEma.update(baseline);
        standardEma.update(baseline);
    }

    Serial.printf("Baseline set to %f\n", kahanEma.getValue());

    float tiny_increment = 0.01f;
    // reading = baseline + tiny_increment
    // expected change per step = alpha * tiny_increment = 0.0001 * 0.01 = 1e-6
    // baseline is 1e4. 1e4 + 1e-6 is beyond float precision (7 decimal digits)

    for(int i=0; i<1000000; ++i) {
        kahanEma.update(baseline + tiny_increment);
        standardEma.update(baseline + tiny_increment);
    }

    Serial.println("After 1M tiny increments:");
    Serial.printf("Kahan EMA: %f (diff: %f)\n", kahanEma.getValue(), kahanEma.getValue() - baseline);
    Serial.printf("Standard EMA: %f (diff: %f)\n", standardEma.getValue(), standardEma.getValue() - baseline);

    // Standard EMA: ema = alpha * reading + (1-alpha) * ema
    // If alpha is small, alpha * reading is small.
    // If ema is large, (1-alpha) * ema is close to ema.
    // Precision loss occurs when (alpha * reading) is added to ((1-alpha) * ema)
    // especially if (1-alpha)*ema is much larger than alpha*reading.
}

void test_precision_float() {
    Serial.println("Testing float precision stability...");
    float alpha = 0.01f;
    KahanEMA kahanEma(alpha);
    StandardEMA<float> standardEma(alpha);

    float val = 1.0f;
    for(int i=0; i<1000000; ++i) {
        kahanEma.update(val);
        standardEma.update(val);
    }

    Serial.printf("Kahan EMA (float): %f\n", kahanEma.getValue());
    Serial.printf("Standard EMA (float): %f\n", standardEma.getValue());

    assert(std::abs(kahanEma.getValue() - 1.0f) < 1e-5);
}

void test_initialization() {
    Serial.println("Testing initialization...");
    KahanEMA ema1(0.1f, true);  // Initialize with first reading
    float val = 10.0f;
    float result = ema1.update(val);
    assert(result == val);
    Serial.println("Initialization with first reading passed.");

    KahanEMA ema2(0.1f, false); // Initialize with 0
    result = ema2.update(val);
    assert(result != val);
    assert(result == 0.1f * val);
    Serial.println("Initialization with zero passed.");
}

void test_noisy_signal() {
    Serial.println("Testing noisy signal smoothing...");
    float alpha = 0.05f;
    KahanEMA kahanEma(alpha);

    Serial.println("SINE_DATA_START");
    for (int i = 0; i < 1000; ++i) {
        float clean = sin(i * 0.1f) * 10.0f;
        float noisy = clean + (random(-100, 100) / 100.0f) * 2.0f;
        float filtered = kahanEma.update(noisy);
        if (i % 5 == 0) {
            Serial.printf("%d,%f,%f,%f\n", i, clean, noisy, filtered);
        }
    }
    Serial.println("SINE_DATA_END");
}

void test_alpha_change() {
    Serial.println("Testing alpha change...");
    KahanEMA ema(0.1f);
    ema.update(10.0f);
    assert(ema.getAlpha() == 0.1f);

    ema.setAlpha(0.5f);
    assert(ema.getAlpha() == 0.5f);

    ema.setAlpha(2.0f);
    assert(ema.getAlpha() == 1.0f); // Clamped

    ema.setAlpha(-1.0f);
    assert(ema.getAlpha() == 0.0f); // Clamped
    Serial.println("Alpha change and clamping passed.");
}

void test_alpha_comparison() {
    Serial.println("ALPHA_COMP_START");
    float alphas[] = {0.01f, 0.05f, 0.1f, 0.2f};
    float step_val = 10.0f;

    for (float alpha : alphas) {
        KahanEMA ema(alpha, false); // Initialize with 0
        Serial.printf("ALPHA:%f\n", alpha);
        for (int i = 0; i < 200; ++i) {
            float val = ema.update(step_val);
            if (i % 2 == 0) {
                Serial.printf("%d,%f\n", i, val);
            }
        }
    }
    Serial.println("ALPHA_COMP_END");
}

void test_data_generation() {
    // This function will print data in a format easy to parse by Python
    float alpha = 0.00001f;
    float baseline = 100000.0f;
    float reading = baseline + 1.0f;
    KahanEMA kahanEma(alpha);
    StandardEMA<float> standardEma(alpha);

    // Init - KahanEMA(alpha, true) by default, so first update sets it to baseline
    kahanEma.update(baseline);

    // Standard EMA needs more steps to reach baseline if not manually set
    for(int i=0; i<1000000; ++i) {
        standardEma.update(baseline);
    }

    Serial.println("DATA_START");
    for(int i=0; i<1000000; ++i) {
        float k = kahanEma.update(reading);
        float s = standardEma.update(reading);
        if (i % 10000 == 0) {
            Serial.printf("%d,%f,%f\n", i, s, k);
        }
    }
    Serial.println("DATA_END");
}

int main() {
    test_basic_convergence();
    test_numerical_stability();
    test_precision_float();
    test_initialization();
    test_alpha_change();
    test_noisy_signal();
    test_alpha_comparison();
    test_data_generation();
    Serial.println("All tests completed.");
    return 0;
}
