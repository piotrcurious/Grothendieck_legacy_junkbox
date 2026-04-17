#include "Arduino.h"
#include <deque>
#include <vector>
#include <cmath>
#include <functional>

#include "../a/2.ino"

typedef std::function<double(double)> SignalGen;

void run_scenario(const std::string& name, SignalGen signal_gen, int steps) {
    Serial.printf("\n--- Scenario: %s ---\n", name.c_str());
    dataPoints.clear();
    filterWindow.clear();
    resetMockTime();

    for(int i = 0; i < steps; i++) {
        uint32_t now = millis();
        double t_sec = now / 1000.0;
        double rawValue = signal_gen(t_sec);

        DataPoint p = {now, rawValue};

        // Change-Point Detection (Inline for test runner)
        static uint8_t localOutlierCount = 0;
        if (dataPoints.size() >= MIN_DATA_POINTS) {
            Value m, l, u;
            resFitter.predictWithInterval(p.t, m, l, u);
            if (p.v < l || p.v > u) {
                localOutlierCount++;
                if (localOutlierCount > 5) {
                    Serial.println("Regime Shift Detected! Resetting buffer.");
                    dataPoints.clear();
                    localOutlierCount = 0;
                }
            } else {
                localOutlierCount = 0;
            }
        }

        dataPoints.push_back(p);
        if (dataPoints.size() > MAX_DATA_POINTS) dataPoints.pop_front();

        if (dataPoints.size() >= MIN_DATA_POINTS) {
            if (resFitter.fit(dataPoints)) {
                if (i % 10 == 0 || i == steps - 1) {
                    double conf = resFitter.growthConfidence(dataPoints);
                    double mean, lower, upper;
                    resFitter.predictWithInterval(now + 5000, mean, lower, upper);

                    Serial.printf("Step %d: RMSE=%.4f, Conf=%.1f%%, Target=%.4f, Pred=%.4f [%.4f, %.4f]\n",
                        i, resFitter.calculateCombinedRMSE(dataPoints), conf * 100.0,
                        signal_gen(t_sec + 5.0), mean, lower, upper);
                }
            }
        }
        delay(500);
    }
}

int main() {
    // 1. Pure Linear
    run_scenario("Linear Growth", [](double t){ return 10.0 + 2.0 * t; }, 50);

    // 2. Exponential
    run_scenario("Exponential Growth", [](double t){ return 2.0 * exp(0.5 * t); }, 50);

    // 3. Quadratic
    run_scenario("Quadratic Growth", [](double t){ return 10.0 + 0.5 * t * t; }, 60);

    // 4. Linear with Outliers
    run_scenario("Linear with Outliers", [](double t){
        double v = 10.0 + 2.0 * t;
        if (((int)(t * 2)) % 20 == 0 && t > 1.0) v += 50.0;
        return v;
    }, 100);

    // 5. Non-uniform sampling
    run_scenario("Non-uniform Sampling", [](double t){
        static double last_t = -1;
        static double acc_t = 0;
        if (last_t < 0) last_t = t;
        acc_t += (t - last_t) * (random(50, 150) / 100.0);
        last_t = t;
        return 5.0 + 1.5 * acc_t;
    }, 60);

    // 6. Step Change
    run_scenario("Step Change", [](double t){
        return (t < 15.0) ? 10.0 : 30.0;
    }, 80);

    // 7. Gapped Sampling
    run_scenario("Gapped Sampling", [](double t){
        static double base_t = -1;
        if (base_t < 0) base_t = t;
        double elapsed = t - base_t;
        if (elapsed > 10.0 && elapsed < 20.0) {
             delay(10000);
        }
        return 10.0 + 2.0 * elapsed;
    }, 50);

    // 8. Logistic Saturation
    run_scenario("Logistic Saturation", [](double t){
        double L = 100.0;
        double k = 0.3;
        double t0 = 15.0;
        return L / (1.0 + exp(-k * (t - t0)));
    }, 100);

    // 9. Compound Growth (Linear -> Exponential)
    run_scenario("Compound Growth", [](double t){
        if (t < 15.0) return 10.0 + 2.0 * t;
        else return (10.0 + 2.0 * 15.0) * exp(0.1 * (t - 15.0));
    }, 80);

    // 10. Periodic Trend
    run_scenario("Periodic Trend", [](double t){
        return 10.0 + 1.0 * t + 5.0 * sin(2.0 * M_PI * t / 5.0);
    }, 100);

    return 0;
}
