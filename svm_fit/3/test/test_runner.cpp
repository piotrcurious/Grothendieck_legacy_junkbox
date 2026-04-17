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
    resetMockTime();

    for(int i = 0; i < steps; i++) {
        uint32_t now = millis();
        double t_sec = now / 1000.0;
        double rawValue = signal_gen(t_sec);

        dataPoints.push_back({now, rawValue});
        if (dataPoints.size() > MAX_DATA_POINTS) dataPoints.pop_front();

        if (dataPoints.size() >= MIN_DATA_POINTS) {
            double bestMetric = 1e30;
            TransformType bestType = TransformType::LINEAR;
            uint8_t bestDegree = 1;

            TransformType types[] = {TransformType::LINEAR, TransformType::LOGARITHMIC, TransformType::SQUARE_ROOT};

            for (auto type : types) {
                uint8_t maxPossibleDegree = std::min((int)MAX_POLYNOMIAL_DEGREE, (int)dataPoints.size() - 2);
                for (uint8_t d = 1; d <= maxPossibleDegree; d++) {
                    PolynomialFitter fitter;
                    if (fitter.fitRobust(dataPoints, d, type)) {
                        double aicc = fitter.calculateAICc(dataPoints);
                        if (aicc < bestMetric) {
                            bestMetric = aicc;
                            bestFitter = fitter;
                            bestType = type;
                            bestDegree = d;
                        }
                    }
                }
            }

            if (i % 10 == 0 || i == steps - 1) {
                const char* typeStr = "Lin";
                if (bestType == TransformType::LOGARITHMIC) typeStr = "Log";
                else if (bestType == TransformType::SQUARE_ROOT) typeStr = "Sqrt";

                double mean, lower, upper;
                bestFitter.predictWithInterval(now + 5000, mean, lower, upper);

                Serial.printf("Step %d: Best=%s, Degree=%d, AICc=%.2f, RMSE=%.4f, Target=%.4f, Pred=%.4f [%.4f, %.4f]\n",
                    i, typeStr, bestDegree, bestMetric, bestFitter.calculateRMSE(dataPoints),
                    signal_gen(t_sec + 5.0), mean, lower, upper);
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

    // 3. Quadratic (should be caught by Sqrt or higher degree Poly)
    run_scenario("Quadratic Growth", [](double t){ return 10.0 + 0.5 * t * t; }, 60);

    // 4. Linear with Outliers
    run_scenario("Linear with Outliers", [](double t){
        double v = 10.0 + 2.0 * t;
        // Deterministic outliers for test
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

    return 0;
}
