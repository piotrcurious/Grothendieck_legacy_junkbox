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
            uint8_t degree = std::min((int)MAX_POLYNOMIAL_DEGREE, (int)dataPoints.size() / 5 + 1);

            TransformType types[] = {TransformType::LINEAR, TransformType::LOGARITHMIC};
            double bestRMSE = 1e30;
            TransformType bestType = TransformType::LINEAR;

            for (auto type : types) {
                PolynomialFitter fitter;
                if (fitter.fitRobust(dataPoints, degree, type)) {
                    double rmse = fitter.calculateRMSE(dataPoints);
                    if (rmse < bestRMSE) {
                        bestRMSE = rmse;
                        bestFitter = fitter;
                        bestType = type;
                    }
                }
            }
            if (i % 10 == 0 || i == steps - 1) {
                Serial.printf("Step %d: Best=%s, Degree=%d, RMSE=%.4f, Target(now+5s)=%.4f, Pred(now+5s)=%.4f\n",
                    i, (bestType == TransformType::LINEAR ? "Lin" : "Log"),
                    degree, bestRMSE, signal_gen(t_sec + 5.0), bestFitter.predict(now + 5000));
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

    // 3. Linear with Outliers
    run_scenario("Linear with Outliers", [](double t){
        double v = 10.0 + 2.0 * t;
        // Deterministic outliers for test
        if (((int)(t * 2)) % 20 == 0 && t > 1.0) v += 50.0;
        return v;
    }, 100);

    return 0;
}
