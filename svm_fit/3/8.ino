#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

// Constants and Configuration
const int MAX_POLYNOMIAL_DEGREE = 5;
const size_t MAX_DATA_POINTS = 100;
const size_t MIN_DATA_POINTS = 3;

// Type Definitions
using Timestamp = unsigned long;
using Value = double;

struct DataPoint {
    Timestamp timestamp;
    Value value;

    DataPoint transformed(const std::function<Timestamp(Timestamp)>& timeTransform,
                          const std::function<Value(Value)>& valueTransform) const {
        return {timeTransform(timestamp), valueTransform(value)};
    }
};

// Automorphism Module
class FieldAutomorphisms {
public:
    std::function<Timestamp(Timestamp)> timeAutomorphism;
    std::function<Value(Value)> valueAutomorphism;

    FieldAutomorphisms()
        : timeAutomorphism([](Timestamp t) { return t; }),
          valueAutomorphism([](Value v) { return v; }) {}

    void selectOptimalAutomorphism(const std::vector<DataPoint>& dataPoints) {
        // Evaluate different automorphisms and choose the best fit
        auto linearScale = [](auto x) { return x; };
        auto logScale = [](auto x) { return x > 0 ? log(x) : 0; };
        auto expScale = [](auto x) { return exp(x); };

        timeAutomorphism = selectBestAutomorphism(dataPoints, linearScale, logScale, expScale, true);
        valueAutomorphism = selectBestAutomorphism(dataPoints, linearScale, logScale, expScale, false);
    }

    void applyTransformations(std::vector<DataPoint>& dataPoints) {
        for (auto& point : dataPoints) {
            point = point.transformed(timeAutomorphism, valueAutomorphism);
        }
    }

private:
    template <typename T>
    std::function<T(T)> selectBestAutomorphism(const std::vector<DataPoint>& dataPoints,
                                               std::function<T(T)> linearScale,
                                               std::function<T(T)> logScale,
                                               std::function<T(T)> expScale,
                                               bool isTime) {
        auto scoreFunction = [&](std::function<T(T)> scale) {
            Value sumResiduals = 0;
            for (const auto& point : dataPoints) {
                T x = isTime ? scale(point.timestamp) : scale(point.value);
                sumResiduals += fabs(x);  // Example scoring based on stability
            }
            return sumResiduals;
        };

        Value linearScore = scoreFunction(linearScale);
        Value logScore = scoreFunction(logScale);
        Value expScore = scoreFunction(expScale);

        if (linearScore <= logScore && linearScore <= expScore)
            return linearScale;
        if (logScore <= expScore)
            return logScale;
        return expScale;
    }
};

// Polynomial Fitting Module
class PolynomialFitter {
private:
    std::vector<Value> coefficients;
    FieldAutomorphisms& automorphisms;

public:
    PolynomialFitter(FieldAutomorphisms& automorphisms)
        : automorphisms(automorphisms), coefficients(MAX_POLYNOMIAL_DEGREE + 1, 0) {}

    bool fit(const std::vector<DataPoint>& data, int degree) {
        if (data.size() <= degree) return false;

        size_t n = data.size();
        std::vector<std::vector<Value>> A(degree + 1, std::vector<Value>(degree + 1, 0));
        std::vector<Value> B(degree + 1, 0);

        for (const auto& point : data) {
            Timestamp x = automorphisms.timeAutomorphism(point.timestamp);
            Value y = automorphisms.valueAutomorphism(point.value);

            for (int i = 0; i <= degree; i++) {
                for (int j = 0; j <= degree; j++) {
                    A[i][j] += pow(x, i + j);
                }
                B[i] += y * pow(x, i);
            }
        }

        coefficients = solveLinearSystem(A, B);
        return true;
    }

    Value evaluate(Timestamp t) const {
        Value x = automorphisms.timeAutomorphism(t);
        Value result = 0;
        for (size_t i = 0; i < coefficients.size(); i++) {
            result += coefficients[i] * pow(x, i);
        }
        return automorphisms.valueAutomorphism(result);
    }

    const std::vector<Value>& getCoefficients() const {
        return coefficients;
    }

private:
    std::vector<Value> solveLinearSystem(std::vector<std::vector<Value>>& A, std::vector<Value>& B) {
        size_t n = A.size();
        std::vector<Value> X(n, 0);

        for (size_t k = 0; k < n; k++) {
            for (size_t i = k + 1; i < n; i++) {
                Value factor = A[i][k] / A[k][k];
                for (size_t j = k; j < n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
                B[i] -= factor * B[k];
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            X[i] = B[i];
            for (size_t j = i + 1; j < n; j++) {
                X[i] -= A[i][j] * X[j];
            }
            X[i] /= A[i][i];
        }

        return X;
    }
};

// Main System
std::vector<DataPoint> dataPoints;
FieldAutomorphisms automorphisms;
PolynomialFitter fitter(automorphisms);

void setup() {
    Serial.begin(115200);
    Serial.println("Enhanced Polynomial Fitting System Initialized.");
}

void loop() {
    // Simulate data input
    Timestamp now = millis();
    Value newValue = sin(now / 1000.0) + random(-10, 10) / 100.0;
    dataPoints.push_back({now, newValue});

    // Maintain data size limit
    if (dataPoints.size() > MAX_DATA_POINTS) {
        dataPoints.erase(dataPoints.begin());
    }

    if (dataPoints.size() >= MIN_DATA_POINTS) {
        // Automorphism normalization
        automorphisms.selectOptimalAutomorphism(dataPoints);
        automorphisms.applyTransformations(dataPoints);

        // Polynomial fitting
        int degree = std::min(MAX_POLYNOMIAL_DEGREE, (int)dataPoints.size() - 1);
        if (fitter.fit(dataPoints, degree)) {
            // Log results
            const auto& coeffs = fitter.getCoefficients();
            Serial.println("Polynomial Coefficients:");
            for (const auto& c : coeffs) {
                Serial.printf("%f ", c);
            }
            Serial.println();
        }
    }

    delay(500);
}
