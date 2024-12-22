#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>

// Configuration
const int MAX_POLYNOMIAL_DEGREE = 6;
const size_t MAX_DATA_POINTS = 100;
const size_t MIN_DATA_POINTS = 5;
const double RESIDUAL_THRESHOLD = 0.05;
const double LEARNING_RATE = 0.1;

// Type Definitions
using Timestamp = unsigned long;
using Value = double;

struct DataPoint {
    Timestamp timestamp;
    Value value;

    DataPoint transform(const std::function<Timestamp(Timestamp)>& timeTransform,
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
        // Evaluate automorphisms (linear, logarithmic, exponential, piecewise)
        timeAutomorphism = findBestTransform(dataPoints, true);
        valueAutomorphism = findBestTransform(dataPoints, false);
    }

private:
    std::function<Value(Value)> findBestTransform(const std::vector<DataPoint>& data, bool isTime) {
        auto linear = [](Value x) { return x; };
        auto logarithmic = [](Value x) { return x > 0 ? log(x) : 0; };
        auto exponential = [](Value x) { return exp(x); };

        auto piecewise = [](Value x) {
            return x < 1000 ? x * 0.5 : x * 1.2; // Example piecewise transform
        };

        // Compare transformations based on residual variance
        std::vector<std::function<Value(Value)>> transforms = {linear, logarithmic, exponential, piecewise};
        Value minVariance = std::numeric_limits<Value>::max();
        std::function<Value(Value)> bestTransform = linear;

        for (const auto& transform : transforms) {
            Value variance = calculateVariance(data, transform, isTime);
            if (variance < minVariance) {
                minVariance = variance;
                bestTransform = transform;
            }
        }
        return bestTransform;
    }

    Value calculateVariance(const std::vector<DataPoint>& data, 
                            const std::function<Value(Value)>& transform, 
                            bool isTime) {
        std::vector<Value> transformedValues;
        for (const auto& point : data) {
            Value value = isTime ? transform(point.timestamp) : transform(point.value);
            transformedValues.push_back(value);
        }
        Value mean = std::accumulate(transformedValues.begin(), transformedValues.end(), 0.0) / transformedValues.size();
        Value variance = 0;
        for (const auto& value : transformedValues) {
            variance += pow(value - mean, 2);
        }
        return variance / transformedValues.size();
    }
};

// Exponential Moving Average for Noise Reduction
class DataFilter {
public:
    static Value applyEMA(Value previousFiltered, Value newValue, double smoothingFactor = 0.1) {
        return (newValue * smoothingFactor) + (previousFiltered * (1 - smoothingFactor));
    }
};

// Polynomial Fitting Module
class PolynomialFitter {
private:
    std::vector<Value> coefficients;

public:
    bool fit(const std::vector<DataPoint>& data, int degree, const std::vector<double>& weights) {
        if (data.size() <= degree) return false;

        size_t n = data.size();
        std::vector<std::vector<Value>> A(degree + 1, std::vector<Value>(degree + 1, 0));
        std::vector<Value> B(degree + 1, 0);

        for (size_t k = 0; k < n; ++k) {
            Value x = data[k].timestamp;
            Value y = data[k].value;
            double w = weights[k];

            for (int i = 0; i <= degree; i++) {
                for (int j = 0; j <= degree; j++) {
                    A[i][j] += w * pow(x, i + j);
                }
                B[i] += w * y * pow(x, i);
            }
        }

        coefficients = solveLinearSystem(A, B);
        return true;
    }

    Value evaluate(Value x) const {
        Value result = 0;
        for (size_t i = 0; i < coefficients.size(); ++i) {
            result += coefficients[i] * pow(x, i);
        }
        return result;
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
PolynomialFitter fitter;

void setup() {
    Serial.begin(115200);
    Serial.println("Enhanced Polynomial Fitting System with Noise Reduction and Dynamic Adjustment");
}

void loop() {
    // Simulate data input
    Timestamp now = millis();
    Value rawValue = sin(now / 1000.0) + random(-10, 10) / 100.0;
    static Value filteredValue = 0;
    filteredValue = DataFilter::applyEMA(filteredValue, rawValue);

    dataPoints.push_back({now, filteredValue});

    if (dataPoints.size() > MAX_DATA_POINTS) {
        dataPoints.erase(dataPoints.begin());
    }

    if (dataPoints.size() >= MIN_DATA_POINTS) {
        automorphisms.selectOptimalAutomorphism(dataPoints);

        // Apply transformations and fit polynomial
        std::vector<double> weights(dataPoints.size(), 1.0);
        int degree = std::min(MAX_POLYNOMIAL_DEGREE, (int)dataPoints.size() - 1);
        fitter.fit(dataPoints, degree, weights);

        // Log polynomial coefficients
        Serial.println("Polynomial Coefficients:");
        for (const auto& coeff : fitter.getCoefficients()) {
            Serial.printf("%f ", coeff);
        }
        Serial.println();
    }

    delay(500);
}
