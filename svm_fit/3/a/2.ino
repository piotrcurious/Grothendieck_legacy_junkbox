#include <Arduino.h>
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include <numeric>

// Configuration for ESP32
const uint8_t MAX_POLYNOMIAL_DEGREE = 6;
const uint16_t MAX_DATA_POINTS = 200;
const uint8_t MIN_DATA_POINTS = 10;
const double DEFAULT_RIDGE_LAMBDA = 1e-6;

using Timestamp = uint32_t;
using Value = double;

struct DataPoint {
    Timestamp t;
    Value v;
};

enum class TransformType {
    LINEAR,
    LOGARITHMIC,
    EXPONENTIAL
};

class Automorphism {
public:
    TransformType type = TransformType::LINEAR;
    Value offset = 0;

    Value forward(Value x) const {
        switch (type) {
            case TransformType::LOGARITHMIC:
                return log(std::max(1e-9, x + offset));
            case TransformType::EXPONENTIAL:
                return exp(x);
            case TransformType::LINEAR:
            default:
                return x;
        }
    }

    Value inverse(Value y) const {
        switch (type) {
            case TransformType::LOGARITHMIC:
                return exp(y) - offset;
            case TransformType::EXPONENTIAL:
                return log(std::max(1e-9, y));
            case TransformType::LINEAR:
            default:
                return y;
        }
    }
};

class PolynomialFitter {
private:
    std::vector<Value> coefficients;
    uint8_t degree = 0;
    Value tMin = 0;
    Value tRange = 1;
    Automorphism valTransform;

    double normalizeTime(Timestamp t) const {
        return static_cast<double>(t - tMin) / tRange;
    }

public:
    PolynomialFitter() : coefficients(MAX_POLYNOMIAL_DEGREE + 1, 0.0) {}

    bool fit(const std::deque<DataPoint>& data, uint8_t targetDegree, TransformType vTrans = TransformType::LINEAR, double lambda = DEFAULT_RIDGE_LAMBDA, const std::vector<double>* weights = nullptr) {
        if (data.size() <= targetDegree) return false;
        degree = targetDegree;

        // Time normalization
        tMin = data.front().t;
        Value tMax = data.front().t;
        for (const auto& p : data) {
            if (p.t < tMin) tMin = p.t;
            if (p.t > tMax) tMax = p.t;
        }
        tRange = tMax - tMin;
        if (tRange < 1.0) tRange = 1.0;

        // Value Automorphism setup
        valTransform.type = vTrans;
        if (vTrans == TransformType::LOGARITHMIC) {
            Value minV = data[0].v;
            for (const auto& p : data) if (p.v < minV) minV = p.v;
            valTransform.offset = (minV <= 0) ? -minV + 1.0 : 0;
        }

        size_t n = targetDegree + 1;
        std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
        std::vector<double> B(n, 0.0);

        for (size_t k = 0; k < data.size(); k++) {
            const auto& p = data[k];
            double x = normalizeTime(p.t);
            double y = valTransform.forward(p.v);
            double w = weights ? (*weights)[k] : 1.0;

            std::vector<double> powers(n);
            powers[0] = 1.0;
            for (uint8_t i = 1; i < n; i++) powers[i] = powers[i-1] * x;

            for (uint8_t i = 0; i < n; i++) {
                for (uint8_t j = 0; j < n; j++) {
                    A[i][j] += w * powers[i] * powers[j];
                }
                B[i] += w * y * powers[i];
            }
        }

        // Ridge Regularization
        for (uint8_t i = 0; i < n; i++) {
            A[i][i] += lambda;
        }

        return solveLinearSystem(A, B, n);
    }

    // Iteratively Reweighted Least Squares for robustness
    bool fitRobust(const std::deque<DataPoint>& data, uint8_t targetDegree, TransformType vTrans = TransformType::LINEAR, int iterations = 5) {
        if (!fit(data, targetDegree, vTrans)) return false;

        for (int iter = 0; iter < iterations; iter++) {
            std::vector<double> residuals(data.size());
            double medianAbsRes = 0;
            for (size_t i = 0; i < data.size(); i++) {
                residuals[i] = std::abs(predict(data[i].t) - data[i].v);
            }

            std::vector<double> sortedRes = residuals;
            std::sort(sortedRes.begin(), sortedRes.end());
            medianAbsRes = sortedRes[sortedRes.size() / 2];
            if (medianAbsRes < 1e-6) medianAbsRes = 1e-6;

            double sigma = 1.4826 * medianAbsRes; // Estimate of standard deviation
            std::vector<double> weights(data.size());
            for (size_t i = 0; i < data.size(); i++) {
                double u = residuals[i] / (4.685 * sigma); // Bisquare weight constant
                if (std::abs(u) < 1.0) {
                    weights[i] = pow(1.0 - u * u, 2);
                } else {
                    weights[i] = 0;
                }
            }
            if (!fit(data, targetDegree, vTrans, DEFAULT_RIDGE_LAMBDA, &weights)) break;
        }
        return true;
    }

    Value predict(Timestamp t) const {
        double x = normalizeTime(t);
        double y_transformed = 0;
        double xi = 1.0;
        for (uint8_t i = 0; i <= degree; i++) {
            y_transformed += coefficients[i] * xi;
            xi *= x;
        }
        return valTransform.inverse(y_transformed);
    }

    double calculateRMSE(const std::deque<DataPoint>& data) const {
        double sumSqErr = 0;
        for (const auto& p : data) {
            double err = predict(p.t) - p.v;
            sumSqErr += err * err;
        }
        return sqrt(sumSqErr / data.size());
    }

private:
    bool solveLinearSystem(std::vector<std::vector<double>>& A, std::vector<double>& B, uint8_t n) {
        for (uint8_t k = 0; k < n; k++) {
            // Pivoting
            uint8_t maxRow = k;
            double maxVal = std::abs(A[k][k]);
            for (uint8_t i = k + 1; i < n; i++) {
                if (std::abs(A[i][k]) > maxVal) {
                    maxVal = std::abs(A[i][k]);
                    maxRow = i;
                }
            }
            if (maxVal < 1e-18) return false;
            std::swap(A[k], A[maxRow]);
            std::swap(B[k], B[maxRow]);

            for (uint8_t i = k + 1; i < n; i++) {
                double factor = A[i][k] / A[k][k];
                B[i] -= factor * B[k];
                for (uint8_t j = k; j < n; j++) {
                    A[i][j] -= factor * A[k][j];
                }
            }
        }

        for (int i = n - 1; i >= 0; i--) {
            double sum = 0;
            for (uint8_t j = i + 1; j < n; j++) {
                sum += A[i][j] * coefficients[j];
            }
            coefficients[i] = (B[i] - sum) / A[i][i];
        }
        return true;
    }
};

std::deque<DataPoint> dataPoints;
PolynomialFitter bestFitter;

void setup() {
    Serial.begin(115200);
    Serial.println(F("ESP32 Optimized Robust Polynomial Fitting System"));
}

void loop() {
    static uint32_t lastSampleTime = 0;
    if (millis() - lastSampleTime >= 500) {
        lastSampleTime = millis();

        // Simulate some interesting data: Exponential growth with noise
        double t_sec = lastSampleTime / 1000.0;
        double rawValue = 2.0 * exp(0.1 * t_sec) + (random(-100, 100) / 500.0);

        dataPoints.push_back({lastSampleTime, rawValue});
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

            Serial.printf("Best Fit: %s, Degree: %d, RMSE: %.4f, Prediction(now+5s): %.4f\n",
                (bestType == TransformType::LINEAR ? "Linear" : "Log"),
                degree, bestRMSE, bestFitter.predict(lastSampleTime + 5000));
        }
    }
}
