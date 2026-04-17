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
    EXPONENTIAL,
    SQUARE_ROOT
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
            case TransformType::SQUARE_ROOT:
                return sqrt(std::max(0.0, x + offset));
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
            case TransformType::SQUARE_ROOT:
                return (y * y) - offset;
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
    Value tMid = 0;
    Value tHalfRange = 1;
    Automorphism valTransform;

    double normalizeTime(Timestamp t) const {
        return (static_cast<double>(t) - tMid) / tHalfRange;
    }

public:
    PolynomialFitter() : coefficients(MAX_POLYNOMIAL_DEGREE + 1, 0.0) {}

    bool fit(const std::deque<DataPoint>& data, uint8_t targetDegree, TransformType vTrans = TransformType::LINEAR, double lambda = DEFAULT_RIDGE_LAMBDA, const std::vector<double>* weights = nullptr) {
        if (data.size() <= targetDegree) return false;
        degree = targetDegree;

        // Time normalization: mapping [tMin, tMax] to [-1, 1]
        Value tMin = data.front().t;
        Value tMax = data.front().t;
        for (const auto& p : data) {
            if (p.t < tMin) tMin = p.t;
            if (p.t > tMax) tMax = p.t;
        }
        tMid = (tMax + tMin) / 2.0;
        tHalfRange = (tMax - tMin) / 2.0;
        if (tHalfRange < 0.5) tHalfRange = 0.5;

        // Value Automorphism setup
        valTransform.type = vTrans;
        if (vTrans == TransformType::LOGARITHMIC || vTrans == TransformType::SQUARE_ROOT) {
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
        // Initial weights based on median-based outlier detection for better convergence
        std::vector<double> initialWeights(data.size(), 1.0);

        // Update transform parameters for initial pass
        valTransform.type = vTrans;
        if (vTrans == TransformType::LOGARITHMIC || vTrans == TransformType::SQUARE_ROOT) {
            Value minV = data[0].v;
            for (const auto& p : data) if (p.v < minV) minV = p.v;
            valTransform.offset = (minV <= 0) ? -minV + 1.0 : 0;
        }

        if (data.size() >= 5) {
            std::vector<double> vals;
            for (const auto& p : data) vals.push_back(valTransform.forward(p.v));
            std::sort(vals.begin(), vals.end());
            double median = vals[vals.size() / 2];
            std::vector<double> absDevs;
            for (double v : vals) absDevs.push_back(std::abs(v - median));
            std::sort(absDevs.begin(), absDevs.end());
            double mad = absDevs[absDevs.size() / 2];
            if (mad > 1e-6) {
                for (size_t i = 0; i < data.size(); i++) {
                    double v = valTransform.forward(data[i].v);
                    if (std::abs(v - median) > 3.0 * 1.4826 * mad) {
                        initialWeights[i] = 0.01; // Aggressively downweight initial outliers
                    }
                }
            }
        }

        if (!fit(data, targetDegree, vTrans, DEFAULT_RIDGE_LAMBDA, &initialWeights)) return false;

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
        double rss = calculateRSS(data);
        return sqrt(rss / data.size());
    }

    double calculateRSS(const std::deque<DataPoint>& data) const {
        double rss = 0;
        for (const auto& p : data) {
            double err = predict(p.t) - p.v;
            rss += err * err;
        }
        return rss;
    }

    // AICc: Akaike Information Criterion corrected for small sample sizes
    double calculateAICc(const std::deque<DataPoint>& data) const {
        size_t n = data.size();
        size_t k = degree + 1; // Number of parameters
        double rss = calculateRSS(data);
        if (rss < 1e-12) rss = 1e-12;

        double aic = n * log(rss / n) + 2 * k;
        if (n > k + 1) {
            aic += (2.0 * k * (k + 1)) / (n - k - 1);
        }
        return aic;
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

            const char* typeStr = "Linear";
            if (bestType == TransformType::LOGARITHMIC) typeStr = "Log";
            else if (bestType == TransformType::SQUARE_ROOT) typeStr = "Sqrt";

            Serial.printf("Best Fit: %s, Degree: %d, AICc: %.2f, RMSE: %.4f, Pred(now+5s): %.4f\n",
                typeStr, bestDegree, bestMetric, bestFitter.calculateRMSE(dataPoints),
                bestFitter.predict(lastSampleTime + 5000));
        }
    }
}
