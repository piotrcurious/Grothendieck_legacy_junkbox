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
    SQUARE_ROOT,
    LOGISTIC
};

class Automorphism {
public:
    TransformType type = TransformType::LINEAR;
    Value offset = 0;
    Value L = 1.0; // Logistic carrying capacity

    Value forward(Value x) const {
        switch (type) {
            case TransformType::LOGARITHMIC:
                return log(std::max(1e-9, x + offset));
            case TransformType::EXPONENTIAL:
                return exp(x);
            case TransformType::SQUARE_ROOT:
                return sqrt(std::max(0.0, x + offset));
            case TransformType::LOGISTIC: {
                double val = std::max(1e-9, std::min(L - 1e-9, x + offset));
                return log(val / (L - val)); // Logit transform
            }
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
            case TransformType::LOGISTIC:
                return L / (1.0 + exp(-y)) - offset;
            case TransformType::LINEAR:
            default:
                return y;
        }
    }
};

class PolynomialFitter {
public:
    enum class BasisType {
        MONOMIAL,
        LEGENDRE
    };

    std::vector<Value> coefficients;

private:
    std::vector<std::vector<double>> R; // Upper triangular matrix from QR
    uint8_t degree = 0;
    Value tMid = 0;
    Value tHalfRange = 1;
    Value vMean = 0;
    Value vStd = 1;
    Automorphism valTransform;
    double sigma_est = 0;

    double normalizeTime(Timestamp t) const {
        return (static_cast<double>(t) - tMid) / tHalfRange;
    }

    double normalizeValue(Value v) const {
        return (v - vMean) / vStd;
    }

    Value denormalizeValue(double vn) const {
        return vn * vStd + vMean;
    }

    double evaluateBasis(uint8_t i, double x) const {
        if (basis == BasisType::MONOMIAL) {
            return pow(x, i);
        } else {
            // Legendre polynomials via recurrence
            if (i == 0) return 1.0;
            if (i == 1) return x;
            double p0 = 1.0, p1 = x, p2 = 0;
            for (uint8_t j = 2; j <= i; j++) {
                p2 = ((2.0 * j - 1.0) * x * p1 - (j - 1.0) * p0) / j;
                p0 = p1;
                p1 = p2;
            }
            return p1;
        }
    }

public:
    BasisType basis = BasisType::MONOMIAL;
    PolynomialFitter() : coefficients(MAX_POLYNOMIAL_DEGREE + 1, 0.0) {}

    bool fit(const std::deque<DataPoint>& data, uint8_t targetDegree, TransformType vTrans = TransformType::LINEAR, double lambda = DEFAULT_RIDGE_LAMBDA, const std::vector<double>* weights = nullptr, bool useLebesgueMeasure = false) {
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

        // Value Automorphism and Normalization Setup
        valTransform.type = vTrans;
        if (vTrans == TransformType::LOGARITHMIC || vTrans == TransformType::SQUARE_ROOT || vTrans == TransformType::LOGISTIC) {
            Value minV = data[0].v;
            Value maxV = data[0].v;
            for (const auto& p : data) {
                if (p.v < minV) minV = p.v;
                if (p.v > maxV) maxV = p.v;
            }
            valTransform.offset = (minV <= 0) ? -minV + 1.0 : 0;
            if (vTrans == TransformType::LOGISTIC) {
                Value lastFewAvg = 0;
                uint8_t count = 0;
                for (int i = (int)data.size() - 1; i >= 0 && count < 5; i--, count++) lastFewAvg += data[i].v;
                if (count > 0) lastFewAvg /= count;
                valTransform.L = std::max(maxV + valTransform.offset, lastFewAvg + valTransform.offset) + 1e-3;
                valTransform.L *= 1.1;
            }
        }

        double sumV = 0, sumV2 = 0;
        for (const auto& p : data) {
            double vf = valTransform.forward(p.v);
            sumV += vf; sumV2 += vf * vf;
        }
        vMean = sumV / data.size();
        vStd = sqrt(std::max(1e-9, (sumV2 / data.size()) - (vMean * vMean)));

        size_t k = targetDegree + 1;
        size_t m = data.size();
        size_t rows = m + k;
        std::vector<std::vector<double>> X(rows, std::vector<double>(k, 0.0));
        std::vector<double> y_vec(rows, 0.0);

        for (size_t i = 0; i < m; i++) {
            double x_norm = normalizeTime(data[i].t);
            double val_norm = normalizeValue(valTransform.forward(data[i].v));
            double w = weights ? (*weights)[i] : 1.0;

            if (useLebesgueMeasure && m > 1) {
                double dt = 0;
                if (i == 0) dt = (data[1].t - data[0].t);
                else if (i == m - 1) dt = (data[m-1].t - data[m-2].t);
                else dt = (data[i+1].t - data[i-1].t) / 2.0;
                w *= dt;
            }

            double sqrtW = sqrt(std::max(0.0, w));

            for (size_t j = 0; j < k; j++) {
                X[i][j] = sqrtW * evaluateBasis(j, x_norm);
            }
            y_vec[i] = sqrtW * val_norm;
        }

        // Ridge augmentation
        double sqrtLambda = sqrt(lambda);
        for (size_t j = 0; j < k; j++) {
            X[m + j][j] = sqrtLambda;
        }

        return solveQR(X, y_vec, rows, k);
    }

    bool fitRobust(const std::deque<DataPoint>& data, uint8_t targetDegree, TransformType vTrans = TransformType::LINEAR, int iterations = 5, bool useLebesgueMeasure = true) {
        std::vector<double> initialWeights(data.size(), 1.0);

        valTransform.type = vTrans;
        if (vTrans == TransformType::LOGARITHMIC || vTrans == TransformType::SQUARE_ROOT || vTrans == TransformType::LOGISTIC) {
            Value minV = data[0].v;
            Value maxV = data[0].v;
            for (const auto& p : data) {
                if (p.v < minV) minV = p.v;
                if (p.v > maxV) maxV = p.v;
            }
            valTransform.offset = (minV <= 0) ? -minV + 1.0 : 0;
            if (vTrans == TransformType::LOGISTIC) {
                Value lastFewAvg = 0;
                uint8_t count = 0;
                for (int i = (int)data.size() - 1; i >= 0 && count < 5; i--, count++) lastFewAvg += data[i].v;
                if (count > 0) lastFewAvg /= count;
                valTransform.L = std::max(maxV + valTransform.offset, lastFewAvg + valTransform.offset) + 1e-3;
                valTransform.L *= 1.1;
            }
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
                        initialWeights[i] = 0.01;
                    }
                }
            }
        }

        if (!fit(data, targetDegree, vTrans, DEFAULT_RIDGE_LAMBDA, &initialWeights, useLebesgueMeasure)) return false;

        for (int iter = 0; iter < iterations; iter++) {
            std::vector<double> residuals(data.size());
            for (size_t i = 0; i < data.size(); i++) {
                residuals[i] = std::abs(predict(data[i].t) - data[i].v);
            }

            std::vector<double> sortedRes = residuals;
            std::sort(sortedRes.begin(), sortedRes.end());
            double medianAbsRes = sortedRes[sortedRes.size() / 2];
            if (medianAbsRes < 1e-6) medianAbsRes = 1e-6;

            double sigma = 1.4826 * medianAbsRes;
            std::vector<double> weights(data.size());
            for (size_t i = 0; i < data.size(); i++) {
                double u = residuals[i] / (4.685 * sigma);
                if (std::abs(u) < 1.0) weights[i] = pow(1.0 - u * u, 2);
                else weights[i] = 0;
            }
            if (!fit(data, targetDegree, vTrans, DEFAULT_RIDGE_LAMBDA, &weights, useLebesgueMeasure)) break;
        }
        return true;
    }

    Value predict(Timestamp t) const {
        double x = normalizeTime(t);
        double y_norm = 0;
        for (uint8_t i = 0; i <= degree; i++) {
            y_norm += coefficients[i] * evaluateBasis(i, x);
        }
        return valTransform.inverse(denormalizeValue(y_norm));
    }

    void predictWithInterval(Timestamp t, Value& mean, Value& lower, Value& upper) const {
        mean = predict(t);
        double x_norm = normalizeTime(t);
        size_t k = degree + 1;

        std::vector<double> p(k);
        for (size_t i = 0; i < k; i++) p[i] = evaluateBasis(i, x_norm);

        std::vector<double> z(k);
        for (size_t i = 0; i < k; i++) {
            double sum = 0;
            for (size_t j = 0; j < i; j++) sum += R[j][i] * z[j];
            z[i] = (p[i] - sum) / R[i][i];
        }

        double leverage = 0;
        for (double val : z) leverage += val * val;

        double se = sigma_est * sqrt(leverage);
        double margin = 1.96 * se;

        lower = valTransform.inverse(denormalizeValue(normalizeValue(valTransform.forward(mean)) - margin));
        upper = valTransform.inverse(denormalizeValue(normalizeValue(valTransform.forward(mean)) + margin));
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

    double calculateAICc(const std::deque<DataPoint>& data) const {
        size_t n = data.size();
        size_t k = degree + 1;
        double rss = calculateRSS(data);
        if (rss < 1e-12) rss = 1e-12;

        double aic = n * log(rss / n) + 2 * k;
        if (n > k + 1) {
            aic += (2.0 * k * (k + 1)) / (n - k - 1);
        }
        return aic;
    }

private:
    bool solveQR(std::vector<std::vector<double>>& X, std::vector<double>& y_vec, size_t m, size_t n) {
        for (size_t k = 0; k < n; k++) {
            double max_val = 0;
            for (size_t i = k; i < m; i++) max_val = std::max(max_val, std::abs(X[i][k]));
            if (max_val < 1e-18) continue;

            double norm = 0;
            for (size_t i = k; i < m; i++) {
                X[i][k] /= max_val;
                norm += X[i][k] * X[i][k];
            }
            norm = sqrt(norm);

            if (X[k][k] > 0) norm = -norm;
            double u1 = X[k][k] - norm;
            X[k][k] = norm * max_val;

            for (size_t j = k + 1; j < n; j++) {
                double dot = u1 * X[k][j];
                for (size_t i = k + 1; i < m; i++) dot += X[i][k] * X[i][j];
                double tau = dot / (u1 * norm);
                X[k][j] += tau * u1;
                for (size_t i = k + 1; i < m; i++) X[i][j] += tau * X[i][k];
            }

            double dot_y = u1 * y_vec[k];
            for (size_t i = k + 1; i < m; i++) dot_y += X[i][k] * y_vec[i];
            double tau_y = dot_y / (u1 * norm);
            y_vec[k] += tau_y * u1;
            for (size_t i = k + 1; i < m; i++) y_vec[i] += tau_y * X[i][k];
        }

        R.assign(n, std::vector<double>(n, 0.0));
        for (size_t i = 0; i < n; i++) {
            for (size_t j = i; j < n; j++) R[i][j] = X[i][j];
        }

        for (int i = n - 1; i >= 0; i--) {
            if (std::abs(X[i][i]) < 1e-18) {
                coefficients[i] = 0;
            } else {
                double sum = 0;
                for (size_t j = i + 1; j < n; j++) sum += X[i][j] * coefficients[j];
                coefficients[i] = (y_vec[i] - sum) / X[i][i];
            }
        }

        double rss_transformed = 0;
        for (size_t i = n; i < m; i++) rss_transformed += y_vec[i] * y_vec[i];
        if (m > n) sigma_est = sqrt(rss_transformed / (m - n));
        else sigma_est = 0;

        return true;
    }
};

class HampelFilter {
public:
    static bool isOutlier(const std::deque<DataPoint>& data, double nSigma = 3.0) {
        if (data.size() < 7) return false;
        std::vector<double> window;
        for (const auto& p : data) window.push_back(p.v);
        size_t n = window.size();
        std::sort(window.begin(), window.end());
        double median = window[n / 2];
        std::vector<double> absDev;
        for (double v : window) absDev.push_back(std::abs(v - median));
        std::sort(absDev.begin(), absDev.end());
        double mad = absDev[n / 2];
        double sigma = 1.4826 * mad;
        return std::abs(data.back().v - median) > nSigma * sigma;
    }
};

struct ResidualFitter;

class RegimeMonitor {
public:
    uint8_t outlierCount = 0;
    const uint8_t threshold = 5;
    bool check(const DataPoint& p, const ResidualFitter& fitter);
};

struct ResidualFitter {
    PolynomialFitter baseFitter;
    PolynomialFitter residualFitter;
    PolynomialFitter directFitter;
    bool isHierarchical = false;
    bool hasModel = false;

    bool fit(const std::deque<DataPoint>& data) {
        hasModel = false;
        double bestOverallAICc = 1e30;
        TransformType types[] = {TransformType::LINEAR, TransformType::LOGARITHMIC, TransformType::EXPONENTIAL, TransformType::SQUARE_ROOT, TransformType::LOGISTIC};

        for (auto type : types) {
            uint8_t maxDegree = std::min((int)MAX_POLYNOMIAL_DEGREE, (int)data.size() - 2);
            for (uint8_t d = 1; d <= maxDegree; d++) {
                PolynomialFitter fitter;
                fitter.basis = (type == TransformType::LINEAR) ? PolynomialFitter::BasisType::LEGENDRE : PolynomialFitter::BasisType::MONOMIAL;
                if (fitter.fitRobust(data, d, type)) {
                    double aicc = fitter.calculateAICc(data);
                    if (aicc < bestOverallAICc) {
                        bestOverallAICc = aicc;
                        directFitter = fitter;
                        isHierarchical = false;
                        hasModel = true;
                    }
                }
            }
        }

        PolynomialFitter tempBase;
        if (tempBase.fit(data, 1, TransformType::LINEAR)) {
            std::deque<DataPoint> residuals;
            for (const auto& p : data) residuals.push_back({p.t, p.v - tempBase.predict(p.t)});
            for (auto type : types) {
                uint8_t maxDegree = std::min((int)MAX_POLYNOMIAL_DEGREE, (int)data.size() - 3);
                if (maxDegree < 1) continue;
                for (uint8_t d = 1; d <= maxDegree; d++) {
                    PolynomialFitter fitter;
                    fitter.basis = (type == TransformType::LINEAR) ? PolynomialFitter::BasisType::LEGENDRE : PolynomialFitter::BasisType::MONOMIAL;
                    if (fitter.fitRobust(residuals, d, type)) {
                        double aicc = fitter.calculateAICc(residuals);
                        if (aicc < bestOverallAICc) {
                            bestOverallAICc = aicc;
                            baseFitter = tempBase;
                            residualFitter = fitter;
                            isHierarchical = true;
                            hasModel = true;
                        }
                    }
                }
            }
        }
        return hasModel;
    }

    Value predict(Timestamp t) const {
        if (!hasModel) return 0;
        return isHierarchical ? (baseFitter.predict(t) + residualFitter.predict(t)) : directFitter.predict(t);
    }

    double calculateCombinedRMSE(const std::deque<DataPoint>& data) const {
        double rss = 0;
        for (const auto& p : data) {
            double err = predict(p.t) - p.v;
            rss += err * err;
        }
        return sqrt(rss / data.size());
    }

    double growthConfidence(const std::deque<DataPoint>& data) const {
        if (!hasModel) return 0.0;
        PolynomialFitter lin;
        if (!lin.fit(data, 1, TransformType::LINEAR)) return 0.0;
        double baseRMSE = lin.calculateRMSE(data);
        double modelRMSE = calculateCombinedRMSE(data);
        return baseRMSE < 1e-9 ? 0.0 : std::max(0.0, std::min(1.0, (baseRMSE - modelRMSE) / baseRMSE));
    }

    void predictWithInterval(Timestamp t, Value& mean, Value& lower, Value& upper) const {
        if (!hasModel) { mean = lower = upper = 0; return; }
        mean = predict(t);
        if (isHierarchical) {
            Value bm, bl, bu, rm, rl, ru;
            baseFitter.predictWithInterval(t, bm, bl, bu);
            residualFitter.predictWithInterval(t, rm, rl, ru);
            lower = bl + rl; upper = bu + ru;
        } else {
            directFitter.predictWithInterval(t, mean, lower, upper);
        }
    }
};

inline bool RegimeMonitor::check(const DataPoint& p, const ResidualFitter& fitter) {
    Value m, l, u;
    fitter.predictWithInterval(p.t, m, l, u);
    if (p.v < l || p.v > u) {
        if (++outlierCount > threshold) { outlierCount = 0; return true; }
    } else { outlierCount = 0; }
    return false;
}

std::deque<DataPoint> dataPoints;
std::deque<DataPoint> filterWindow;
ResidualFitter resFitter;
RegimeMonitor regimeMonitor;

void setup() {
    Serial.begin(115200);
    Serial.println(F("ESP32 Optimized Robust Polynomial Fitting System"));
}

void loop() {
    static uint32_t lastSampleTime = 0;
    if (millis() - lastSampleTime >= 500) {
        lastSampleTime = millis();
        double t_sec = lastSampleTime / 1000.0;
        double rawValue = 2.0 * exp(0.1 * t_sec) + (random(-100, 100) / 500.0);
        DataPoint p = {lastSampleTime, rawValue};
        filterWindow.push_back(p);
        if (filterWindow.size() > 10) filterWindow.pop_front();
        if (HampelFilter::isOutlier(filterWindow)) return;
        if (dataPoints.size() >= MIN_DATA_POINTS && regimeMonitor.check(p, resFitter)) dataPoints.clear();
        dataPoints.push_back(p);
        if (dataPoints.size() > MAX_DATA_POINTS) dataPoints.pop_front();
        if (dataPoints.size() >= MIN_DATA_POINTS && resFitter.fit(dataPoints)) {
            Value m, l, u;
            resFitter.predictWithInterval(lastSampleTime + 5000, m, l, u);
            Serial.printf("RMSE: %.4f, Conf: %.1f%%, Pred: %.4f [%.4f, %.4f]\n",
                resFitter.calculateCombinedRMSE(dataPoints), resFitter.growthConfidence(dataPoints) * 100.0, m, l, u);
        }
    }
}
