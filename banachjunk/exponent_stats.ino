#include <Arduino.h>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>
#include "math_utils.h"

// Exponential Detection and Analysis Class
template <typename T>
class ExponentialDetector {
private:
    std::vector<T> rawData;
    
    // Exponential Characteristic Descriptor
    struct ExponentialCharacteristics {
        T baseExponent;        // Detected base of exponential growth
        T growthRate;          // Exponential growth rate
        T rSquared;            // Goodness of fit
        T inflectionPoint;     // Point of maximum rate change
        std::vector<T> residuals; // Deviation from perfect exponential
    };

    // Least Squares Exponential Regression
    ExponentialCharacteristics detectExponentialProperties() {
        if (rawData.size() < 3) return {};

        // Transform to log-linear space
        std::vector<T> logData;
        std::vector<T> xValues;
        for (size_t i = 0; i < rawData.size(); ++i) {
            if (rawData[i] > 0) {  // Avoid log(0)
                logData.push_back(std::log(rawData[i]));
                xValues.push_back(static_cast<T>(i));
            }
        }

        // Linear regression in log space
        T n = logData.size();
        T sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
        
        for (size_t i = 0; i < logData.size(); ++i) {
            sumX += xValues[i];
            sumY += logData[i];
            sumXY += xValues[i] * logData[i];
            sumXX += xValues[i] * xValues[i];
        }

        T denom = (n * sumXX - sumX * sumX);
        T slope = (abs(denom) > 1e-9) ? (n * sumXY - sumX * sumY) / denom : 0;
        T intercept = (sumY - slope * sumX) / n;

        // Compute characteristics
        ExponentialCharacteristics characteristics;
        characteristics.baseExponent = std::exp(intercept);
        characteristics.growthRate = slope;

        // Compute R-squared
        T yMean = sumY / n;
        T ssTotal = 0, ssResidual = 0;
        
        for (size_t i = 0; i < logData.size(); ++i) {
            T predicted = intercept + slope * xValues[i];
            ssTotal += std::pow(logData[i] - yMean, 2);
            ssResidual += std::pow(logData[i] - predicted, 2);
        }

        characteristics.rSquared = (abs(ssTotal) > 1e-9) ? (1 - (ssResidual / ssTotal)) : 1;

        // Compute residuals
        characteristics.residuals.resize(rawData.size());
        for (size_t i = 0; i < rawData.size(); ++i) {
            T predicted = characteristics.baseExponent * std::exp(characteristics.growthRate * i);
            characteristics.residuals[i] = rawData[i] - predicted;
        }

        // Simple exponential y = a*e^(bx) has no inflection point.
        // For detection purposes, we'll store the time constant.
        characteristics.inflectionPoint = (abs(characteristics.growthRate) > 1e-9) ? (1 / characteristics.growthRate) : 0;

        return characteristics;
    }

public:
    void reset() {
        rawData.clear();
    }

    void addDataPoint(T value) {
        rawData.push_back(value);
        
        // Limit buffer size
        if (rawData.size() > 100) {
            rawData.erase(rawData.begin());
        }
    }

    ExponentialCharacteristics analyzeExponentialProperties() {
        return detectExponentialProperties();
    }
};

// Banach Space for Statistical Analysis
template <typename T, size_t Dimension>
class StatisticalBanachSpace {
private:
    // Multi-dimensional statistical representation
    std::vector<std::vector<T>> statisticalData;
    std::vector<T> timestamps;

    // Statistical Descriptors
    struct StatisticalDescriptors {
        std::vector<T> mean;
        std::vector<T> variance;
        std::vector<T> stdDev;
        std::vector<T> skewness;
        std::vector<T> kurtosis;
        std::vector<T> entropy;
        std::vector<T> totalVariation;
        std::vector<T> confidenceIntervalLower;
        std::vector<T> confidenceIntervalUpper;
        std::vector<T> hurstExponent;
        std::vector<T> approxEntropy;
    };

    // Compute Comprehensive Statistical Metrics using Lebesgue-style measure weighting
    StatisticalDescriptors computeStatisticalMetrics() {
        StatisticalDescriptors metrics;
        metrics.mean.resize(Dimension, 0);
        metrics.variance.resize(Dimension, 0);
        metrics.stdDev.resize(Dimension, 0);
        metrics.skewness.resize(Dimension, 0);
        metrics.kurtosis.resize(Dimension, 0);
        metrics.entropy.resize(Dimension, 0);
        metrics.totalVariation.resize(Dimension, 0);
        metrics.confidenceIntervalLower.resize(Dimension, 0);
        metrics.confidenceIntervalUpper.resize(Dimension, 0);
        metrics.hurstExponent.resize(Dimension, 0.5);
        metrics.approxEntropy.resize(Dimension, 0);

        if (statisticalData.empty() || timestamps.size() < 2) return metrics;
        T totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return metrics;

        // Compute for each dimension
        for (size_t dim = 0; dim < Dimension; ++dim) {
            const auto& dimData = statisticalData[dim];
            if (dimData.size() < 2) continue;
            
            // Using centralized moment calculation
            auto moments = banach::Statistics::calculateMoments(dimData, timestamps);
            T mean = static_cast<T>(moments.mean);
            T variance = static_cast<T>(moments.variance);
            T sdev = std::sqrt(variance);

            metrics.mean[dim] = mean;
            metrics.variance[dim] = variance;
            metrics.stdDev[dim] = sdev;
            metrics.skewness[dim] = static_cast<T>(moments.skewness);
            metrics.kurtosis[dim] = static_cast<T>(moments.kurtosis);

            // Entropy (Shannon Information Entropy) using a simple histogram
            T entropy = 0;
            if (dimData.size() > 1) {
                T minVal = *std::min_element(dimData.begin(), dimData.end());
                T maxVal = *std::max_element(dimData.begin(), dimData.end());
                T range = maxVal - minVal;

                const int numBins = 10;
                std::vector<int> bins(numBins, 0);
                for (const auto& val : dimData) {
                    int binIdx = (range > 1e-9) ? static_cast<int>((val - minVal) / range * (numBins - 1)) : 0;
                    bins[binIdx]++;
                }

                for (int count : bins) {
                    if (count > 0) {
                        T p = static_cast<T>(count) / dimData.size();
                        entropy -= p * std::log2(p);
                    }
                }
            }
            metrics.entropy[dim] = entropy;

            // Total Variation (measure of signal "wiggliness")
            T tv = 0;
            for (size_t i = 1; i < dimData.size(); ++i) {
                tv += std::abs(dimData[i] - dimData[i-1]);
            }
            metrics.totalVariation[dim] = tv;

            // 95% Confidence Interval for the mean (assuming normality for proxy)
            // CI = mean +/- 1.96 * (sdev / sqrt(N_eff))
            // N_eff is approximated here by actual sample count
            T marginOfError = 1.96 * (sdev / std::sqrt(static_cast<T>(dimData.size())));
            metrics.confidenceIntervalLower[dim] = mean - marginOfError;
            metrics.confidenceIntervalUpper[dim] = mean + marginOfError;

            // Unified metrics from math_utils
            metrics.hurstExponent[dim] = static_cast<T>(banach::Statistics::calculateHurst(dimData, timestamps));
            metrics.approxEntropy[dim] = static_cast<T>(banach::Statistics::calculateApEn(dimData));
        }

        return metrics;
    }

public:
    void reset() {
        statisticalData.clear();
        timestamps.clear();
    }

    // Add multi-dimensional statistical data point with timestamp
    void addStatisticalDataPoint(const std::vector<T>& point, T timestamp = -1.0) {
        if (point.size() != Dimension) return;
        
        if (statisticalData.size() < Dimension) statisticalData.resize(Dimension);
        
        if (timestamp < 0) {
            timestamp = timestamps.empty() ? 0 : timestamps.back() + 1.0;
        }
        timestamps.push_back(timestamp);
        if (timestamps.size() > 100) timestamps.erase(timestamps.begin());

        for (size_t i = 0; i < Dimension; ++i) {
            statisticalData[i].push_back(point[i]);
            if (statisticalData[i].size() > 100) {
                statisticalData[i].erase(statisticalData[i].begin());
            }
        }
    }

    // Compute covariance between two dimensions (Lebesgue-weighted)
    T computeCovariance(size_t dim1, size_t dim2) {
        if (dim1 >= Dimension || dim2 >= Dimension) return 0;
        const auto& data1 = statisticalData[dim1];
        const auto& data2 = statisticalData[dim2];
        if (data1.size() != data2.size() || data1.size() < 2) return 0;
        T totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return 0;

        T m1 = 0, m2 = 0;
        for (size_t i = 0; i < data1.size() - 1; i++) {
            T dt = timestamps[i+1] - timestamps[i];
            m1 += (data1[i] + data1[i+1]) / 2.0 * dt;
            m2 += (data2[i] + data2[i+1]) / 2.0 * dt;
        }
        T mean1 = m1 / totalMeasure;
        T mean2 = m2 / totalMeasure;

        T covarIntegral = 0;
        for (size_t i = 0; i < data1.size() - 1; ++i) {
            T dt = timestamps[i+1] - timestamps[i];
            T d1 = (data1[i] + data1[i+1]) / 2.0 - mean1;
            T d2 = (data2[i] + data2[i+1]) / 2.0 - mean2;
            covarIntegral += d1 * d2 * dt;
        }
        return covarIntegral / totalMeasure;
    }

    // Comprehensive Statistical Banach Space Analysis
    void performStatisticalAnalysis() {
        if (statisticalData.empty()) return;
        
        // Compute Statistical Metrics
        StatisticalDescriptors metrics = computeStatisticalMetrics();
        
        // Output Analysis
        Serial.println("\n--- Statistical Banach Space Analysis ---");
        
        for (size_t dim = 0; dim < Dimension; ++dim) {
            Serial.printf("\nDimension %d:\n", dim);
            Serial.printf("  Mean: %f\n", static_cast<float>(metrics.mean[dim]));
            Serial.printf("  StdDev: %f\n", static_cast<float>(metrics.stdDev[dim]));
            Serial.printf("  Variance: %f\n", static_cast<float>(metrics.variance[dim]));
            Serial.printf("  Skewness: %f\n", static_cast<float>(metrics.skewness[dim]));
            Serial.printf("  Kurtosis: %f\n", static_cast<float>(metrics.kurtosis[dim]));
            Serial.printf("  Entropy: %f\n", static_cast<float>(metrics.entropy[dim]));
            Serial.printf("  Hurst Exponent: %f\n", static_cast<float>(metrics.hurstExponent[dim]));
            Serial.printf("  Approx Entropy: %f\n", static_cast<float>(metrics.approxEntropy[dim]));
            Serial.printf("  Total Variation: %f\n", static_cast<float>(metrics.totalVariation[dim]));
            Serial.printf("  95%% CI Mean: [%f, %f]\n",
                static_cast<float>(metrics.confidenceIntervalLower[dim]),
                static_cast<float>(metrics.confidenceIntervalUpper[dim]));
        }

        Serial.println("\nCovariance Matrix:");
        for (size_t i = 0; i < Dimension; ++i) {
            for (size_t j = 0; j < Dimension; ++j) {
                Serial.printf("%f ", static_cast<float>(computeCovariance(i, j)));
            }
            Serial.println();
        }
    }
};

// Exponential Detector
ExponentialDetector<float> expDetector;

// Statistical Banach Space with 3 dimensions
StatisticalBanachSpace<float, 3> statisticalSpace;

void setup() {
    Serial.begin(115200);
    
    // Simulated exponential and multi-dimensional data
    std::vector<float> exponentialData = {
        1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0
    };
    
    std::vector<std::vector<float>> statisticalMultiData = {
        {1.2, 2.4, 4.8, 9.6, 19.2},   // Dimension 1
        {3.5, 7.0, 14.0, 28.0, 56.0},  // Dimension 2
        {2.1, 4.2, 8.4, 16.8, 33.6}    // Dimension 3
    };
    
    // Add exponential data
    for (const auto& val : exponentialData) {
        expDetector.addDataPoint(val);
    }
    
    // Transpose and add statistical data points
    for (size_t i = 0; i < statisticalMultiData[0].size(); ++i) {
        std::vector<float> point = {
            statisticalMultiData[0][i],
            statisticalMultiData[1][i],
            statisticalMultiData[2][i]
        };
        statisticalSpace.addStatisticalDataPoint(point);
    }
}

void loop() {
    // Analyze Exponential Properties
    Serial.println("\n--- Exponential Detection ---");
    auto expCharacteristics = expDetector.analyzeExponentialProperties();
    
    Serial.printf("Base Exponent: %f\n", expCharacteristics.baseExponent);
    Serial.printf("Growth Rate: %f\n", expCharacteristics.growthRate);
    Serial.printf("R-Squared: %f\n", expCharacteristics.rSquared);
    Serial.printf("Inflection Point: %f\n", expCharacteristics.inflectionPoint);
    
    Serial.println("\nResiduals:");
    for (const auto& residual : expCharacteristics.residuals) {
        Serial.printf("%f ", residual);
    }
    Serial.println();

    // Perform Statistical Banach Space Analysis
    statisticalSpace.performStatisticalAnalysis();
    
    delay(5000);  // Analyze every 5 seconds
}
