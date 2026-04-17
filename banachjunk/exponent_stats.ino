#include <Arduino.h>
#include <vector>
#include <cmath>
#include <complex>
#include <algorithm>

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

    // Statistical Descriptors
    struct StatisticalDescriptors {
        std::vector<T> mean;
        std::vector<T> variance;
        std::vector<T> skewness;
        std::vector<T> kurtosis;
        std::vector<T> entropy;
    };

    // Compute Comprehensive Statistical Metrics
    StatisticalDescriptors computeStatisticalMetrics() {
        StatisticalDescriptors metrics;
        metrics.mean.resize(Dimension, 0);
        metrics.variance.resize(Dimension, 0);
        metrics.skewness.resize(Dimension, 0);
        metrics.kurtosis.resize(Dimension, 0);
        metrics.entropy.resize(Dimension, 0);

        // Compute for each dimension
        for (size_t dim = 0; dim < Dimension; ++dim) {
            const auto& dimData = statisticalData[dim];
            
            // Mean
            T sum = std::accumulate(dimData.begin(), dimData.end(), T(0));
            metrics.mean[dim] = sum / dimData.size();

            // Variance and Higher Moments
            T variance = 0, m3 = 0, m4 = 0;
            for (const auto& val : dimData) {
                T diff = val - metrics.mean[dim];
                T diffSq = diff * diff;
                variance += diffSq;
                m3 += diff * diffSq;
                m4 += diffSq * diffSq;
            }
            
            variance /= dimData.size();
            metrics.variance[dim] = variance;

            // Skewness
            T stdDev = std::sqrt(variance);
            metrics.skewness[dim] = (abs(stdDev) > 1e-9) ? (m3 / dimData.size()) / std::pow(stdDev, 3) : 0;

            // Kurtosis
            metrics.kurtosis[dim] = (abs(variance) > 1e-9) ? (m4 / dimData.size()) / std::pow(variance, 2) - 3 : 0;

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
        }

        return metrics;
    }

public:
    // Add multi-dimensional statistical data point
    void addStatisticalDataPoint(const std::vector<T>& point) {
        if (point.size() != Dimension) {
            // Handle dimension mismatch
            return;
        }
        
        // Expand or create dimensions as needed
        if (statisticalData.size() < Dimension) {
            statisticalData.resize(Dimension);
        }
        
        // Add point to each dimension
        for (size_t i = 0; i < Dimension; ++i) {
            statisticalData[i].push_back(point[i]);
            
            // Maintain fixed buffer size
            if (statisticalData[i].size() > 100) {
                statisticalData[i].erase(statisticalData[i].begin());
            }
        }
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
            Serial.printf("  Variance: %f\n", static_cast<float>(metrics.variance[dim]));
            Serial.printf("  Skewness: %f\n", static_cast<float>(metrics.skewness[dim]));
            Serial.printf("  Kurtosis: %f\n", static_cast<float>(metrics.kurtosis[dim]));
            Serial.printf("  Entropy: %f\n", static_cast<float>(metrics.entropy[dim]));
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
