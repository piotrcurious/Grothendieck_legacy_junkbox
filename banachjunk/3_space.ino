#include <Arduino.h>
#include <vector>
#include <cmath>
#include <complex>
#include "math_utils.h"

// Abstract Banach Space Representation
template <typename T, size_t Dimension>
class BanachSpace {
private:
    // Multi-dimensional numerical representation
    std::vector<std::vector<T>> dimensionalData;
    std::vector<T> timestamps;
    
    // Metric space properties
    struct MetricProperties {
        T normedDistance;
        T completenessIndex;
        T dimensionalCoherence;
        std::vector<T> spectralFlatness;
        std::vector<T> sparsity;
        std::vector<std::vector<T>> instantaneousCoherence;
        T phaseSpaceArea;
        T averageCurvature;
    };

    // Projection and Transformation Operators
    class TransformationOperators {
    public:
        // Linear transformation between numerical spaces
        static std::vector<T> linearProjection(
            const std::vector<T>& input, 
            const std::vector<std::vector<T>>& transformationMatrix
        ) {
            std::vector<T> result(transformationMatrix.size(), 0);
            
            for (size_t i = 0; i < transformationMatrix.size(); ++i) {
                for (size_t j = 0; j < input.size(); ++j) {
                    result[i] += transformationMatrix[i][j] * input[j];
                }
            }
            
            return result;
        }

        // Non-linear transformation using complex analysis
        static std::complex<T> complexProjection(
            const std::vector<T>& input
        ) {
            T realPart = 0, imagPart = 0;
            for (size_t i = 0; i < input.size(); ++i) {
                realPart += input[i] * std::cos(input[i]);
                imagPart += input[i] * std::sin(input[i]);
            }
            return std::complex<T>(realPart, imagPart);
        }
    };

    // Compute Banach Space Metrics
    MetricProperties computeMetrics() {
        MetricProperties metrics;
        
        // Compute Lp norms for different p
        metrics.normedDistance = computeLpNorm(2);  // L2 norm
        metrics.completenessIndex = computeCompleteness();
        metrics.dimensionalCoherence = computeDimensionalCoherence();
        metrics.spectralFlatness = computeSpectralFlatness();
        metrics.sparsity = computeSparsity();
        metrics.instantaneousCoherence = computeInstantaneousCoherence(5);
        metrics.phaseSpaceArea = computePhaseSpaceArea();
        metrics.averageCurvature = computeAverageCurvature();
        
        return metrics;
    }

public:
    // L-p Norm Computation using Lebesgue-style measure weighting
    T computeLpNorm(int p) {
        return static_cast<T>(banach::Statistics::calculateLpNorm(dimensionalData, timestamps, p));
    }

    // Completeness Metric using a Lebesgue-measure-aware Cauchy-like convergence proxy
    T computeCompleteness() {
        if (dimensionalData.empty() || timestamps.size() < 2) return T(0);
        T totalMeasure = timestamps.back() - timestamps.front();
        if (std::abs(totalMeasure) < 1e-9) return T(0);
        
        T completenessScore = 0;
        for (const auto& dimension : dimensionalData) {
            if (dimension.size() < 2) continue;

            // Analyze the decay of differences weighted by their time intervals
            T cauchySum = 0;
            for (size_t i = 1; i < dimension.size(); ++i) {
                T dt = timestamps[i] - timestamps[i-1];
                if (dt <= 0) continue;
                // Normalize by time index to check for late-sequence stability
                cauchySum += (std::abs(dimension[i] - dimension[i-1]) / dt) / static_cast<T>(i);
            }
            completenessScore += 1.0 / (1.0 + cauchySum / totalMeasure);
        }
        
        return completenessScore / dimensionalData.size();
    }

    // Dimensional Coherence
    T computeDimensionalCoherence() {
        if (dimensionalData.size() < 2) return T(0);
        
        // Compute correlation between dimensions
        T coherenceScore = 0;
        for (size_t i = 0; i < dimensionalData.size() - 1; ++i) {
            for (size_t j = i + 1; j < dimensionalData.size(); ++j) {
                coherenceScore += computeDimensionCorrelation(
                    dimensionalData[i], 
                    dimensionalData[j]
                );
            }
        }
        
        return coherenceScore;
    }

    // Instantaneous Coherence between dimensions using a sliding window
    std::vector<std::vector<T>> computeInstantaneousCoherence(size_t windowSize = 5) {
        if (dimensionalData.size() < 2) return {};
        size_t n = dimensionalData[0].size();
        if (n < windowSize) return {};

        std::vector<std::vector<T>> matrix(Dimension, std::vector<T>(Dimension, 1.0));
        for (size_t i = 0; i < Dimension; ++i) {
            for (size_t j = i + 1; j < Dimension; ++j) {
                T score = static_cast<T>(banach::Statistics::calculateCoherence(dimensionalData[i], dimensionalData[j], n - windowSize, windowSize));
                matrix[i][j] = matrix[j][i] = score;
            }
        }
        return matrix;
    }

    // Phase Space Area: Sum of cross-products between consecutive 2D projections (Dim 0, Dim 1)
    T computePhaseSpaceArea() {
        if (Dimension < 2 || dimensionalData[0].size() < 2) return 0;
        T area = 0;
        const auto& d0 = dimensionalData[0];
        const auto& d1 = dimensionalData[1];
        for (size_t i = 1; i < d0.size(); ++i) {
            // Shoelace-like incremental area in phase space
            area += (d0[i-1] * d1[i] - d0[i] * d1[i-1]);
        }
        return std::abs(area) / 2.0;
    }

    // Menger Curvature Proxy: 1/R for circumcircle of 3 consecutive points in projection
    T computeAverageCurvature() {
        if (Dimension < 2 || dimensionalData[0].size() < 3) return 0;
        T totalCurv = 0;
        const auto& d0 = dimensionalData[0];
        const auto& d1 = dimensionalData[1];
        int count = 0;
        for (size_t i = 1; i < d0.size() - 1; ++i) {
            // Triangle sides
            T a = std::sqrt(std::pow(d0[i]-d0[i-1], 2) + std::pow(d1[i]-d1[i-1], 2));
            T b = std::sqrt(std::pow(d0[i+1]-d0[i], 2) + std::pow(d1[i+1]-d1[i], 2));
            T c = std::sqrt(std::pow(d0[i+1]-d0[i-1], 2) + std::pow(d1[i+1]-d1[i-1], 2));
            T area = 0.5 * std::abs(d0[i-1]*(d1[i]-d1[i+1]) + d0[i]*(d1[i+1]-d1[i-1]) + d0[i+1]*(d1[i]-d1[i]));
            if (a*b*c > 1e-9) {
                totalCurv += 4.0 * area / (a*b*c);
                count++;
            }
        }
        return (count > 0) ? totalCurv / count : 0;
    }

    // Spectral Flatness (Wiener Entropy proxy) using Lebesgue-weighted geometric/arithmetic means
    std::vector<T> computeSpectralFlatness() {
        std::vector<T> flatness(Dimension, 0);
        for (size_t dim = 0; dim < Dimension; ++dim) {
            flatness[dim] = static_cast<T>(banach::Statistics::calculateFlatness(dimensionalData[dim], timestamps));
        }
        return flatness;
    }

    std::vector<T> computeSparsity() {
        std::vector<T> sparsity(Dimension, 0);
        for (size_t dim = 0; dim < Dimension; ++dim) {
            sparsity[dim] = static_cast<T>(banach::Statistics::calculateSparsity(dimensionalData[dim]));
        }
        return sparsity;
    }

    // Correlation between Dimensions
    T computeDimensionCorrelation(
        const std::vector<T>& dim1, 
        const std::vector<T>& dim2
    ) {
        // Compute Pearson correlation coefficient
        if (dim1.size() != dim2.size() || dim1.empty()) return T(0);
        
        T mean1 = 0, mean2 = 0;
        for (size_t i = 0; i < dim1.size(); ++i) {
            mean1 += dim1[i];
            mean2 += dim2[i];
        }
        mean1 /= dim1.size();
        mean2 /= dim2.size();
        
        T numerator = 0, denominator1 = 0, denominator2 = 0;
        for (size_t i = 0; i < dim1.size(); ++i) {
            T diff1 = dim1[i] - mean1;
            T diff2 = dim2[i] - mean2;
            
            numerator += diff1 * diff2;
            denominator1 += diff1 * diff1;
            denominator2 += diff2 * diff2;
        }
        
        T denom = std::sqrt(denominator1 * denominator2);
        if (std::abs(denom) < 1e-9) return T(0);

        return numerator / denom;
    }

public:
    // Clear data buffers
    void reset() {
        dimensionalData.clear();
        timestamps.clear();
    }

    // Add multi-dimensional data point with timestamp
    void addDataPoint(const std::vector<T>& point, T timestamp = -1.0) {
        if (point.size() != Dimension) return;

        if (dimensionalData.size() < Dimension) dimensionalData.resize(Dimension);

        // Auto-increment timestamp if not provided
        if (timestamp < 0) {
            timestamp = timestamps.empty() ? 0 : timestamps.back() + 1.0;
        }
        
        // Buffer management: For small N=100, vector::erase(begin) is acceptable on ESP32,
        // but we ensure consistent size across dimensions.
        timestamps.push_back(timestamp);
        if (timestamps.size() > 100) timestamps.erase(timestamps.begin());

        for (size_t i = 0; i < Dimension; ++i) {
            dimensionalData[i].push_back(point[i]);
            if (dimensionalData[i].size() > 100) {
                dimensionalData[i].erase(dimensionalData[i].begin());
            }
        }
    }

    // Comprehensive Banach Space Analysis
    void performSpaceAnalysis() {
        if (dimensionalData.empty()) return;
        
        // Compute Metrics
        MetricProperties metrics = computeMetrics();
        
        // Projection Demonstrations
        std::vector<T> sampleDimension = dimensionalData[0];
        
        // Linear Projection using a more descriptive transformation matrix
        // We project the dimension onto a lower space that highlights differences
        std::vector<std::vector<T>> transformationMatrix(Dimension);
        for (size_t r = 0; r < Dimension; ++r) {
            transformationMatrix[r].resize(sampleDimension.size());
            for (size_t c = 0; c < sampleDimension.size(); ++c) {
                // Mix of weighted average and finite difference patterns
                transformationMatrix[r][c] = (r == 0) ? (1.0 / sampleDimension.size()) :
                                            ((c % (r + 1) == 0) ? 1.0 : -1.0);
            }
        }

        std::vector<T> linearProjectedSpace = 
            TransformationOperators::linearProjection(
                sampleDimension, 
                transformationMatrix
            );
        
        // Complex Projection
        std::complex<T> complexProjection = 
            TransformationOperators::complexProjection(sampleDimension);

        // Output Analysis
        Serial.println("\n--- Banach Space Analysis ---");
        
        Serial.printf("Normed Distance (L2): %f\n", 
            static_cast<float>(metrics.normedDistance));
        Serial.printf("Completeness Index: %f\n", 
            static_cast<float>(metrics.completenessIndex));
        Serial.printf("Dimensional Coherence: %f\n", 
            static_cast<float>(metrics.dimensionalCoherence));
        Serial.printf("Phase Space Area (D0 x D1): %f\n",
            static_cast<float>(metrics.phaseSpaceArea));
        Serial.printf("Average Trajectory Curvature: %f\n",
            static_cast<float>(metrics.averageCurvature));

        Serial.println("\nSpectral Flatness (per dimension):");
        for (size_t i = 0; i < Dimension; ++i) {
            Serial.printf("Dim %zu: %f\n", i, static_cast<float>(metrics.spectralFlatness[i]));
        }

        Serial.println("\nSparsity (Hoyer Metric):");
        for (size_t i = 0; i < Dimension; ++i) {
            Serial.printf("Dim %zu: %f\n", i, static_cast<float>(metrics.sparsity[i]));
        }
        
        Serial.println("\nInstantaneous Coherence (Last Window):");
        for (size_t i = 0; i < metrics.instantaneousCoherence.size(); ++i) {
            for (size_t j = 0; j < metrics.instantaneousCoherence[i].size(); ++j) {
                Serial.printf("%f ", static_cast<float>(metrics.instantaneousCoherence[i][j]));
            }
            Serial.println();
        }

        Serial.println("\nLinear Projection:");
        for (const auto& val : linearProjectedSpace) {
            Serial.printf("%f ", static_cast<float>(val));
        }
        Serial.println();
        
        Serial.printf("\nComplex Projection: %f + %fi\n", 
            complexProjection.real(), 
            complexProjection.imag()
        );
    }
};

// Instantiate Banach Space with float type and 3 dimensions
BanachSpace<float, 3> numericalSpace;

void setup() {
    Serial.begin(115200);
    
    // Simulated multi-dimensional data
    std::vector<std::vector<float>> simulatedData = {
        {1.2, 2.4, 4.8, 9.6, 19.2},   // Dimension 1
        {3.5, 7.0, 14.0, 28.0, 56.0},  // Dimension 2
        {2.1, 4.2, 8.4, 16.8, 33.6}    // Dimension 3
    };
    
    // Transpose and add data points
    for (size_t i = 0; i < simulatedData[0].size(); ++i) {
        std::vector<float> point = {
            simulatedData[0][i],
            simulatedData[1][i],
            simulatedData[2][i]
        };
        numericalSpace.addDataPoint(point);
    }
}

void loop() {
    // Perform Banach space analysis periodically
    numericalSpace.performSpaceAnalysis();
    
    delay(5000);  // Analyze every 5 seconds
}
