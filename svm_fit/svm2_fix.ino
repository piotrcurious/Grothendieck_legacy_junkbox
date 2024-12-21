#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Previous DataPoint and PolynomialSV structures remain the same...

class PolynomialSVM {
    // Previous constant definitions and member variables remain the same...

private:
    // Solve normal equations using Cholesky decomposition for better stability
    std::vector<float> solveNormalEquations(const std::vector<std::vector<float>>& A, 
                                          const std::vector<float>& b) {
        int n = A[0].size();
        std::vector<std::vector<float>> ATA(n, std::vector<float>(n, 0.0f));
        std::vector<float> ATb(n, 0.0f);
        
        // Compute A^T * A and A^T * b
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                for (size_t k = 0; k < A.size(); k++) {
                    ATA[i][j] += A[k][i] * A[k][j];
                }
            }
            for (size_t k = 0; k < A.size(); k++) {
                ATb[i] += A[k][i] * b[k];
            }
        }
        
        // Cholesky decomposition: ATA = L * L^T
        std::vector<std::vector<float>> L(n, std::vector<float>(n, 0.0f));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                float sum = 0.0f;
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = sqrt(ATA[j][j] - sum);
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (ATA[i][j] - sum) / L[j][j];
                }
            }
        }
        
        // Forward substitution to solve L * y = A^T * b
        std::vector<float> y(n);
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < i; j++) {
                sum += L[i][j] * y[j];
            }
            y[i] = (ATb[i] - sum) / L[i][i];
        }
        
        // Back substitution to solve L^T * x = y
        std::vector<float> x(n);
        for (int i = n - 1; i >= 0; i--) {
            float sum = 0.0f;
            for (int j = i + 1; j < n; j++) {
                sum += L[j][i] * x[j];
            }
            x[i] = (y[i] - sum) / L[i][i];
        }
        
        return x;
    }

    // Fit polynomial using moving average with noise-aware windowing
    std::vector<float> fitMovingAveragePoly() {
        if (trainingData.size() < 3) return std::vector<float>(MAX_DEGREE + 1, 0.0f);
        
        // Calculate noise-based window size
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        
        int windowSize = max(3, min(15, (int)(1.0f / totalError)));
        
        std::vector<std::vector<float>> windows;
        std::vector<float> windowWeights;
        
        // Create overlapping windows
        for (size_t i = 0; i < trainingData.size() - windowSize + 1; i++) {
            std::vector<std::vector<float>> A;
            std::vector<float> b;
            
            // Build local system of equations
            for (int j = 0; j < windowSize; j++) {
                const auto& point = trainingData[i + j];
                std::vector<float> row;
                float x = point.normalized_time;
                float power = 1.0f;
                
                for (int k = 0; k <= min(2, MAX_DEGREE); k++) {
                    row.push_back(power);
                    power *= x;
                }
                
                A.push_back(row);
                b.push_back(point.value);
            }
            
            // Solve local polynomial
            auto coeffs = solveNormalEquations(A, b);
            windows.push_back(coeffs);
            
            // Calculate window weight based on fit quality
            float residualSum = 0;
            for (int j = 0; j < windowSize; j++) {
                float predicted = evaluatePolynomial(coeffs, trainingData[i + j].normalized_time);
                float residual = trainingData[i + j].value - predicted;
                residualSum += pow(residual, 2);
            }
            float weight = exp(-residualSum / (2 * pow(totalError, 2)));
            windowWeights.push_back(weight);
        }
        
        // Combine windows with weights
        std::vector<float> result(MAX_DEGREE + 1, 0.0f);
        float totalWeight = 0;
        
        for (size_t i = 0; i < windows.size(); i++) {
            for (size_t j = 0; j < windows[i].size(); j++) {
                result[j] += windows[i][j] * windowWeights[i];
            }
            totalWeight += windowWeights[i];
        }
        
        if (totalWeight > 0) {
            for (float& coeff : result) {
                coeff /= totalWeight;
            }
        }
        
        return result;
    }

    // Fit polynomial using robust estimation (Huber loss)
    std::vector<float> fitRobustPoly() {
        if (trainingData.size() < MAX_DEGREE + 1) {
            return std::vector<float>(MAX_DEGREE + 1, 0.0f);
        }
        
        // Calculate Huber loss threshold based on noise characteristics
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        float huberThreshold = 1.345f * totalError;  // Standard Huber threshold
        
        // Initialize coefficients with least squares solution
        std::vector<float> coeffs = fitLeastSquares(MAX_DEGREE);
        
        // Iterative reweighting for robust estimation
        const int MAX_ITER = 10;
        for (int iter = 0; iter < MAX_ITER; iter++) {
            std::vector<std::vector<float>> A;
            std::vector<float> b;
            std::vector<float> weights;
            
            // Calculate weights based on residuals
            for (const auto& point : trainingData) {
                float predicted = evaluatePolynomial(coeffs, point.normalized_time);
                float residual = abs(point.value - predicted);
                
                // Huber weight
                float weight = (residual < huberThreshold) ? 
                    1.0f : huberThreshold / residual;
                weights.push_back(weight);
            }
            
            // Build weighted system of equations
            for (size_t i = 0; i < trainingData.size(); i++) {
                std::vector<float> row;
                float x = trainingData[i].normalized_time;
                float power = 1.0f;
                
                for (int j = 0; j <= MAX_DEGREE; j++) {
                    row.push_back(power * weights[i]);
                    power *= x;
                }
                
                A.push_back(row);
                b.push_back(trainingData[i].value * weights[i]);
            }
            
            // Solve weighted system
            std::vector<float> newCoeffs = solveNormalEquations(A, b);
            
            // Check convergence
            float maxDiff = 0;
            for (size_t i = 0; i < coeffs.size(); i++) {
                maxDiff = max(maxDiff, abs(newCoeffs[i] - coeffs[i]));
            }
            
            coeffs = newCoeffs;
            if (maxDiff < totalError * 0.01f) break;
        }
        
        return coeffs;
    }

    // Improved trainSVM method with noise-based feature space division
    void trainSVM(const std::vector<std::vector<float>>& candidates) {
        supportVectors.clear();
        
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        
        // For each candidate polynomial
        for (const auto& candidate : candidates) {
            std::vector<float> residuals;
            std::vector<int> signs;
            float maxResidual = 0;
            
            // Calculate residuals and their statistics
            for (const auto& point : trainingData) {
                float residual = calculateResidual(point, candidate);
                residuals.push_back(residual);
                signs.push_back(residual > 0 ? 1 : -1);
                maxResidual = max(maxResidual, abs(residual));
            }
            
            // Calculate normalized residuals and cluster points
            float residualScale = maxResidual > 0 ? maxResidual : 1.0f;
            std::vector<float> normalizedResiduals;
            for (float residual : residuals) {
                normalizedResiduals.push_back(residual / residualScale);
            }
            
            // Calculate support vector weight based on noise characteristics
            float svWeight = 0;
            int consistentPoints = 0;
            
            for (size_t i = 0; i < residuals.size(); i++) {
                float normalizedError = abs(residuals[i]) / totalError;
                
                // Point is consistent if its error is within expected noise bounds
                if (normalizedError <= 3.0f) {  // 3-sigma rule
                    consistentPoints++;
                    svWeight += exp(-pow(normalizedError, 2) / 2.0f);
                }
            }
            
            // Accept candidate as support vector if it explains enough points
            if (consistentPoints > trainingData.size() / 3) {  // At least 1/3 of points
                PolynomialSV sv;
                sv.coefficients = candidate;
                sv.alpha = svWeight / residuals.size();  // Normalize weight
                sv.sign = (std::accumulate(signs.begin(), signs.end(), 0.0f) >= 0) ? 1 : -1;
                supportVectors.push_back(sv);
            }
        }
        
        // Normalize support vector weights
        float totalWeight = 0;
        for (const auto& sv : supportVectors) {
            totalWeight += abs(sv.alpha);
        }
        if (totalWeight > 0) {
            for (auto& sv : supportVectors) {
                sv.alpha /= totalWeight;
            }
        }
    }

    // Rest of the class implementation remains the same...
};

// setup() and loop() remain the same...
