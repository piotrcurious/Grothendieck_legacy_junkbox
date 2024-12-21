#include <Arduino.h>
#include <vector>
#include <cmath>

// Structure for timestamp-value pairs
struct DataPoint {
    unsigned long timestamp;
    float value;
    float normalized_time;  // Normalized time for numerical stability
};

// Structure for polynomial support vector
struct PolynomialSV {
    std::vector<float> coefficients;
    float alpha;  // Lagrange multiplier
    int sign;     // +1 for above perfect fit, -1 for below
};

class PolynomialSVM {
private:
    static const int MAX_CANDIDATES = 5;    // Number of candidate polynomials
    static const int MAX_DEGREE = 3;        // Maximum polynomial degree
    static const int MAX_SAMPLES = 100;     // Maximum samples to store
    static const float C = 1.0;            // SVM regularization parameter
    
    std::vector<DataPoint> trainingData;
    std::vector<PolynomialSV> supportVectors;
    float timeScale;                       // For time normalization
    float valueScale;                      // For value normalization
    float quantizationError;
    float samplingNoise;

    // Evaluate polynomial at a point
    float evaluatePolynomial(const std::vector<float>& coeffs, float x) {
        float result = 0;
        float power = 1;
        for (float coeff : coeffs) {
            result += coeff * power;
            power *= x;
        }
        return result;
    }

    // Generate candidate polynomial coefficients using different fitting approaches
    std::vector<std::vector<float>> generateCandidates() {
        std::vector<std::vector<float>> candidates;
        
        // First candidate: Linear least squares
        std::vector<float> linear = fitLeastSquares(1);
        candidates.push_back(linear);
        
        // Second candidate: Quadratic least squares
        std::vector<float> quadratic = fitLeastSquares(2);
        candidates.push_back(quadratic);
        
        // Third candidate: Cubic least squares
        std::vector<float> cubic = fitLeastSquares(3);
        candidates.push_back(cubic);
        
        // Fourth candidate: Moving average based polynomial
        std::vector<float> movingAvg = fitMovingAveragePoly();
        candidates.push_back(movingAvg);
        
        // Fifth candidate: Robust fit (median-based)
        std::vector<float> robust = fitRobustPoly();
        candidates.push_back(robust);
        
        return candidates;
    }

    // Fit polynomial using least squares
    std::vector<float> fitLeastSquares(int degree) {
        std::vector<std::vector<float>> A;
        std::vector<float> b;
        
        // Build system of equations
        for (const auto& point : trainingData) {
            std::vector<float> row;
            float x = point.normalized_time;
            float power = 1;
            
            for (int i = 0; i <= degree; i++) {
                row.push_back(power);
                power *= x;
            }
            
            A.push_back(row);
            b.push_back(point.value);
        }
        
        // Solve using normal equations (A^T * A * x = A^T * b)
        // Note: In practice, you might want to use a more numerically stable method
        return solveNormalEquations(A, b);
    }

    // Fit polynomial using moving average approach
    std::vector<float> fitMovingAveragePoly() {
        // Implementation of moving average based polynomial fitting
        // This could use windowed data points to create local fits
        return std::vector<float>(MAX_DEGREE + 1, 0.0f);  // Placeholder
    }

    // Fit polynomial using robust estimation
    std::vector<float> fitRobustPoly() {
        // Implementation of robust polynomial fitting
        // This could use median regression or other robust methods
        return std::vector<float>(MAX_DEGREE + 1, 0.0f);  // Placeholder
    }

    // Calculate residual (distance from perfect fit)
    float calculateResidual(const DataPoint& point, const std::vector<float>& coeffs) {
        float predicted = evaluatePolynomial(coeffs, point.normalized_time);
        return point.value - predicted;
    }

    // Determine if a point is above or below the perfect fit
    int determineSign(const DataPoint& point, const std::vector<float>& coeffs) {
        float residual = calculateResidual(point, coeffs);
        return (residual > 0) ? 1 : -1;
    }

    // Train SVM using SMO algorithm adapted for polynomial support vectors
    void trainSVM(const std::vector<std::vector<float>>& candidates) {
        supportVectors.clear();
        
        // For each candidate polynomial
        for (const auto& candidate : candidates) {
            // Calculate residuals and signs for all points
            std::vector<int> signs;
            std::vector<float> residuals;
            
            for (const auto& point : trainingData) {
                signs.push_back(determineSign(point, candidate));
                residuals.push_back(std::abs(calculateResidual(point, candidate)));
            }
            
            // Simplified SMO algorithm
            float sumAlpha = 0;
            float margin = calculateMargin(residuals);
            
            if (margin > 0) {
                PolynomialSV sv;
                sv.coefficients = candidate;
                sv.alpha = margin;
                sv.sign = (sumAlpha >= 0) ? 1 : -1;
                supportVectors.push_back(sv);
            }
        }
    }

    // Calculate margin for SVM
    float calculateMargin(const std::vector<float>& residuals) {
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        
        // Calculate margin based on residuals and error distribution
        float margin = 0;
        for (float residual : residuals) {
            margin += exp(-pow(residual, 2) / (2 * pow(totalError, 2)));
        }
        return margin / residuals.size();
    }

    // Solve normal equations (simplified)
    std::vector<float> solveNormalEquations(const std::vector<std::vector<float>>& A, 
                                          const std::vector<float>& b) {
        // Simplified matrix solver - in practice, use a more robust method
        return std::vector<float>(A[0].size(), 0.0f);  // Placeholder
    }

public:
    PolynomialSVM(float adcResolution, float noiseEstimate) {
        quantizationError = adcResolution;
        samplingNoise = noiseEstimate;
        timeScale = 1.0;
        valueScale = 1.0;
    }

    // Add new data point and update model
    void addDataPoint(unsigned long timestamp, float value) {
        // Normalize time and value
        float normalized_time = (float)(timestamp) / timeScale;
        float normalized_value = value / valueScale;
        
        DataPoint point = {timestamp, value, normalized_time};
        
        if (trainingData.size() >= MAX_SAMPLES) {
            trainingData.erase(trainingData.begin());
        }
        trainingData.push_back(point);
        
        // Update scaling factors
        updateScaling();
        
        // Retrain if enough data
        if (trainingData.size() > MAX_DEGREE + 1) {
            auto candidates = generateCandidates();
            trainSVM(candidates);
        }
    }

    // Update scaling factors for numerical stability
    void updateScaling() {
        if (trainingData.empty()) return;
        
        // Update time scale
        unsigned long maxTime = 0;
        for (const auto& point : trainingData) {
            maxTime = max(maxTime, point.timestamp);
        }
        timeScale = (float)maxTime;
        
        // Update value scale
        float maxValue = 0;
        for (const auto& point : trainingData) {
            maxValue = max(maxValue, abs(point.value));
        }
        valueScale = maxValue > 0 ? maxValue : 1.0f;
    }

    // Predict value using support vector polynomials
    float predict(unsigned long timestamp) {
        float x = (float)timestamp / timeScale;
        float prediction = 0;
        float totalWeight = 0;
        
        for (const auto& sv : supportVectors) {
            float value = evaluatePolynomial(sv.coefficients, x);
            prediction += value * sv.alpha * sv.sign;
            totalWeight += abs(sv.alpha);
        }
        
        return totalWeight > 0 ? prediction / totalWeight : 0;
    }

    // Get prediction interval
    void getPredictionInterval(unsigned long timestamp, float& lower, float& upper) {
        float prediction = predict(timestamp);
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        
        lower = prediction - 2 * totalError;
        upper = prediction + 2 * totalError;
    }
};

// Example usage
PolynomialSVM* svm;

void setup() {
    Serial.begin(115200);
    
    float adcResolution = 3.3 / (1 << ADC_WIDTH);
    float estimatedNoise = 0.01;
    svm = new PolynomialSVM(adcResolution, estimatedNoise);
}

void loop() {
    int rawValue = analogRead(36);
    float voltage = (float)rawValue * (3.3 / (1 << ADC_WIDTH));
    unsigned long timestamp = millis();
    
    svm->addDataPoint(timestamp, voltage);
    
    unsigned long futureTime = timestamp + 1000;
    float prediction = svm->predict(futureTime);
    
    float lower, upper;
    svm->getPredictionInterval(futureTime, lower, upper);
    
    Serial.printf("Current Time: %lu, Value: %.3f\n", timestamp, voltage);
    Serial.printf("Prediction for %lu: %.3f (%.3f - %.3f)\n", 
                 futureTime, prediction, lower, upper);
    
    delay(100);
}
