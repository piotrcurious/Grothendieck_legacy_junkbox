#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

// Extended DataPoint structure to include prediction metrics
struct DataPoint {
    unsigned long timestamp;
    float value;
    float normalized_time;
    float prediction_error;  // Store prediction error for convergence monitoring
    float estimated_noise;   // Local noise estimate
};

// Extended PolynomialSV structure
struct PolynomialSV {
    std::vector<float> coefficients;
    float alpha;
    int sign;
    float fit_quality;      // Measure of fit quality
    float stability_score;  // Measure of prediction stability
};

// New structure for convergence monitoring
struct ConvergenceMetrics {
    float short_term_error;
    float long_term_error;
    float error_trend;
    float stability_score;
    bool is_converged;
    float noise_estimate;
};

class PolynomialSVM {
private:
    static const int MAX_CANDIDATES = 5;
    static const int MAX_DEGREE = 3;
    static const int MAX_SAMPLES = 100;
    static const float C = 1.0;
    
    // New parameters for adaptive estimation
    static const int SHORT_WINDOW = 10;     // Window for short-term metrics
    static const int LONG_WINDOW = 30;      // Window for long-term metrics
    static const float STABILITY_THRESHOLD = 2.0f;  // Threshold for stability detection
    
    std::vector<DataPoint> trainingData;
    std::vector<PolynomialSV> supportVectors;
    float timeScale;
    float valueScale;
    float baseQuantizationError;  // Base ADC quantization error
    float adaptiveNoiseEstimate;  // Adaptive noise estimate
    ConvergenceMetrics convergenceMetrics;
    
    // New method: Estimate noise using Multiple methods
    float estimateNoise() {
        if (trainingData.size() < 4) return baseQuantizationError;
        
        std::vector<float> noiseEstimates;
        
        // Method 1: Median Absolute Deviation (MAD)
        {
            std::vector<float> residuals;
            for (const auto& point : trainingData) {
                float predicted = predict(point.timestamp);
                residuals.push_back(abs(point.value - predicted));
            }
            std::sort(residuals.begin(), residuals.end());
            float mad = residuals[residuals.size() / 2] / 0.6745f; // MAD to sigma conversion
            noiseEstimates.push_back(mad);
        }
        
        // Method 2: Difference-based estimation
        {
            std::vector<float> differences;
            for (size_t i = 1; i < trainingData.size(); i++) {
                differences.push_back(abs(trainingData[i].value - trainingData[i-1].value));
            }
            float rms_diff = 0;
            for (float diff : differences) rms_diff += diff * diff;
            rms_diff = sqrt(rms_diff / differences.size()) / sqrt(2.0f);
            noiseEstimates.push_back(rms_diff);
        }
        
        // Method 3: Robust scale estimation
        {
            std::vector<float> pairwise_diffs;
            for (size_t i = 1; i < trainingData.size(); i++) {
                for (size_t j = 0; j < i; j++) {
                    float diff = abs(trainingData[i].value - trainingData[j].value);
                    pairwise_diffs.push_back(diff);
                }
            }
            std::sort(pairwise_diffs.begin(), pairwise_diffs.end());
            float q1 = pairwise_diffs[pairwise_diffs.size() / 4];
            float q3 = pairwise_diffs[3 * pairwise_diffs.size() / 4];
            float iqr = q3 - q1;
            noiseEstimates.push_back(iqr / 2.0f);
        }
        
        // Combine estimates using median
        std::sort(noiseEstimates.begin(), noiseEstimates.end());
        return max(noiseEstimates[noiseEstimates.size() / 2], baseQuantizationError);
    }
    
    // New method: MM-estimator for robust fitting
    std::vector<float> fitMEstimator() {
        if (trainingData.size() < MAX_DEGREE + 1) {
            return std::vector<float>(MAX_DEGREE + 1, 0.0f);
        }
        
        // Initial S-estimate using LTS (Least Trimmed Squares)
        auto initialEstimate = fitLTS();
        float scale = calculateScale(initialEstimate);
        
        // MM-estimation iteration
        std::vector<float> coeffs = initialEstimate;
        const int MAX_ITER = 20;
        float prevObjective = 1e10;
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            std::vector<std::vector<float>> A;
            std::vector<float> b;
            std::vector<float> weights;
            float objective = 0;
            
            // Calculate Tukey's biweight
            for (const auto& point : trainingData) {
                float residual = abs(point.value - evaluatePolynomial(coeffs, point.normalized_time));
                float u = residual / (4.685f * scale);
                float weight;
                
                if (abs(u) <= 1) {
                    float temp = 1 - u * u;
                    weight = temp * temp;
                } else {
                    weight = 0;
                }
                
                objective += (residual * residual * weight);
                weights.push_back(weight);
            }
            
            // Check convergence
            if (abs(objective - prevObjective) < adaptiveNoiseEstimate * 0.001f) break;
            prevObjective = objective;
            
            // Weighted least squares iteration
            for (size_t i = 0; i < trainingData.size(); i++) {
                if (weights[i] > 0) {
                    std::vector<float> row;
                    float x = trainingData[i].normalized_time;
                    float power = 1.0f;
                    
                    for (int j = 0; j <= MAX_DEGREE; j++) {
                        row.push_back(power * sqrt(weights[i]));
                        power *= x;
                    }
                    
                    A.push_back(row);
                    b.push_back(trainingData[i].value * sqrt(weights[i]));
                }
            }
            
            coeffs = solveNormalEquations(A, b);
        }
        
        return coeffs;
    }
    
    // New method: Least Trimmed Squares
    std::vector<float> fitLTS() {
        const int h = trainingData.size() * 0.75; // Use 75% of points
        std::vector<float> bestCoeffs(MAX_DEGREE + 1, 0.0f);
        float minError = 1e10;
        
        // Multiple random starts
        for (int start = 0; start < 10; start++) {
            // Random subset selection
            std::vector<int> indices(trainingData.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());
            
            std::vector<std::vector<float>> A;
            std::vector<float> b;
            
            for (int i = 0; i < h; i++) {
                std::vector<float> row;
                float x = trainingData[indices[i]].normalized_time;
                float power = 1.0f;
                
                for (int j = 0; j <= MAX_DEGREE; j++) {
                    row.push_back(power);
                    power *= x;
                }
                
                A.push_back(row);
                b.push_back(trainingData[indices[i]].value);
            }
            
            auto coeffs = solveNormalEquations(A, b);
            
            // Calculate trimmed error
            std::vector<float> errors;
            for (const auto& point : trainingData) {
                float predicted = evaluatePolynomial(coeffs, point.normalized_time);
                errors.push_back((point.value - predicted) * (point.value - predicted));
            }
            
            std::sort(errors.begin(), errors.end());
            float trimmedError = std::accumulate(errors.begin(), errors.begin() + h, 0.0f);
            
            if (trimmedError < minError) {
                minError = trimmedError;
                bestCoeffs = coeffs;
            }
        }
        
        return bestCoeffs;
    }
    
    // New method: Calculate robust scale estimate
    float calculateScale(const std::vector<float>& coeffs) {
        std::vector<float> residuals;
        for (const auto& point : trainingData) {
            float predicted = evaluatePolynomial(coeffs, point.normalized_time);
            residuals.push_back(abs(point.value - predicted));
        }
        
        std::sort(residuals.begin(), residuals.end());
        return residuals[residuals.size() / 2] / 0.6745f;
    }
    
    // Enhanced method: Monitor convergence
    void updateConvergenceMetrics() {
        if (trainingData.size() < SHORT_WINDOW) return;
        
        // Calculate short-term error
        float shortTermSum = 0;
        for (int i = trainingData.size() - SHORT_WINDOW; i < trainingData.size(); i++) {
            shortTermSum += pow(trainingData[i].prediction_error, 2);
        }
        convergenceMetrics.short_term_error = sqrt(shortTermSum / SHORT_WINDOW);
        
        // Calculate long-term error if possible
        if (trainingData.size() >= LONG_WINDOW) {
            float longTermSum = 0;
            for (int i = trainingData.size() - LONG_WINDOW; i < trainingData.size(); i++) {
                longTermSum += pow(trainingData[i].prediction_error, 2);
            }
            convergenceMetrics.long_term_error = sqrt(longTermSum / LONG_WINDOW);
        }
        
        // Calculate error trend
        convergenceMetrics.error_trend = (convergenceMetrics.short_term_error / 
                                        (convergenceMetrics.long_term_error + 1e-6)) - 1.0f;
        
        // Update stability score
        float maxError = adaptiveNoiseEstimate * STABILITY_THRESHOLD;
        convergenceMetrics.stability_score = exp(-pow(convergenceMetrics.error_trend, 2) / 
                                               pow(maxError, 2));
        
        // Update convergence status
        convergenceMetrics.is_converged = 
            (convergenceMetrics.stability_score > 0.8f) &&
            (convergenceMetrics.short_term_error < adaptiveNoiseEstimate * 3.0f);
        
        // Update noise estimate
        convergenceMetrics.noise_estimate = adaptiveNoiseEstimate;
    }
    
    // Enhanced method: Generate candidate polynomials
    std::vector<std::vector<float>> generateCandidates() {
        std::vector<std::vector<float>> candidates;
        
        // Standard least squares
        candidates.push_back(fitLeastSquares(MAX_DEGREE));
        
        // Robust M-estimator
        candidates.push_back(fitMEstimator());
        
        // LTS estimator
        candidates.push_back(fitLTS());
        
        // Moving average polynomial
        candidates.push_back(fitMovingAveragePoly());
        
        // Adaptive degree polynomial based on noise
        int adaptiveDegree = min(MAX_DEGREE, 
                                (int)(1.0f / (adaptiveNoiseEstimate + 1e-6)));
        candidates.push_back(fitLeastSquares(adaptiveDegree));
        
        return candidates;
    }
    
    // Previous methods remain but are enhanced with noise estimation...

public:
    PolynomialSVM(float adcResolution) : 
        baseQuantizationError(adcResolution),
        adaptiveNoiseEstimate(adcResolution),
        timeScale(1.0f),
        valueScale(1.0f) {
        convergenceMetrics = {0, 0, 0, 1.0f, true, adcResolution};
    }
    
    // Enhanced method: Add new data point with convergence monitoring
    void addDataPoint(unsigned long timestamp, float value) {
        // Calculate prediction before adding point
        float predicted = predict(timestamp);
        float predictionError = value - predicted;
        
        // Normalize and store point
        float normalized_time = (float)(timestamp) / timeScale;
        DataPoint point = {
            timestamp, 
            value, 
            normalized_time,
            predictionError,
            estimateNoise()
        };
        
        if (trainingData.size() >= MAX_SAMPLES) {
            trainingData.erase(trainingData.begin());
        }
        trainingData.push_back(point);
        
        // Update scaling and noise estimate
        updateScaling();
        adaptiveNoiseEstimate = estimateNoise();
        
        // Retrain if enough data or convergence lost
        if (trainingData.size() > MAX_DEGREE + 1 && 
            (!convergenceMetrics.is_converged || 
             abs(predictionError) > adaptiveNoiseEstimate * STABILITY_THRESHOLD)) {
            auto candidates = generateCandidates();
            trainSVM(candidates);
        }
        
        // Update convergence metrics
        updateConvergenceMetrics();
    }
    
    // New method: Get convergence status
    ConvergenceMetrics getConvergenceMetrics() const {
        return convergenceMetrics;
    }
    
    // Previous methods remain the same...
};

// Enhanced setup() and loop() implementation
PolynomialSVM* svm;

void setup() {
    Serial.begin(115200);
    float adcResolution = 3.3f / (1 << ADC_WIDTH);
    svm = new PolynomialSVM(adcResolution);
}

void loop() {
    int rawValue = analogRead(36);
    float voltage = (float)rawValue * (3.3f / (1 << ADC_WIDTH));
    unsigned long timestamp = millis();
    
    svm->addDataPoint(timestamp, voltage);
    
    unsigned long futureTime = timestamp + 1000;
    float prediction = svm->predict(futureTime);
    
    float lower, upper;
    svm->getPredictionInterval(futureTime, lower, upper);
    
    auto metrics = svm->getConvergenceMetrics();
    
    Serial.printf("Time: %lu, Value: %.3f, Pred: %.3f (%.3f - %.3f)\n", 
                 timestamp, voltage, prediction, lower, upper);
    Serial.printf("Convergence: %s, Noise: %.3f, Stability: %.3f\n",
                 metrics.is_converged ? "Yes" : "No",
                 metrics.noise_estimate,
                 metrics.stability_score);
    
    delay(100);
    
    // Add diagnostics every second
    static unsigned long lastDiagnosticTime = 0;
    if (timestamp - lastDiagnosticTime >= 1000) {
        printDiagnostics();
        lastDiagnosticTime = timestamp;
    }
}

// New diagnostic functions
void printDiagnostics() {
    auto metrics = svm->getConvergenceMetrics();
    
    Serial.println("\n=== System Diagnostics ===");
    Serial.printf("Short-term Error: %.3f\n", metrics.short_term_error);
    Serial.printf("Long-term Error: %.3f\n", metrics.long_term_error);
    Serial.printf("Error Trend: %.3f\n", metrics.error_trend);
    Serial.printf("Stability Score: %.3f\n", metrics.stability_score);
    Serial.printf("Noise Estimate: %.3f\n", metrics.noise_estimate);
    Serial.printf("System State: %s\n", 
                 metrics.is_converged ? "Converged" : "Adapting");
    
    // Add warning messages based on system state
    if (!metrics.is_converged) {
        if (metrics.error_trend > 0.5) {
            Serial.println("WARNING: Rapid error growth detected");
        }
        if (metrics.stability_score < 0.3) {
            Serial.println("WARNING: System stability compromised");
        }
        if (metrics.short_term_error > metrics.noise_estimate * 5.0) {
            Serial.println("WARNING: Prediction error exceeds noise bounds");
        }
    }
    
    Serial.println("========================\n");
}

// Add support for data logging to analyze system performance
class DataLogger {
private:
    static const int LOG_BUFFER_SIZE = 100;
    struct LogEntry {
        unsigned long timestamp;
        float actual;
        float predicted;
        float noise_estimate;
        float stability_score;
        bool converged;
    };
    
    std::vector<LogEntry> logBuffer;
    
public:
    DataLogger() {
        logBuffer.reserve(LOG_BUFFER_SIZE);
    }
    
    void addEntry(unsigned long timestamp, float actual, float predicted,
                 const ConvergenceMetrics& metrics) {
        LogEntry entry = {
            timestamp,
            actual,
            predicted,
            metrics.noise_estimate,
            metrics.stability_score,
            metrics.is_converged
        };
        
        if (logBuffer.size() >= LOG_BUFFER_SIZE) {
            logBuffer.erase(logBuffer.begin());
        }
        logBuffer.push_back(entry);
    }
    
    void printSummary() {
        if (logBuffer.empty()) return;
        
        Serial.println("\n=== Performance Summary ===");
        
        // Calculate statistics
        float avgError = 0;
        float maxError = 0;
        int convergenceCount = 0;
        float avgNoiseEstimate = 0;
        float avgStabilityScore = 0;
        
        for (const auto& entry : logBuffer) {
            float error = abs(entry.actual - entry.predicted);
            avgError += error;
            maxError = max(maxError, error);
            if (entry.converged) convergenceCount++;
            avgNoiseEstimate += entry.noise_estimate;
            avgStabilityScore += entry.stability_score;
        }
        
        avgError /= logBuffer.size();
        avgNoiseEstimate /= logBuffer.size();
        avgStabilityScore /= logBuffer.size();
        
        Serial.printf("Average Error: %.3f\n", avgError);
        Serial.printf("Maximum Error: %.3f\n", maxError);
        Serial.printf("Convergence Rate: %.1f%%\n", 
                     100.0f * convergenceCount / logBuffer.size());
        Serial.printf("Average Noise: %.3f\n", avgNoiseEstimate);
        Serial.printf("Average Stability: %.3f\n", avgStabilityScore);
        Serial.println("=========================\n");
    }
};

// Create global instances
PolynomialSVM* svm;
DataLogger* logger;

void setup() {
    Serial.begin(115200);
    float adcResolution = 3.3f / (1 << ADC_WIDTH);
    svm = new PolynomialSVM(adcResolution);
    logger = new DataLogger();
    
    // Configure ADC for best resolution
    analogReadResolution(ADC_WIDTH);
    analogSetAttenuation(ADC_11db);  // Full 0-3.3V range
    
    Serial.println("System initialized");
    Serial.printf("ADC Resolution: %.3f V\n", adcResolution);
    Serial.println("Starting data collection...\n");
}

void loop() {
    int rawValue = analogRead(36);
    float voltage = (float)rawValue * (3.3f / (1 << ADC_WIDTH));
    unsigned long timestamp = millis();
    
    svm->addDataPoint(timestamp, voltage);
    
    unsigned long futureTime = timestamp + 1000;
    float prediction = svm->predict(futureTime);
    
    float lower, upper;
    svm->getPredictionInterval(futureTime, lower, upper);
    
    auto metrics = svm->getConvergenceMetrics();
    
    // Log data
    logger->addEntry(timestamp, voltage, prediction, metrics);
    
    // Basic status output
    Serial.printf("Time: %lu, Value: %.3f, Pred: %.3f (%.3f - %.3f)\n", 
                 timestamp, voltage, prediction, lower, upper);
    
    // Periodic detailed diagnostics
    static unsigned long lastDiagnosticTime = 0;
    if (timestamp - lastDiagnosticTime >= 5000) {  // Every 5 seconds
        printDiagnostics();
        logger->printSummary();
        lastDiagnosticTime = timestamp;
    }
    
    // Add warning for significant prediction deviation
    if (abs(voltage - prediction) > metrics.noise_estimate * 3.0) {
        Serial.printf("WARNING: Significant deviation detected: %.3f > %.3f\n",
                     abs(voltage - prediction), metrics.noise_estimate * 3.0);
    }
    
    delay(100);
}
