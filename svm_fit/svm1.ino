#include <Arduino.h>
#include <vector>
#include <cmath>

// Structure to hold timestamp-value pairs
struct DataPoint {
    unsigned long timestamp;
    float value;
};

// Structure for polynomial coefficients
struct Polynomial {
    std::vector<float> coefficients;
    float bias;
};

class PolynomialSVM {
private:
    static const int MAX_DEGREE = 3;  // Maximum polynomial degree
    static const int MAX_SAMPLES = 100;  // Maximum number of samples to store
    static const float LEARNING_RATE = 0.01;
    static const float LAMBDA = 0.1;  // Regularization parameter
    static const float GAUSSIAN_SIGMA = 0.1;  // Width of Gaussian kernel
    
    std::vector<DataPoint> trainingData;
    std::vector<Polynomial> supportVectors;
    float quantizationError;  // ADC quantization error
    float samplingNoise;      // Estimated sampling noise
    
    // Calculate Gaussian kernel between two points
    float gaussianKernel(const DataPoint& x1, const DataPoint& x2) {
        float diff = x1.value - x2.value;
        return exp(-(diff * diff) / (2 * GAUSSIAN_SIGMA * GAUSSIAN_SIGMA));
    }
    
    // Calculate polynomial features for a given point
    std::vector<float> calculateFeatures(const DataPoint& point) {
        std::vector<float> features;
        float x = (float)(point.timestamp) / 1000.0;  // Convert to seconds
        
        for (int degree = 0; degree <= MAX_DEGREE; degree++) {
            features.push_back(pow(x, degree));
        }
        return features;
    }
    
    // Calculate error distribution
    float calculateErrorDistribution(float measured, float predicted) {
        float totalError = sqrt(
            pow(quantizationError, 2) +  // ADC quantization
            pow(samplingNoise, 2) +      // Sampling noise
            pow(0.5 * ADC_WIDTH, 2)      // Input quantization (assuming ADC_WIDTH bits)
        );
        
        float error = measured - predicted;
        return exp(-(error * error) / (2 * totalError * totalError));
    }

public:
    PolynomialSVM(float adcResolution, float noiseEstimate) {
        quantizationError = adcResolution;
        samplingNoise = noiseEstimate;
    }
    
    // Add new data point to training set
    void addDataPoint(unsigned long timestamp, float value) {
        DataPoint point = {timestamp, value};
        
        if (trainingData.size() >= MAX_SAMPLES) {
            trainingData.erase(trainingData.begin());
        }
        trainingData.push_back(point);
        
        // Retrain model if we have enough data
        if (trainingData.size() > MAX_DEGREE + 1) {
            train();
        }
    }
    
    // Train the SVM model
    void train() {
        supportVectors.clear();
        
        // Implement Sequential Minimal Optimization (SMO) algorithm
        for (int i = 0; i < trainingData.size(); i++) {
            Polynomial sv;
            sv.coefficients = calculateFeatures(trainingData[i]);
            sv.bias = 0;
            
            for (int j = 0; j < trainingData.size(); j++) {
                float kernel = gaussianKernel(trainingData[i], trainingData[j]);
                float error = calculateErrorDistribution(
                    trainingData[j].value,
                    predict(trainingData[j].timestamp)
                );
                
                sv.bias += LEARNING_RATE * kernel * error;
            }
            
            // Add support vector if significant
            if (abs(sv.bias) > 1e-5) {
                supportVectors.push_back(sv);
            }
        }
    }
    
    // Predict value for a given timestamp
    float predict(unsigned long timestamp) {
        DataPoint point = {timestamp, 0};
        float prediction = 0;
        
        for (const auto& sv : supportVectors) {
            float feature_sum = 0;
            auto features = calculateFeatures(point);
            
            for (size_t i = 0; i < features.size(); i++) {
                feature_sum += sv.coefficients[i] * features[i];
            }
            
            prediction += sv.bias * feature_sum;
        }
        
        return prediction;
    }
    
    // Get confidence interval for prediction
    void getPredictionInterval(unsigned long timestamp, float& lower, float& upper) {
        float prediction = predict(timestamp);
        float totalError = sqrt(
            pow(quantizationError, 2) +
            pow(samplingNoise, 2) +
            pow(0.5 * ADC_WIDTH, 2)
        );
        
        // 95% confidence interval (approximately 2 standard deviations)
        lower = prediction - 2 * totalError;
        upper = prediction + 2 * totalError;
    }
};

// Example usage in setup() and loop()
PolynomialSVM* svm;

void setup() {
    Serial.begin(115200);
    
    // Initialize SVM with ADC resolution and estimated noise
    float adcResolution = 3.3 / (1 << ADC_WIDTH);  // Assuming 3.3V reference
    float estimatedNoise = 0.01;  // Estimated noise in volts
    svm = new PolynomialSVM(adcResolution, estimatedNoise);
}

void loop() {
    // Example: Read analog value
    int rawValue = analogRead(36);  // GPIO36 (ADC1_CH0)
    float voltage = (float)rawValue * (3.3 / (1 << ADC_WIDTH));
    unsigned long timestamp = millis();
    
    // Add data point to model
    svm->addDataPoint(timestamp, voltage);
    
    // Make prediction for near future
    unsigned long futureTime = timestamp + 1000;  // Predict 1 second ahead
    float prediction = svm->predict(futureTime);
    
    // Get confidence interval
    float lower, upper;
    svm->getPredictionInterval(futureTime, lower, upper);
    
    // Print results
    Serial.printf("Current Time: %lu, Value: %.3f\n", timestamp, voltage);
    Serial.printf("Prediction for %lu: %.3f (%.3f - %.3f)\n", 
                 futureTime, prediction, lower, upper);
    
    delay(100);  // Adjust sampling rate as needed
}
