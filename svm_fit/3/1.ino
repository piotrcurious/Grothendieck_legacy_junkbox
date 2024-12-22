#include <Eigen.h>
#include <Eigen/Dense>
#include <vector>
#include <queue>

// Constants for system configuration
const int MAX_POLYNOMIAL_DEGREE = 6;
const int MIN_DATA_POINTS = 10;
const double DIVISION_QUALITY_THRESHOLD = 0.85;
const double RESIDUAL_THRESHOLD = 0.1;
const int PREDICTION_HORIZON = 10;

// Structure to hold timestamp-value pairs
struct DataPoint {
    double timestamp;
    double value;
    DataPoint(double t, double v) : timestamp(t), value(v) {}
};

class PolynomialFitter {
private:
    std::vector<DataPoint> dataPoints;
    std::vector<double> coefficients;
    int currentDegree;
    double lastDivisionQuality;
    
    // Circular buffer for recent residuals
    std::queue<double> recentResiduals;
    const int RESIDUAL_BUFFER_SIZE = 20;

public:
    PolynomialFitter() : currentDegree(2), lastDivisionQuality(1.0) {}

    // Add new data point and update model if necessary
    void addDataPoint(double timestamp, double value) {
        dataPoints.emplace_back(timestamp, value);
        
        if (dataPoints.size() >= MIN_DATA_POINTS) {
            updateModel();
        }
    }

    // Fit polynomial using least squares method
    void fitPolynomial() {
        int n = dataPoints.size();
        Eigen::MatrixXd A(n, currentDegree + 1);
        Eigen::VectorXd b(n);

        // Build matrices for least squares
        for (int i = 0; i < n; i++) {
            double x = dataPoints[i].timestamp;
            for (int j = 0; j <= currentDegree; j++) {
                A(i, j) = pow(x, j);
            }
            b(i) = dataPoints[i].value;
        }

        // Solve least squares problem
        Eigen::VectorXd solution = A.colPivHouseholderQr().solve(b);
        
        // Update coefficients
        coefficients.resize(currentDegree + 1);
        for (int i = 0; i <= currentDegree; i++) {
            coefficients[i] = solution(i);
        }
    }

    // Calculate polynomial value at given timestamp
    double evaluatePolynomial(double timestamp) {
        double result = 0.0;
        for (int i = 0; i <= currentDegree; i++) {
            result += coefficients[i] * pow(timestamp, i);
        }
        return result;
    }

    // Calculate division quality using SVM-inspired approach
    double calculateDivisionQuality() {
        if (dataPoints.size() < MIN_DATA_POINTS) return 1.0;

        std::vector<double> abovePolynomial;
        std::vector<double> belowPolynomial;

        // Separate points into two sets based on polynomial
        for (const auto& point : dataPoints) {
            double polyValue = evaluatePolynomial(point.timestamp);
            double distance = point.value - polyValue;
            if (distance > 0) {
                abovePolynomial.push_back(distance);
            } else {
                belowPolynomial.push_back(-distance);
            }
        }

        // Calculate separation quality
        double meanAbove = calculateMean(abovePolynomial);
        double meanBelow = calculateMean(belowPolynomial);
        double stdAbove = calculateStd(abovePolynomial, meanAbove);
        double stdBelow = calculateStd(belowPolynomial, meanBelow);

        // Quality metric based on separation and spread
        return 1.0 / (1.0 + stdAbove + stdBelow);
    }

    // Update model based on new data
    void updateModel() {
        fitPolynomial();
        double newQuality = calculateDivisionQuality();
        
        // Check if model quality has degraded
        if (newQuality < lastDivisionQuality * DIVISION_QUALITY_THRESHOLD && 
            currentDegree < MAX_POLYNOMIAL_DEGREE) {
            currentDegree++;
            fitPolynomial();
            newQuality = calculateDivisionQuality();
        }
        
        lastDivisionQuality = newQuality;
        updateResiduals();
    }

    // Update and analyze residuals
    void updateResiduals() {
        if (dataPoints.empty()) return;
        
        // Calculate latest residual
        const auto& lastPoint = dataPoints.back();
        double predicted = evaluatePolynomial(lastPoint.timestamp);
        double residual = abs(lastPoint.value - predicted);
        
        // Update residual buffer
        if (recentResiduals.size() >= RESIDUAL_BUFFER_SIZE) {
            recentResiduals.pop();
        }
        recentResiduals.push(residual);
    }

    // Predict future values
    std::vector<DataPoint> predictFuture() {
        if (dataPoints.empty()) return {};
        
        std::vector<DataPoint> predictions;
        double lastTimestamp = dataPoints.back().timestamp;
        
        for (int i = 1; i <= PREDICTION_HORIZON; i++) {
            double futureTimestamp = lastTimestamp + i;
            double predictedValue = evaluatePolynomial(futureTimestamp);
            predictions.emplace_back(futureTimestamp, predictedValue);
        }
        
        return predictions;
    }

private:
    // Utility functions for statistical calculations
    double calculateMean(const std::vector<double>& values) {
        if (values.empty()) return 0.0;
        double sum = 0.0;
        for (double value : values) sum += value;
        return sum / values.size();
    }

    double calculateStd(const std::vector<double>& values, double mean) {
        if (values.empty()) return 0.0;
        double sumSquares = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquares += diff * diff;
        }
        return sqrt(sumSquares / values.size());
    }
};

// Global instance of polynomial fitter
PolynomialFitter polyFitter;

void setup() {
    Serial.begin(115200);
}

void loop() {
    // Example of reading sensor data
    if (Serial.available()) {
        // Format: "timestamp,value"
        String input = Serial.readStringUntil('\n');
        int commaIndex = input.indexOf(',');
        if (commaIndex > 0) {
            double timestamp = input.substring(0, commaIndex).toDouble();
            double value = input.substring(commaIndex + 1).toDouble();
            
            // Add new data point
            polyFitter.addDataPoint(timestamp, value);
            
            // Get predictions
            auto predictions = polyFitter.predictFuture();
            
            // Output predictions
            Serial.println("Predictions:");
            for (const auto& pred : predictions) {
                Serial.print(pred.timestamp);
                Serial.print(",");
                Serial.println(pred.value);
            }
        }
    }
    
    delay(100); // Adjust delay as needed
}
