// Signal Feature Detection using Banach Space Analysis
// For ESP32 Arduino
// Uses functional analysis concepts to detect features in time-series data

struct DataPoint {
  float timestamp;
  float value;
};

// Parameters for feature detection
const int WINDOW_SIZE = 20;  // Analysis window size
const float EPSILON = 0.01;  // Convergence threshold
const int MAX_FEATURES = 10; // Maximum number of features to detect

class BanachFeatureDetector {
private:
  // Store the norm history for convergence checking
  float previousNorm = 0;
  
  // Calculate L1 norm (Manhattan distance) using Lebesgue-style weighting
  float calculateL1Norm(const DataPoint* signal, int startIdx, int size) {
    float integral = 0;
    float totalDt = 0;
    for (int i = 0; i < size - 1; i++) {
      float dt = signal[startIdx + i + 1].timestamp - signal[startIdx + i].timestamp;
      if (dt <= 0) continue;
      integral += (abs(signal[startIdx + i].value) + abs(signal[startIdx + i + 1].value)) / 2.0 * dt;
      totalDt += dt;
    }
    return (totalDt > 1e-9) ? (integral / totalDt) * size : 0;
  }
  
  // Calculate L2 norm (Euclidean distance) using Lebesgue-style weighting
  float calculateL2Norm(const DataPoint* signal, int startIdx, int size) {
    float integral = 0;
    float totalDt = 0;
    for (int i = 0; i < size - 1; i++) {
      float dt = signal[startIdx + i + 1].timestamp - signal[startIdx + i].timestamp;
      if (dt <= 0) continue;
      float v1 = signal[startIdx + i].value;
      float v2 = signal[startIdx + i + 1].value;
      integral += (v1 * v1 + v2 * v2) / 2.0 * dt;
      totalDt += dt;
    }
    return (totalDt > 1e-9) ? sqrt(integral / totalDt) * sqrt(size) : 0;
  }
  
  // Calculate Lp norm for p=infinity (Maximum norm)
  float calculateLInfNorm(float* vector, int size) {
    float maxVal = 0;
    for (int i = 0; i < size; i++) {
      maxVal = max(maxVal, abs(vector[i]));
    }
    return maxVal;
  }
  
  // Check if sequence converges in Banach space
  bool checkConvergence(float currentNorm) {
    bool converged = abs(currentNorm - previousNorm) < EPSILON;
    previousNorm = currentNorm;
    return converged;
  }

  // Calculate Spectral Flatness (Wiener entropy proxy)
  float calculateFlatness(const DataPoint* signal, int startIdx, int size) {
    float logSum = 0, arithSum = 0, totalDt = 0;
    for(int i=0; i<size-1; ++i) {
        float dt = signal[startIdx+i+1].timestamp - signal[startIdx+i].timestamp;
        if(dt <= 0) continue;
        float val = (abs(signal[startIdx+i].value) + abs(signal[startIdx+i+1].value)) / 2.0f;
        if(val < 1e-6) val = 1e-6;
        logSum += log(val) * dt;
        arithSum += val * dt;
        totalDt += dt;
    }
    if(totalDt < 1e-9 || arithSum < 1e-9) return 0;
    return exp(logSum / totalDt) / (arithSum / totalDt);
  }

  // Hurst Exponent estimation (R/S analysis)
  float calculateHurst(const DataPoint* signal, int startIdx, int size) {
    float mean = 0, totalDt = 0;
    for(int i=0; i<size-1; ++i) {
        float dt = signal[startIdx+i+1].timestamp - signal[startIdx+i].timestamp;
        if(dt <= 0) continue;
        mean += (signal[startIdx+i].value + signal[startIdx+i+1].value) / 2.0f * dt;
        totalDt += dt;
    }
    if(totalDt < 1e-9) return 0.5;
    mean /= totalDt;

    float cumSum = 0, minZ = 1e30, maxZ = -1e30, sqSum = 0;
    for(int i=0; i<size; ++i) {
        float val = signal[startIdx+i].value;
        cumSum += (val - mean);
        if(cumSum < minZ) minZ = cumSum;
        if(cumSum > maxZ) maxZ = cumSum;
        sqSum += (val - mean) * (val - mean);
    }
    float sdev = sqrt(sqSum / size);
    if(sdev < 1e-9) return 0.5;
    float rs = (maxZ - minZ) / sdev;
    if(rs <= 0) return 0.5;
    return log(rs) / log(size);
  }
  
public:
  struct Feature {
    float timestamp;
    float value;
    float strength;
    String type;
  };
  
  // Detect features in a window of signal data
  int detectFeatures(const DataPoint* signal, int signalSize, Feature* features) {
    int featureCount = 0;
    float windowValues[WINDOW_SIZE];
    
    // Slide window through signal
    for (int i = 0; i <= signalSize - WINDOW_SIZE && featureCount < MAX_FEATURES; i++) {
      // Fill analysis window (still needed for L-inf and compatibility)
      for (int j = 0; j < WINDOW_SIZE; j++) {
        windowValues[j] = signal[i + j].value;
      }
      
      // Calculate different norms using Lebesgue weighting
      float l1Norm = calculateL1Norm(signal, i, WINDOW_SIZE);
      float l2Norm = calculateL2Norm(signal, i, WINDOW_SIZE);
      float lInfNorm = calculateLInfNorm(windowValues, WINDOW_SIZE);
      
      // Complexity gating: ignore high-entropy noise
      float flatness = calculateFlatness(signal, i, WINDOW_SIZE);
      float hurst = calculateHurst(signal, i, WINDOW_SIZE);

      // Feature detection criteria using Banach space properties
      // Gated by complexity: stationary noise (flatness -> 1, hurst -> 0.5) is ignored
      bool isSignificant = (l2Norm > 0.1) && (flatness < 0.9 || abs(hurst - 0.5) > 0.2);

      if (checkConvergence(l2Norm) && isSignificant) {
        // Analyze relationship between norms to classify feature
        float l2Norm_scaled = l2Norm * sqrt(WINDOW_SIZE);
        float normRatio = (l2Norm_scaled > 1e-9) ? (l1Norm / l2Norm_scaled) : 0;
        
        Feature newFeature;
        newFeature.timestamp = signal[i + WINDOW_SIZE/2].timestamp;
        newFeature.value = signal[i + WINDOW_SIZE/2].value;
        newFeature.strength = l2Norm;
        
        // Improved feature classification
        if (l2Norm > 1e-9 && lInfNorm / l2Norm > 0.8) {
          newFeature.type = "spike";
        } else if (normRatio > 1.2) {
          newFeature.type = "step";
        } else if (normRatio < 0.6 && normRatio > 0) {
          newFeature.type = "oscillation";
        } else {
          newFeature.type = "transition";
        }
        
        // Add feature if it's significantly different from previous
        bool significantlyDifferent = (featureCount == 0);
        if (!significantlyDifferent) {
            const auto& last = features[featureCount-1];
            significantlyDifferent = (abs(last.strength - newFeature.strength) > EPSILON) ||
                                     (last.type != newFeature.type) ||
                                     (abs(last.value - newFeature.value) > 0.1);
        }

        if (significantlyDifferent) {
            features[featureCount++] = newFeature;
        }
        
        // Skip overlapping windows
        i += WINDOW_SIZE/2;
      }
    }
    
    return featureCount;
  }
};

// Example usage
void setup() {
  Serial.begin(115200);
  
  // Example signal data
  DataPoint signalData[] = {
    {0.0, 0.1}, {0.1, 0.15}, {0.2, 0.12}, {0.3, 0.8},  // Step
    {0.4, 0.85}, {0.5, 0.82}, {0.6, 0.81}, {0.7, 0.79}, 
    {0.8, 0.2}, {0.9, 0.18}, {1.0, 1.5}, {1.1, 0.2},   // Spike
    {1.2, 0.22}, {1.3, 0.19}, {1.4, 0.21}, {1.5, 0.18},
    {1.6, 0.3}, {1.7, -0.3}, {1.8, 0.3}, {1.9, -0.3},  // Oscillation
    {2.0, 0.3}, {2.1, -0.3}, {2.2, 0.3}, {2.3, -0.3}
  };
  
  BanachFeatureDetector detector;
  BanachFeatureDetector::Feature features[MAX_FEATURES];
  
  int numFeatures = detector.detectFeatures(signalData, 24, features);
  
  // Print detected features
  Serial.println("Detected Features:");
  for (int i = 0; i < numFeatures; i++) {
    Serial.print("Time: ");
    Serial.print(features[i].timestamp);
    Serial.print(", Value: ");
    Serial.print(features[i].value);
    Serial.print(", Type: ");
    Serial.print(features[i].type);
    Serial.print(", Strength: ");
    Serial.println(features[i].strength);
  }
}

void loop() {
  // Main loop empty for this example
}
