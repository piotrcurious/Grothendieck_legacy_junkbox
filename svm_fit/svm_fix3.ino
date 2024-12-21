#include <Arduino.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Define candidate polynomial structures in PROGMEM
struct PolynomialCandidate {
    uint8_t degree;
    float coefficients[6];  // Max degree 5 + constant term
    uint8_t features;       // Bit flags for feature characteristics
};

// Store candidate polynomials in PROGMEM
const PROGMEM PolynomialCandidate candidatePolynomials[] = {
    // Linear trends
    {1, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 0b00000001},
    {1, {1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, 0b00000001},
    // Quadratic patterns
    {2, {0.0f, 1.0f, 0.5f, 0.0f, 0.0f, 0.0f}, 0b00000010},
    {2, {1.0f, -1.0f, 0.5f, 0.0f, 0.0f, 0.0f}, 0b00000010},
    // Oscillatory patterns
    {3, {0.0f, 1.0f, -0.5f, 0.1f, 0.0f, 0.0f}, 0b00000100},
    {3, {1.0f, -1.0f, 0.5f, -0.1f, 0.0f, 0.0f}, 0b00000100},
    // Exponential-like patterns
    {4, {0.0f, 1.0f, 0.5f, 0.25f, 0.125f, 0.0f}, 0b00001000},
    {4, {1.0f, 2.0f, 1.0f, 0.5f, 0.25f, 0.0f}, 0b00001000},
    // Complex patterns
    {5, {1.0f, -1.0f, 0.5f, -0.25f, 0.125f, -0.0625f}, 0b00010000}
};

const int NUM_CANDIDATES = sizeof(candidatePolynomials) / sizeof(PolynomialCandidate);

// Feature hashing class
class FeatureHasher {
private:
    static const int HASH_SIZE = 16;
    static const int WINDOW_SIZE = 10;
    
    // Murmurhash3 implementation for feature hashing
    uint32_t murmurhash3(const float* data, size_t len, uint32_t seed) {
        const uint32_t c1 = 0xcc9e2d51;
        const uint32_t c2 = 0x1b873593;
        uint32_t h1 = seed;
        
        const int nblocks = len / 4;
        const uint32_t* blocks = (const uint32_t*)data;
        
        for (int i = 0; i < nblocks; i++) {
            uint32_t k1 = blocks[i];
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            
            h1 ^= k1;
            h1 = (h1 << 13) | (h1 >> 19);
            h1 = h1 * 5 + 0xe6546b64;
        }
        
        h1 ^= len;
        h1 ^= h1 >> 16;
        h1 *= 0x85ebca6b;
        h1 ^= h1 >> 13;
        h1 *= 0xc2b2ae35;
        h1 ^= h1 >> 16;
        
        return h1;
    }
    
    // Extract features from time series
    void extractFeatures(const std::vector<DataPoint>& data, std::vector<float>& features) {
        if (data.size() < WINDOW_SIZE) return;
        
        // Calculate statistical features
        float mean = 0, variance = 0, slope = 0;
        float min_val = data[0].value, max_val = data[0].value;
        
        // First pass - mean and extremes
        for (int i = data.size() - WINDOW_SIZE; i < data.size(); i++) {
            mean += data[i].value;
            min_val = min(min_val, data[i].value);
            max_val = max(max_val, data[i].value);
        }
        mean /= WINDOW_SIZE;
        
        // Second pass - variance and trend
        for (int i = data.size() - WINDOW_SIZE; i < data.size(); i++) {
            float diff = data[i].value - mean;
            variance += diff * diff;
            slope += diff * (i - (data.size() - WINDOW_SIZE));
        }
        variance /= WINDOW_SIZE;
        slope /= WINDOW_SIZE;
        
        // Calculate zero crossings and turning points
        int zero_crossings = 0, turning_points = 0;
        for (int i = data.size() - WINDOW_SIZE + 1; i < data.size(); i++) {
            if ((data[i].value - mean) * (data[i-1].value - mean) < 0) {
                zero_crossings++;
            }
            if (i > data.size() - WINDOW_SIZE + 1) {
                float diff1 = data[i].value - data[i-1].value;
                float diff2 = data[i-1].value - data[i-2].value;
                if (diff1 * diff2 < 0) turning_points++;
            }
        }
        
        // Pack features
        features.clear();
        features.push_back(mean);
        features.push_back(sqrt(variance));
        features.push_back(slope);
        features.push_back(max_val - min_val);
        features.push_back(float(zero_crossings) / WINDOW_SIZE);
        features.push_back(float(turning_points) / WINDOW_SIZE);
    }

public:
    // Hash features to select appropriate polynomials
    uint32_t hashFeatures(const std::vector<DataPoint>& data) {
        std::vector<float> features;
        extractFeatures(data, features);
        
        if (features.empty()) return 0;
        
        return murmurhash3(features.data(), features.size() * sizeof(float), 0x1234567);
    }
    
    // Match features to candidate polynomials
    std::vector<int> matchCandidates(uint32_t featureHash) {
        std::vector<int> matches;
        uint32_t hash_bin = featureHash % HASH_SIZE;
        
        // Match candidates based on feature characteristics
        for (int i = 0; i < NUM_CANDIDATES; i++) {
            PolynomialCandidate candidate;
            memcpy_P(&candidate, &candidatePolynomials[i], sizeof(PolynomialCandidate));
            
            // Check if candidate matches feature hash bin
            if ((candidate.features & (1 << (hash_bin % 5))) != 0) {
                matches.push_back(i);
            }
        }
        
        return matches;
    }
};

// Enhanced PolynomialSVM class with auto-tuning
class PolynomialSVM {
    // Previous member variables...
    FeatureHasher featureHasher;
    std::vector<int> currentCandidates;
    float candidateScores[NUM_CANDIDATES];
    
    // New method: Update candidate scores based on prediction performance
    void updateCandidateScores(float predictionError) {
        const float LEARNING_RATE = 0.1f;
        
        for (int candidateIdx : currentCandidates) {
            float normalizedError = predictionError / adaptiveNoiseEstimate;
            float score = exp(-normalizedError * normalizedError / 2.0f);
            candidateScores[candidateIdx] = 
                (1 - LEARNING_RATE) * candidateScores[candidateIdx] + 
                LEARNING_RATE * score;
        }
    }
    
    // Enhanced method: Generate candidates using feature hashing
    std::vector<std::vector<float>> generateCandidates() {
        std::vector<std::vector<float>> candidates;
        
        // Get feature hash for current data
        uint32_t featureHash = featureHasher.hashFeatures(trainingData);
        
        // Match candidates based on features
        currentCandidates = featureHasher.matchCandidates(featureHash);
        
        // Add matched candidates from PROGMEM
        for (int idx : currentCandidates) {
            PolynomialCandidate candidate;
            memcpy_P(&candidate, &candidatePolynomials[idx], sizeof(PolynomialCandidate));
            
            std::vector<float> coeffs;
            for (int i = 0; i <= candidate.degree; i++) {
                coeffs.push_back(candidate.coefficients[i]);
            }
            
            candidates.push_back(coeffs);
        }
        
        // Add adaptive candidates based on current data
        candidates.push_back(fitMEstimator());
        candidates.push_back(fitLTS());
        
        return candidates;
    }
    
    // Enhanced method: Select best candidates based on scores
    std::vector<std::vector<float>> selectBestCandidates(
        const std::vector<std::vector<float>>& allCandidates) {
        std::vector<std::pair<float, int>> scoredCandidates;
        
        for (size_t i = 0; i < currentCandidates.size(); i++) {
            scoredCandidates.push_back({
                candidateScores[currentCandidates[i]], 
                i
            });
        }
        
        // Add scores for adaptive candidates
        for (size_t i = currentCandidates.size(); i < allCandidates.size(); i++) {
            scoredCandidates.push_back({0.5f, i}); // Neutral score for new candidates
        }
        
        // Sort by score
        std::sort(scoredCandidates.rbegin(), scoredCandidates.rend());
        
        // Select top candidates
        std::vector<std::vector<float>> selectedCandidates;
        for (int i = 0; i < min(3, (int)scoredCandidates.size()); i++) {
            selectedCandidates.push_back(allCandidates[scoredCandidates[i].second]);
        }
        
        return selectedCandidates;
    }
    
public:
    PolynomialSVM(float adcResolution) : 
        baseQuantizationError(adcResolution),
        adaptiveNoiseEstimate(adcResolution),
        timeScale(1.0f),
        valueScale(1.0f) {
        // Initialize candidate scores
        for (int i = 0; i < NUM_CANDIDATES; i++) {
            candidateScores[i] = 0.5f; // Neutral initial score
        }
        convergenceMetrics = {0, 0, 0, 1.0f, true, adcResolution};
    }
    
    // Enhanced method: Add new data point with auto-tuning
    void addDataPoint(unsigned long timestamp, float value) {
        // Calculate prediction before adding point
        float predicted = predict(timestamp);
        float predictionError = value - predicted;
        
        // Update candidate scores based on prediction performance
        updateCandidateScores(predictionError);
        
        // Normal data point processing...
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
        
        // Update system parameters
        updateScaling();
        adaptiveNoiseEstimate = estimateNoise();
        
        // Retrain with auto-tuned candidates
        if (trainingData.size() > MAX_DEGREE + 1 && 
            (!convergenceMetrics.is_converged || 
             abs(predictionError) > adaptiveNoiseEstimate * STABILITY_THRESHOLD)) {
            auto allCandidates = generateCandidates();
            auto selectedCandidates = selectBestCandidates(allCandidates);
            trainSVM(selectedCandidates);
        }
        
        updateConvergenceMetrics();
    }
    
    // New method: Get current candidate scores for diagnostics
    void getCandidateScores(float* scores) {
        memcpy(scores, candidateScores, NUM_CANDIDATES * sizeof(float));
    }
};

// Enhanced diagnostics...
void printDiagnostics() {
    auto metrics = svm->getConvergenceMetrics();
    float candidateScores[NUM_CANDIDATES];
    svm->getCandidateScores(candidateScores);
    
    Serial.println("\n=== System Diagnostics ===");
    Serial.println("Candidate Polynomial Scores:");
    for (int i = 0; i < NUM_CANDIDATES; i++) {
        Serial.printf("Candidate %d: %.3f\n", i, candidateScores[i]);
    }
    
    // Rest of diagnostics...
}

// Rest of the implementation remains the same...
