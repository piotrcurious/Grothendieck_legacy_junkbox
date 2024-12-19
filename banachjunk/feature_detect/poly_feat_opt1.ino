// Memory-Optimized Signal Feature Detection using Computed Values
// For ESP32 Arduino

#include <vector>
#include <memory>
#include <optional>
#include <cmath>
#include <algorithm>
#include <bitset>

// Configuration using constexpr for compile-time optimization
namespace Config {
    constexpr int WINDOW_SIZE = 20;
    constexpr float EPSILON = 0.01f;
    constexpr int MAX_FEATURES = 10;
    constexpr int FIELD_PRIME = 251;
    constexpr int MAX_POLY_DEGREE = 4;
    constexpr size_t BUFFER_SIZE = 64;  // Reduced from 1024
    constexpr float MIN_SIGNIFICANCE = 0.1f;
    constexpr int NUM_SAMPLES_FOR_NOISE = 20;  // Reduced from 50
}

// Optimized data structures using bit fields
struct DataPoint {
    float timestamp;
    float value;
    
    DataPoint(float t = 0.0f, float v = 0.0f) : timestamp(t), value(v) {}
};

struct Feature {
    float timestamp;
    float value;
    uint8_t polynomialDegree : 3;  // Max degree is 4, needs 3 bits
    float algebraicComplexity;
    uint8_t typeIndex : 4;         // Up to 16 different types
    float confidence;
    // Store only essential coefficients
    std::array<float, Config::MAX_POLY_DEGREE + 1> coefficients;
    
    static const char* getTypeString(uint8_t index) {
        static const char* types[] = {
            "linear", "quadratic_vertex", "quadratic_transition",
            "periodic", "constant_piecewise", "complex_nonlinear",
            "polynomial", "unknown"
        };
        return types[std::min(index, uint8_t(7))];
    }
};

// Memory-efficient circular buffer using std::array
template<typename T, size_t Size>
class CircularBuffer {
private:
    std::array<T, Size> buffer;
    uint8_t head = 0;
    uint8_t tail = 0;
    bool full = false;

public:
    void push(const T& item) {
        buffer[head] = item;
        if (full) tail = (tail + 1) % Size;
        head = (head + 1) % Size;
        full = head == tail;
    }

    std::optional<T> pop() {
        if (empty()) return std::nullopt;
        T item = buffer[tail];
        full = false;
        tail = (tail + 1) % Size;
        return item;
    }

    bool empty() const { return !full && (head == tail); }
    bool full() const { return full; }
    size_t size() const {
        if (full) return Size;
        return (head - tail) % Size;
    }
};

// Optimized Galois Field implementation using computed values
class GaloisField {
private:
    const int prime;
    
    // Fast modular exponentiation
    uint32_t modPow(uint32_t base, uint32_t exp, uint32_t modulus) const {
        uint32_t result = 1;
        base %= modulus;
        while (exp > 0) {
            if (exp & 1) result = (result * base) % modulus;
            base = (base * base) % modulus;
            exp >>= 1;
        }
        return result;
    }
    
    // Extended Euclidean Algorithm for modular multiplicative inverse
    int modInverse(int a) const {
        int m = prime;
        int y = 0, x = 1;
        
        if (m == 1) return 0;
        
        while (a > 1) {
            int q = a / m;
            int t = m;
            m = a % m;
            a = t;
            t = y;
            y = x - q * y;
            x = t;
        }
        
        return x < 0 ? x + prime : x;
    }

public:
    explicit GaloisField(int p = Config::FIELD_PRIME) : prime(p) {
        if (!isPrime(p)) {
            throw std::runtime_error("Field order must be prime");
        }
    }

    bool isPrime(int n) const {
        if (n < 2) return false;
        if (n == 2) return true;
        if (n % 2 == 0) return false;
        
        for (int i = 3; i * i <= n; i += 2) {
            if (n % i == 0) return false;
        }
        return true;
    }

    uint8_t add(uint8_t a, uint8_t b) const {
        return (a + b) % prime;
    }

    uint8_t subtract(uint8_t a, uint8_t b) const {
        return (a + prime - b) % prime;
    }

    uint8_t multiply(uint8_t a, uint8_t b) const {
        return (static_cast<uint32_t>(a) * b) % prime;
    }

    uint8_t divide(uint8_t a, uint8_t b) const {
        if (b == 0) throw std::runtime_error("Division by zero");
        return multiply(a, modInverse(b));
    }

    uint8_t power(uint8_t base, int exp) const {
        if (exp < 0) {
            if (base == 0) throw std::runtime_error("Zero cannot be raised to negative power");
            return power(modInverse(base), -exp);
        }
        return modPow(base, exp, prime);
    }
};

// Memory-efficient polynomial implementation
class GFPolynomial {
private:
    std::array<uint8_t, Config::MAX_POLY_DEGREE + 1> coefficients;
    uint8_t degree;
    const GaloisField& gf;

public:
    GFPolynomial(const std::vector<uint8_t>& coeffs, const GaloisField& field)
        : gf(field), degree(0) {
        std::fill(coefficients.begin(), coefficients.end(), 0);
        for (size_t i = 0; i < std::min(coeffs.size(), coefficients.size()); i++) {
            coefficients[i] = coeffs[i];
        }
        normalize();
    }

    void normalize() {
        degree = 0;
        for (int i = Config::MAX_POLY_DEGREE; i >= 0; i--) {
            if (coefficients[i] != 0) {
                degree = i;
                break;
            }
        }
    }

    uint8_t evaluate(uint8_t x) const {
        // Horner's method for polynomial evaluation
        uint8_t result = coefficients[degree];
        for (int i = degree - 1; i >= 0; i--) {
            result = gf.add(gf.multiply(result, x), coefficients[i]);
        }
        return result;
    }

    GFPolynomial derivative() const {
        std::vector<uint8_t> derivCoeffs;
        for (uint8_t i = 1; i <= degree; i++) {
            derivCoeffs.push_back(gf.multiply(coefficients[i], i));
        }
        return GFPolynomial(derivCoeffs, gf);
    }

    uint8_t getDegree() const { return degree; }
    
    const auto& getCoefficients() const { return coefficients; }
};

// Optimized feature detector
class AlgebraicFeatureDetector {
private:
    GaloisField gf;
    CircularBuffer<DataPoint, Config::BUFFER_SIZE> buffer;
    float noiseLevel;
    std::array<Feature, Config::MAX_FEATURES> previousFeatures;
    uint8_t numPreviousFeatures;

    // Optimized float to GF conversion using bit manipulation
    uint8_t floatToGF(float value) {
        int32_t scaled = static_cast<int32_t>((value + 1.0f) * ((Config::FIELD_PRIME - 1) >> 1));
        return static_cast<uint8_t>(std::clamp(scaled, 0, Config::FIELD_PRIME - 1));
    }

    float gfToFloat(uint8_t value) {
        return (2.0f * value / (Config::FIELD_PRIME - 1.0f)) - 1.0f;
    }

    // Memory-efficient polynomial fitting
    GFPolynomial fitPolynomial(const uint8_t* x, const uint8_t* y, size_t n) {
        std::vector<uint8_t> coeffs(std::min(n, static_cast<size_t>(Config::MAX_POLY_DEGREE + 1)), 0);
        
        for (size_t i = 0; i < coeffs.size(); i++) {
            uint8_t term = y[i];
            for (size_t j = 0; j < coeffs.size(); j++) {
                if (i != j) {
                    uint8_t diff = gf.subtract(x[i], x[j]);
                    if (diff == 0) continue;
                    term = gf.multiply(term, gf.multiply(x[j], gf.power(diff, Config::FIELD_PRIME - 2)));
                }
            }
            coeffs[i] = term;
        }
        
        return GFPolynomial(coeffs, gf);
    }

    // Optimized variety dimension calculation using bitset
    int calculateVarietyDimension(const uint8_t* values, size_t n) {
        std::bitset<Config::FIELD_PRIME> seen;
        for (size_t i = 0; i < n; i++) {
            seen.set(values[i]);
        }
        return seen.count();
    }

    // Efficient noise estimation using moving median
    float estimateNoiseLevel(const DataPoint* data, size_t n) {
        if (n < Config::NUM_SAMPLES_FOR_NOISE) return 0.0f;
        
        std::array<float, Config::NUM_SAMPLES_FOR_NOISE> diffs;
        for (size_t i = 1; i < n && i < Config::NUM_SAMPLES_FOR_NOISE; i++) {
            diffs[i-1] = std::abs(data[i].value - data[i-1].value);
        }
        
        size_t mid = Config::NUM_SAMPLES_FOR_NOISE / 2;
        std::nth_element(diffs.begin(), diffs.begin() + mid, diffs.begin() + n - 1);
        return diffs[mid] * 1.4826f;
    }

public:
    AlgebraicFeatureDetector() : gf(Config::FIELD_PRIME), noiseLevel(0.0f), numPreviousFeatures(0) {}

    void processDataPoint(const DataPoint& point) {
        buffer.push(point);
        if (buffer.size() >= Config::WINDOW_SIZE) {
            detectAndUpdateFeatures();
        }
    }

 /*   // Main feature detection function (continued in next section)
    std::vector<Feature> detectAndUpdateFeatures() {
        // Implementation continues...

[Note: Would you like me to continue with the remaining optimized implementation? The code includes significant memory optimizations through:
1. Use of bit fields and compact data structures
2. Computed values instead of lookup tables
3. Efficient algorithms for mathematical operations
4. Stack-based arrays instead of heap allocations
5. Optimized buffer sizes
6. Bit manipulation techniques

I can continue with the rest of the implementation if you'd like to see more.]
*/

std::vector<Feature> detectAndUpdateFeatures() {
        std::vector<Feature> newFeatures;
        std::array<DataPoint, Config::WINDOW_SIZE> windowData;
        std::array<uint8_t, Config::WINDOW_SIZE> windowX;
        std::array<uint8_t, Config::WINDOW_SIZE> windowY;
        size_t windowSize = 0;

        // Fill window arrays using stack memory
        for (size_t i = 0; i < Config::WINDOW_SIZE; i++) {
            auto point = buffer.pop();
            if (!point) break;
            windowData[i] = *point;
            windowX[i] = floatToGF(point->timestamp);
            windowY[i] = floatToGF(point->value);
            buffer.push(*point);
            windowSize++;
        }

        if (windowSize < Config::WINDOW_SIZE) return newFeatures;

        try {
            // Update noise estimate using stack-based calculation
            noiseLevel = estimateNoiseLevel(windowData.data(), windowSize);

            // Fit polynomial using optimized implementation
            GFPolynomial fitted = fitPolynomial(windowX.data(), windowY.data(), windowSize);
            int varietyDim = calculateVarietyDimension(windowY.data(), windowSize);

            Feature newFeature;
            newFeature.timestamp = windowData[Config::WINDOW_SIZE/2].timestamp;
            newFeature.value = windowData[Config::WINDOW_SIZE/2].value;
            newFeature.polynomialDegree = fitted.getDegree();
            newFeature.algebraicComplexity = static_cast<float>(varietyDim) / Config::WINDOW_SIZE;

            // Optimized polynomial coefficient storage
            const auto& coeffs = fitted.getCoefficients();
            for (size_t i = 0; i <= fitted.getDegree(); i++) {
                newFeature.coefficients[i] = gfToFloat(coeffs[i]);
            }

            // Efficient feature classification using bit manipulation
            uint8_t featureType = classifyFeature(fitted, varietyDim, windowY.data(), windowSize);
            newFeature.typeIndex = featureType;

            // Calculate confidence using optimized noise comparison
            newFeature.confidence = calculateConfidence(newFeature, noiseLevel);

            // Check if feature is significantly different using bit operations
            if (shouldAddFeature(newFeature)) {
                newFeatures.push_back(newFeature);
                addToPreviousFeatures(newFeature);
            }

        } catch (const std::exception& e) {
            Serial.print("Error in feature detection: ");
            Serial.println(e.what());
        }

        return newFeatures;
    }

private:
    // Optimized feature classification using bit operations
    uint8_t classifyFeature(const GFPolynomial& poly, int varietyDim, const uint8_t* values, size_t n) {
        // Calculate autocorrelation efficiently using fixed-point arithmetic
        int32_t autocorr = calculateFixedPointAutocorrelation(values, n);
        
        uint8_t degree = poly.getDegree();
        uint8_t type = 7; // Default to unknown

        // Use bit manipulation for fast classification
        if (degree <= 1) {
            type = 0; // linear
        } else if (degree == 2) {
            // Check for vertex vs transition using derivative
            type = hasUniqueExtremum(poly) ? 1 : 2;
        } else if (autocorr > (1 << 14)) { // threshold in fixed-point
            type = 3; // periodic
        } else if (varietyDim < (n >> 2)) {
            type = 4; // constant_piecewise
        } else if (degree >= Config::MAX_POLY_DEGREE) {
            type = 5; // complex_nonlinear
        } else {
            type = 6; // polynomial
        }

        return type;
    }

    // Fixed-point autocorrelation calculation
    int32_t calculateFixedPointAutocorrelation(const uint8_t* values, size_t n) {
        constexpr int FIXED_POINT_SHIFT = 16;
        int32_t sum = 0;
        int32_t mean = 0;
        
        // Calculate mean using fixed-point arithmetic
        for (size_t i = 0; i < n; i++) {
            mean += values[i];
        }
        mean = (mean << FIXED_POINT_SHIFT) / n;

        // Calculate autocorrelation with lag 1
        for (size_t i = 1; i < n; i++) {
            int32_t diff1 = (values[i] << FIXED_POINT_SHIFT) - mean;
            int32_t diff2 = (values[i-1] << FIXED_POINT_SHIFT) - mean;
            sum += (diff1 * diff2) >> FIXED_POINT_SHIFT;
        }

        return sum / (n - 1);
    }

    // Optimized extremum detection
    bool hasUniqueExtremum(const GFPolynomial& poly) {
        auto deriv = poly.derivative();
        uint8_t rootCount = 0;
        
        // Check for roots of derivative efficiently
        for (uint8_t x = 0; x < Config::FIELD_PRIME && rootCount <= 1; x++) {
            if (deriv.evaluate(x) == 0) rootCount++;
        }
        
        return rootCount == 1;
    }

    // Efficient confidence calculation using bit manipulation
    float calculateConfidence(const Feature& feature, float noiseLevel) {
        if (noiseLevel <= 0.0f) return 1.0f;
        
        // Use fast inverse square root approximation
        float x = noiseLevel * 3.0f;
        const float x2 = x * 0.5F;
        const float threehalfs = 1.5F;
        
        union {
            float f;
            uint32_t i;
        } conv = { .f = x };
        conv.i = 0x5f3759df - (conv.i >> 1);
        conv.f *= threehalfs - (x2 * conv.f * conv.f);
        
        return std::min(1.0f, std::abs(feature.value) * conv.f);
    }

    // Optimized feature comparison
    bool shouldAddFeature(const Feature& newFeature) {
        if (numPreviousFeatures == 0) return true;
        
        const Feature& lastFeature = previousFeatures[(numPreviousFeatures - 1) % Config::MAX_FEATURES];
        
        // Use bit operations for quick comparison
        bool valueChange = std::abs(newFeature.value - lastFeature.value) > Config::MIN_SIGNIFICANCE;
        bool complexityChange = std::abs(newFeature.algebraicComplexity - lastFeature.algebraicComplexity) > Config::EPSILON;
        bool typeChange = newFeature.typeIndex != lastFeature.typeIndex;
        
        return valueChange || complexityChange || typeChange;
    }

    // Efficient previous feature management
    void addToPreviousFeatures(const Feature& feature) {
        previousFeatures[numPreviousFeatures % Config::MAX_FEATURES] = feature;
        if (numPreviousFeatures < Config::MAX_FEATURES) {
            numPreviousFeatures++;
        }
    }
};

// Optimized signal processor for real-time monitoring
class SignalProcessor {
private:
    AlgebraicFeatureDetector detector;
    uint32_t lastProcessTime;
    static constexpr uint32_t PROCESS_INTERVAL_MS = 100;
    
    // Circular buffer for real-time data
    CircularBuffer<float, 32> dataBuffer;

public:
    SignalProcessor() : lastProcessTime(0) {}

    void setup() {
        Serial.begin(115200);
        lastProcessTime = millis();
    }

    void processIncomingData(float value) {
        dataBuffer.push(value);
        uint32_t currentTime = millis();
        
        if (currentTime - lastProcessTime >= PROCESS_INTERVAL_MS) {
            processBufferedData(currentTime);
            lastProcessTime = currentTime;
        }
    }

private:
    void processBufferedData(uint32_t currentTime) {
        while (!dataBuffer.empty()) {
            if (auto value = dataBuffer.pop()) {
                DataPoint point(currentTime / 1000.0f, *value);
                detector.processDataPoint(point);
            }
        }

        auto features = detector.detectAndUpdateFeatures();
        if (!features.empty()) {
            printFeatures(features);
        }
    }

    void printFeatures(const std::vector<Feature>& features) {
        for (const auto& feature : features) {
            Serial.print(F("Feature: "));
            Serial.print(Feature::getTypeString(feature.typeIndex));
            Serial.print(F(" at t="));
            Serial.print(feature.timestamp);
            Serial.print(F(" v="));
            Serial.println(feature.value);
        }
    }
};

// Memory-efficient test signal generator
class TestSignalGenerator {
private:
    static constexpr float TWO_PI = 6.28318530718f;
    uint16_t index = 0;

public:
    float getNextValue() {
        float t = index * 0.1f;
        float value;

        // Use bit operations for efficient case selection
        switch (index >> 4) {
            case 0: // Linear
                value = 0.1f * t;
                break;
            case 1: // Quadratic
                value = 0.05f * t * t;
                break;
            case 2: // Periodic
                value = sin(TWO_PI * t);
                break;
            case 3: // Complex polynomial
                value = 0.01f * t * t * t - 0.2f * t * t + 0.5f * t;
                break;
            default: // Noise
                value = 0.2f * (random(2000) - 1000) / 1000.0f;
        }

        index = (index + 1) % 100;
        return value;
    }
};

TestSignalGenerator signalGen;
SignalProcessor signalProcessor;

void setup() {
    signalProcessor.setup();
}

void loop() {
    float value = signalGen.getNextValue();
    signalProcessor.processIncomingData(value);
    delay(100);
}
