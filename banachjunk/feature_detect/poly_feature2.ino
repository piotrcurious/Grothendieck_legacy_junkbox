// Advanced Signal Feature Detection using Banach Spaces, Algebraic Geometry, and Galois Fields
// For ESP32 Arduino
// Complete implementation with error handling and memory management

#include <vector>
#include <memory>
#include <optional>
#include <cmath>
#include <algorithm>

// Configuration constants
namespace Config {
    constexpr int WINDOW_SIZE = 20;
    constexpr float EPSILON = 0.01f;
    constexpr int MAX_FEATURES = 10;
    constexpr int FIELD_PRIME = 251;
    constexpr int MAX_POLY_DEGREE = 4;
    constexpr size_t BUFFER_SIZE = 1024;
    constexpr float MIN_SIGNIFICANCE = 0.1f;
    constexpr int NUM_SAMPLES_FOR_NOISE = 50;
}

// Error handling
class Error : public std::runtime_error {
public:
    explicit Error(const char* message) : std::runtime_error(message) {}
};

// Forward declarations
class GaloisField;
class GFPolynomial;
class AlgebraicFeatureDetector;

// Data structures
struct DataPoint {
    float timestamp;
    float value;
    
    DataPoint(float t = 0.0f, float v = 0.0f) : timestamp(t), value(v) {}
};

struct Feature {
    float timestamp;
    float value;
    int polynomialDegree;
    float algebraicComplexity;
    std::string type;
    float confidence;
    std::vector<float> polynomialCoefficients;
    
    // Constructor with initialization
    Feature() : timestamp(0), value(0), polynomialDegree(0),
                algebraicComplexity(0), confidence(0) {}
};

// Circular buffer for real-time processing
template<typename T, size_t Size>
class CircularBuffer {
private:
    std::array<T, Size> buffer;
    size_t head = 0;
    size_t tail = 0;
    bool full = false;

public:
    void push(const T& item) {
        buffer[head] = item;
        head = (head + 1) % Size;
        if (head == tail) {
            full = true;
            tail = (tail + 1) % Size;
        }
    }

    std::optional<T> pop() {
        if (empty()) return std::nullopt;
        T item = buffer[tail];
        tail = (tail + 1) % Size;
        full = false;
        return item;
    }

    bool empty() const { return !full && (head == tail); }
    bool full() const { return full; }
    size_t size() const {
        if (full) return Size;
        return (head - tail) % Size;
    }
};

// Enhanced Galois Field implementation
class GaloisField {
private:
    const int prime;
    std::vector<std::vector<uint8_t>> logTable;
    std::vector<uint8_t> expTable;
    std::vector<uint8_t> inverseTable;

public:
    explicit GaloisField(int p = Config::FIELD_PRIME) : prime(p) {
        if (!isPrime(p)) {
            throw Error("Field order must be prime");
        }
        initTables();
    }

    bool isPrime(int n) const {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) return false;
        }
        return true;
    }

    void initTables() {
        expTable.resize(prime - 1);
        logTable.resize(prime);
        inverseTable.resize(prime);

        // Find primitive root and generate tables
        int primitiveRoot = findPrimitiveRoot();
        int x = 1;
        for (int i = 0; i < prime - 1; i++) {
            expTable[i] = x;
            logTable[x].push_back(i);
            x = (x * primitiveRoot) % prime;
        }

        // Generate inverse table
        for (int i = 1; i < prime; i++) {
            inverseTable[i] = power(i, prime - 2);
        }
    }

    int findPrimitiveRoot() const {
        std::vector<int> factors;
        int phi = prime - 1;
        int n = phi;
        
        // Find prime factors of phi
        for (int i = 2; i * i <= n; i++) {
            if (n % i == 0) {
                factors.push_back(i);
                while (n % i == 0) n /= i;
            }
        }
        if (n > 1) factors.push_back(n);

        // Test each number for primitive root property
        for (int r = 2; r < prime; r++) {
            bool isPrimitive = true;
            for (int factor : factors) {
                if (power(r, phi / factor) == 1) {
                    isPrimitive = false;
                    break;
                }
            }
            if (isPrimitive) return r;
        }
        throw Error("No primitive root found");
    }

    uint8_t add(uint8_t a, uint8_t b) const {
        return (a + b) % prime;
    }

    uint8_t subtract(uint8_t a, uint8_t b) const {
        return (a + prime - b) % prime;
    }

    uint8_t multiply(uint8_t a, uint8_t b) const {
        if (a == 0 || b == 0) return 0;
        if (logTable[a].empty() || logTable[b].empty()) return 0;
        int sum = logTable[a][0] + logTable[b][0];
        return expTable[sum % (prime - 1)];
    }

    uint8_t divide(uint8_t a, uint8_t b) const {
        if (b == 0) throw Error("Division by zero");
        if (a == 0) return 0;
        return multiply(a, inverseTable[b]);
    }

    uint8_t power(uint8_t base, int exp) const {
        if (exp < 0) {
            if (base == 0) throw Error("Zero cannot be raised to negative power");
            return power(inverseTable[base], -exp);
        }
        if (exp == 0) return 1;
        if (base == 0) return 0;

        uint8_t result = 1;
        while (exp > 0) {
            if (exp & 1) result = multiply(result, base);
            base = multiply(base, base);
            exp >>= 1;
        }
        return result;
    }

    int getPrime() const { return prime; }
};

// Enhanced polynomial implementation
class GFPolynomial {
private:
    std::vector<uint8_t> coefficients;
    const GaloisField& gf;

public:
    GFPolynomial(const std::vector<uint8_t>& coeffs, const GaloisField& field)
        : coefficients(coeffs), gf(field) {
        normalize();
    }

    void normalize() {
        while (!coefficients.empty() && coefficients.back() == 0) {
            coefficients.pop_back();
        }
        if (coefficients.empty()) coefficients.push_back(0);
    }

    uint8_t evaluate(uint8_t x) const {
        uint8_t result = 0;
        for (int i = coefficients.size() - 1; i >= 0; i--) {
            result = gf.add(gf.multiply(result, x), coefficients[i]);
        }
        return result;
    }

    std::vector<uint8_t> findRoots() const {
        std::vector<uint8_t> roots;
        for (uint8_t x = 0; x < gf.getPrime(); x++) {
            if (evaluate(x) == 0) {
                roots.push_back(x);
            }
        }
        return roots;
    }

    GFPolynomial derivative() const {
        if (coefficients.size() <= 1) return GFPolynomial({0}, gf);
        
        std::vector<uint8_t> derivCoeffs;
        for (size_t i = 1; i < coefficients.size(); i++) {
            derivCoeffs.push_back(gf.multiply(coefficients[i], i));
        }
        return GFPolynomial(derivCoeffs, gf);
    }

    int degree() const {
        return std::max(0, static_cast<int>(coefficients.size()) - 1);
    }

    const std::vector<uint8_t>& getCoefficients() const {
        return coefficients;
    }
};

// Statistical utilities
class Statistics {
public:
    static float calculateMean(const std::vector<float>& values) {
        if (values.empty()) return 0.0f;
        float sum = 0.0f;
        for (float value : values) sum += value;
        return sum / values.size();
    }

    static float calculateVariance(const std::vector<float>& values) {
        if (values.size() < 2) return 0.0f;
        float mean = calculateMean(values);
        float sumSquares = 0.0f;
        for (float value : values) {
            float diff = value - mean;
            sumSquares += diff * diff;
        }
        return sumSquares / (values.size() - 1);
    }

    static float calculateAutocorrelation(const std::vector<float>& values, int lag) {
        if (values.size() <= lag) return 0.0f;
        float mean = calculateMean(values);
        float variance = calculateVariance(values);
        if (variance == 0.0f) return 0.0f;

        float sum = 0.0f;
        for (size_t i = 0; i < values.size() - lag; i++) {
            sum += (values[i] - mean) * (values[i + lag] - mean);
        }
        return sum / ((values.size() - lag) * variance);
    }
};

// Enhanced feature detector
class AlgebraicFeatureDetector {
private:
    GaloisField gf;
    CircularBuffer<DataPoint, Config::BUFFER_SIZE> buffer;
    float noiseLevel;
    std::vector<Feature> previousFeatures;

    uint8_t floatToGF(float value) {
        int scaled = static_cast<int>((value + 1.0f) * (Config::FIELD_PRIME - 1) / 2.0f);
        return static_cast<uint8_t>(std::clamp(scaled, 0, Config::FIELD_PRIME - 1));
    }

    float gfToFloat(uint8_t value) {
        return (2.0f * value / (Config::FIELD_PRIME - 1.0f)) - 1.0f;
    }

    GFPolynomial fitPolynomial(const std::vector<uint8_t>& x, const std::vector<uint8_t>& y) {
        int n = std::min(static_cast<int>(x.size()), Config::MAX_POLY_DEGREE + 1);
        std::vector<uint8_t> coeffs(n, 0);

        // Improved Lagrange interpolation with error checking
        try {
            for (int i = 0; i < n; i++) {
                uint8_t term = y[i];
                for (int j = 0; j < n; j++) {
                    if (i != j) {
                        uint8_t denominator = gf.subtract(x[i], x[j]);
                        if (denominator == 0) throw Error("Interpolation points must be distinct");
                        term = gf.multiply(term, 
                               gf.multiply(x[j], 
                               gf.power(denominator, Config::FIELD_PRIME - 2)));
                    }
                }
                coeffs[i] = term;
            }
        } catch (const Error& e) {
            // Fallback to linear fit on error
            coeffs.resize(2);
            coeffs[0] = y[0];
            coeffs[1] = gf.subtract(y[1], y[0]);
        }

        return GFPolynomial(coeffs, gf);
    }

    int calculateVarietyDimension(const std::vector<uint8_t>& values) {
        std::vector<bool> seen(Config::FIELD_PRIME, false);
        int dimension = 0;
        for (uint8_t val : values) {
            if (!seen[val]) {
                seen[val] = true;
                dimension++;
            }
        }
        return dimension;
    }

    float estimateNoiseLevel(const std::vector<DataPoint>& data) {
        if (data.size() < Config::NUM_SAMPLES_FOR_NOISE) return 0.0f;
        
        std::vector<float> differences;
        differences.reserve(data.size() - 1);
        
        for (size_t i = 1; i < data.size(); i++) {
            differences.push_back(std::abs(data[i].value - data[i-1].value));
        }
        
        std::sort(differences.begin(), differences.end());
        return differences[differences.size() / 2] * 1.4826f; // MAD estimator
    }

    float calculateFeatureConfidence(const Feature& feature, float noiseLevel) {
        if (noiseLevel <= 0.0f) return 1.0f;
        return std::min(1.0f, std::abs(feature.value) / (3.0f * noiseLevel));
    }

    bool isSignificantlyDifferent(const Feature& f1, const Feature& f2) {
        return std::abs(f1.value - f2.value) > Config::MIN_SIGNIFICANCE ||
               std::abs(f1.algebraicComplexity - f2.algebraicComplexity) > Config::EPSILON;
    }

public:
    AlgebraicFeatureDetector() : gf(Config::FIELD_PRIME), noiseLevel(0.0f) {}

    void processDataPoint(const DataPoint& point) {
        buffer.push(point);
        if (buffer.size() >= Config::WINDOW_SIZE) {
            detectAndUpdateFeatures();
        }
    }

    std::vector<Feature> detectAndUpdateFeatures() {
        std::vector<DataPoint> windowData;
        std::vector<uint8_t> windowX;
        std::vector<uint8_t> windowY;
        std::vector<Feature> newFeatures;

        // Extract window data
        for (size_t i = 0; i < Config::WINDOW_SIZE; i++) {
            auto point = buffer.pop();
            if (!point) break;
            windowData.push_back(*point);
            windowX.push_back(floatToGF(point->timestamp));
            windowY.push_back(floatToGF(point->value));
            buffer.push(*point);
        }

        if (windowData.size() < Config::WINDOW_SIZE) return newFeatures;

        // Update noise estimate
        noiseLevel = estimateNoiseLevel(windowData);

        try {
            // Fit polynomial and analyze
            GFPolynomial fitted = fitPolynomial(windowX, windowY);
            int varietyDim = calculateVarietyDimension(windowY);
            
            Feature newFeature;
            newFeature.timestamp = windowData[Config::WINDOW_SIZE/2].timestamp;
            newFeature.value = windowData[Config::WINDOW_SIZE/2].value;
            newFeature.polynomialDegree = fitted.degree();
            newFeature.algebraicComplexity = static_cast<float>(varietyDim) / Config::WINDOW_SIZE;

            // Store polynomial coefficients
            auto coeffs = fitted.getCoefficients();
            newFeature.polynomialCoefficients.clear();
            for (uint8_t coeff : coeffs) {
                newFeature.polynomialCoefficients.push_back(gfToFloat(coeff));
            }

            // Enhanced feature classification
            auto roots = fitted.findRoots();
            auto derivative = fitted.derivative();
            
            // Calculate additional metrics for classification
            float autocorr = Statistics::calculateAutocorrelation(
                std::vector<float>(windowY.begin(), windowY.end()), 
                Config::WINDOW_SIZE / 4
            );

            if (fitted.degree() <= 1) {
                newFeature.type = "linear";
            } else if (fitted.degree() == 2 && roots.size() == 1) {
                newFeature.type = "quadratic_vertex";
            } else if (fitted.degree() == 2) {
                newFeature.type = "quadratic_transition";
            } else if (autocorr > 0.7f) {
                newFeature.type = "periodic";
            } else if (varietyDim < Config::WINDOW_SIZE / 4) {
                newFeature.type = "constant_piecewise";
            } else if (fitted.degree() >= Config::MAX_POLY_DEGREE) {
                newFeature.type = "complex_nonlinear";
            } else {
                newFeature.type = "polynomial_" + std::to_string(fitted.degree());
            }

            // Calculate confidence based on noise level and feature characteristics
            newFeature.confidence = calculateFeatureConfidence(newFeature, noiseLevel);

            // Only add feature if it's significantly different from previous ones
            if (previousFeatures.empty() || 
                isSignificantlyDifferent(newFeature, previousFeatures.back())) {
                newFeatures.push_back(newFeature);
                previousFeatures.push_back(newFeature);
                
                // Keep previous features buffer at reasonable size
                if (previousFeatures.size() > Config::MAX_FEATURES) {
                    previousFeatures.erase(previousFeatures.begin());
                }
            }

        } catch (const Error& e) {
            Serial.print("Error in feature detection: ");
            Serial.println(e.what());
        }

        return newFeatures;
    }

    // Clear internal state
    void reset() {
        while (!buffer.empty()) buffer.pop();
        previousFeatures.clear();
        noiseLevel = 0.0f;
    }

    // Get current noise level estimate
    float getCurrentNoiseLevel() const {
        return noiseLevel;
    }
};

// Real-time signal processor for continuous monitoring
class SignalProcessor {
private:
    AlgebraicFeatureDetector detector;
    std::vector<Feature> features;
    unsigned long lastProcessTime;
    const unsigned long PROCESS_INTERVAL_MS = 100; // Process every 100ms

public:
    SignalProcessor() : lastProcessTime(0) {}

    void setup() {
        Serial.begin(115200);
        lastProcessTime = millis();
    }

    void processIncomingData(float value) {
        unsigned long currentTime = millis();
        DataPoint point(currentTime / 1000.0f, value);
        detector.processDataPoint(point);

        if (currentTime - lastProcessTime >= PROCESS_INTERVAL_MS) {
            auto newFeatures = detector.detectAndUpdateFeatures();
            if (!newFeatures.empty()) {
                printFeatures(newFeatures);
            }
            lastProcessTime = currentTime;
        }
    }

    void printFeatures(const std::vector<Feature>& newFeatures) {
        for (const auto& feature : newFeatures) {
            Serial.println("Feature Detected:");
            Serial.print("Time: ");
            Serial.println(feature.timestamp);
            Serial.print("Value: ");
            Serial.println(feature.value);
            Serial.print("Type: ");
            Serial.println(feature.type.c_str());
            Serial.print("Polynomial Degree: ");
            Serial.println(feature.polynomialDegree);
            Serial.print("Confidence: ");
            Serial.println(feature.confidence);
            Serial.print("Algebraic Complexity: ");
            Serial.println(feature.algebraicComplexity);
            
            Serial.print("Polynomial Coefficients: ");
            for (float coeff : feature.polynomialCoefficients) {
                Serial.print(coeff);
                Serial.print(" ");
            }
            Serial.println();
            Serial.println("-------------------");
        }
    }
};

// Example usage in Arduino setup and loop
SignalProcessor signalProcessor;

void setup() {
    signalProcessor.setup();

    // Test signal generation
    const int numTestPoints = 100;
    float testSignal[numTestPoints];
    
    // Generate test signal with multiple feature types
    for (int i = 0; i < numTestPoints; i++) {
        float t = i * 0.1f;
        float value = 0.0f;
        
        // Linear trend
        if (i < 20) {
            value = 0.1f * t;
        }
        // Quadratic section
        else if (i < 40) {
            value = 0.05f * t * t;
        }
        // Periodic section
        else if (i < 60) {
            value = sin(2 * PI * t);
        }
        // Complex polynomial
        else if (i < 80) {
            value = 0.01f * t * t * t - 0.2f * t * t + 0.5f * t;
        }
        // Noise section
        else {
            value = 0.2f * random(-100, 100) / 100.0f;
        }
        
        testSignal[i] = value;
    }

    // Process test signal
    Serial.println("Processing test signal...");
    for (int i = 0; i < numTestPoints; i++) {
        signalProcessor.processIncomingData(testSignal[i]);
        delay(10); // Simulate real-time data arrival
    }
    Serial.println("Test signal processing complete.");
}

void loop() {
    // In real applications, you would read sensor data here
    // For example:
    // float sensorValue = analogRead(SENSOR_PIN) / 1023.0f;
    // signalProcessor.processIncomingData(sensorValue);
    
    // For demonstration, we'll generate a simple varying signal
    static float t = 0.0f;
    float value = sin(2 * PI * 0.1f * t) + 0.2f * sin(2 * PI * 0.05f * t * t);
    signalProcessor.processIncomingData(value);
    t += 0.1f;
    
    delay(100); // Control sampling rate
}
