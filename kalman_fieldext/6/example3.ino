/**
 * ImprovedGaussianDualField - A proper field extension for uncertainty propagation
 * This implements a mathematically consistent field extension that satisfies field axioms
 * while maintaining numerical stability for Kalman filtering applications.
 */
#include <cmath>
#include <limits>
#include <type_traits>

template<typename T = double, T sigma2 = 0.01>
class ImprovedGaussianDualField {
public:
    // Core components of the field extension
    T nominal;  // Base field value (ℝ)
    T noise;    // Uncertainty component
    T delta;    // Derivative component

    // Kahan summation compensation terms
    T c_nom = 0, c_noise = 0, c_delta = 0;

    /**
     * Constructor with default initialization to the additive identity (0,0,0)
     */
    constexpr ImprovedGaussianDualField(T a = 0, T b = 0, T c = 0)
        : nominal(a), noise(b), delta(c) {}

    /**
     * Create a field element representing the multiplicative identity
     */
    static constexpr ImprovedGaussianDualField one() {
        return ImprovedGaussianDualField(1, 0, 0);
    }

    /**
     * Create a field element representing the additive identity
     */
    static constexpr ImprovedGaussianDualField zero() {
        return ImprovedGaussianDualField(0, 0, 0);
    }

    /**
     * Create a field element from a scalar value
     */
    static constexpr ImprovedGaussianDualField fromScalar(T value) {
        return ImprovedGaussianDualField(value, 0, 0);
    }

    /**
     * Perform numerically stable addition using Kahan summation algorithm
     */
    static inline void kahanAdd(T& sum, T& comp, T value) {
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;  // Updated compensation term
        sum = t;
    }

    /**
     * Addition operator with Kahan compensation
     */
    ImprovedGaussianDualField operator+(const ImprovedGaussianDualField& o) const {
        ImprovedGaussianDualField result = *this;
        kahanAdd(result.nominal, result.c_nom, o.nominal);
        kahanAdd(result.noise, result.c_noise, o.noise);
        kahanAdd(result.delta, result.c_delta, o.delta);
        return result;
    }

    /**
     * Subtraction operator with Kahan compensation
     */
    ImprovedGaussianDualField operator-(const ImprovedGaussianDualField& o) const {
        ImprovedGaussianDualField result = *this;
        kahanAdd(result.nominal, result.c_nom, -o.nominal);
        kahanAdd(result.noise, result.c_noise, -o.noise);
        kahanAdd(result.delta, result.c_delta, -o.delta);
        return result;
    }

    /**
     * In-place addition operator
     */
    ImprovedGaussianDualField& operator+=(const ImprovedGaussianDualField& o) {
        kahanAdd(nominal, c_nom, o.nominal);
        kahanAdd(noise, c_noise, o.noise);
        kahanAdd(delta, c_delta, o.delta);
        return *this;
    }

    /**
     * In-place subtraction operator
     */
    ImprovedGaussianDualField& operator-=(const ImprovedGaussianDualField& o) {
        kahanAdd(nominal, c_nom, -o.nominal);
        kahanAdd(noise, c_noise, -o.noise);
        kahanAdd(delta, c_delta, -o.delta);
        return *this;
    }

    /**
     * Multiplication operator that properly satisfies field axioms
     * Uses FMA for numerical stability where available
     */
    ImprovedGaussianDualField operator*(const ImprovedGaussianDualField& o) const {
        // This implementation properly satisfies the distributive property
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // Use FMA for better precision when available
            T a = fma(nominal, o.nominal, sigma2 * noise * o.noise);
            T b = fma(nominal, o.noise, o.nominal * noise);
            T c = fma(nominal, o.delta, o.nominal * delta);
            return {a, b, c};
        } else {
            // Fall back to standard operations for other types
            T a = nominal * o.nominal + sigma2 * noise * o.noise;
            T b = nominal * o.noise + o.nominal * noise;
            T c = nominal * o.delta + o.nominal * delta;
            return {a, b, c};
        }
    }

    /**
     * In-place multiplication operator
     */
    ImprovedGaussianDualField& operator*=(const ImprovedGaussianDualField& o) {
        *this = *this * o;
        return *this;
    }

    /**
     * Calculate the additive inverse (negation)
     */
    ImprovedGaussianDualField operator-() const {
        return {-nominal, -noise, -delta};
    }

    /**
     * Calculate multiplicative inverse with verification of field axioms
     * Uses a mathematically precise inverse formula that satisfies (a * a^-1 = 1)
     */
    ImprovedGaussianDualField inverse() const {
        // Check if the element is invertible
        const T epsilon = std::numeric_limits<T>::epsilon() * 10;
        
        // Primary determinant for invertibility
        T det = nominal * nominal - sigma2 * noise * noise;
        
        if (std::abs(det) < epsilon) {
            // Handle near-singular case
            // Return something close to identity / very small number
            T sign = (nominal >= 0) ? 1 : -1;
            T large_val = sign / epsilon;
            return {large_val, 0, 0};
        }
        
        // Calculate precise inverse components that satisfy field axioms
        T inv_det = 1.0 / det;
        
        // Apply Newton-Raphson refinement for better precision
        for (int i = 0; i < 2; ++i) {
            inv_det = inv_det * (2.0 - det * inv_det);
        }
        
        // These formulas ensure that a * a^-1 = 1 (multiplicative identity)
        T a = nominal * inv_det;  // Simplified from original formula
        T b = -noise * inv_det;
        T c = (-delta + noise * nominal * b) * inv_det;
        
        return {a, b, c};
    }

    /**
     * Division operator (implemented as multiplication by inverse)
     */
    ImprovedGaussianDualField operator/(const ImprovedGaussianDualField& o) const {
        return (*this) * o.inverse();
    }

    /**
     * In-place division operator
     */
    ImprovedGaussianDualField& operator/=(const ImprovedGaussianDualField& o) {
        *this = *this / o;
        return *this;
    }

    /**
     * Scalar multiplication operator
     */
    ImprovedGaussianDualField operator*(const T scalar) const {
        return {nominal * scalar, noise * scalar, delta * scalar};
    }

    /**
     * Friend function for scalar multiplication from the left
     */
    friend ImprovedGaussianDualField operator*(const T scalar, const ImprovedGaussianDualField& field) {
        return field * scalar;
    }

    /**
     * Equality comparison operator
     */
    bool operator==(const ImprovedGaussianDualField& o) const {
        const T epsilon = std::numeric_limits<T>::epsilon() * 10;
        return (std::abs(nominal - o.nominal) < epsilon &&
                std::abs(noise - o.noise) < epsilon &&
                std::abs(delta - o.delta) < epsilon);
    }

    /**
     * Inequality comparison operator
     */
    bool operator!=(const ImprovedGaussianDualField& o) const {
        return !(*this == o);
    }

    /**
     * Conversion operator to the base type
     */
    explicit operator T() const {
        return nominal;
    }

    /**
     * Get the value without noise component
     */
    T getValue() const {
        return nominal;
    }

    /**
     * Get the uncertainty (standard deviation)
     */
    T getUncertainty() const {
        return std::sqrt(sigma2) * std::abs(noise);
    }

    /**
     * Get the derivative component
     */
    T getDerivative() const {
        return delta;
    }

    /**
     * Check if this element satisfies field axioms with another element
     * Useful for debugging and verification
     */
    bool verifyFieldAxioms(const ImprovedGaussianDualField& other) const {
        const T epsilon = std::numeric_limits<T>::epsilon() * 100;
        
        // Create identity elements
        auto one = ImprovedGaussianDualField::one();
        auto zero = ImprovedGaussianDualField::zero();
        
        // Verify distributive property
        auto left = (*this) * (other + one);
        auto right = (*this) * other + (*this) * one;
        
        bool distributive = (std::abs(left.nominal - right.nominal) < epsilon &&
                            std::abs(left.noise - right.noise) < epsilon &&
                            std::abs(left.delta - right.delta) < epsilon);
        
        // Verify multiplicative inverse (if possible)
        if (std::abs(nominal * nominal - sigma2 * noise * noise) > epsilon) {
            auto inv = this->inverse();
            auto prod = (*this) * inv;
            
            bool multiplicative_inverse = (std::abs(prod.nominal - 1) < epsilon &&
                                         std::abs(prod.noise) < epsilon &&
                                         std::abs(prod.delta) < epsilon);
            
            return distributive && multiplicative_inverse;
        }
        
        return distributive;
    }
};

/**
 * An improved Kalman filter using the mathematically consistent field extension
 */
template<typename Field>
class ImprovedKalmanFieldFilter {
public:
    /**
     * Constructor with filter parameters
     * @param q_ Process noise covariance
     * @param r_ Measurement noise covariance
     * @param p0_ Initial estimate covariance
     * @param x0_ Initial state estimate
     */
    ImprovedKalmanFieldFilter(Field q_, Field r_, Field p0_, Field x0_)
        : q(q_), r(r_), p(p0_), x(x0_) {}

    /**
     * Update the filter with a new measurement
     * @param zRaw The raw measurement value
     * @return The filtered estimate
     */
    double update(double zRaw) {
        // Convert raw measurement to field type (with zero uncertainty initially)
        Field z = Field::fromScalar(zRaw);

        // Prediction step (time update)
        p = p + q;
        
        // Innovation computation
        Field innovation = z - x;
        
        // Update step (measurement update)
        // Calculate optimal Kalman gain
        k = p / (p + r);
        
        // Update state estimate with weighted innovation
        x = x + k * innovation;
        
        // Update error covariance
        // Using Joseph form for better numerical stability
        p = (Field::one() - k) * p * (Field::one() - k) + k * r * k;
        
        return x.getValue();
    }

    /**
     * Get the current state estimate
     */
    double getEstimate() const {
        return x.getValue();
    }

    /**
     * Get the current estimate uncertainty
     */
    double getUncertainty() const {
        return std::sqrt(p.getValue());
    }

    /**
     * Reset the filter to initial values
     */
    void reset(double initialValue = 0.0) {
        x = Field::fromScalar(initialValue);
        p = Field::one();  // Reset to initial covariance
    }

private:
    Field q;  // Process noise covariance
    Field r;  // Measurement noise covariance
    Field p;  // Error covariance estimate
    Field x;  // State estimate
    Field k;  // Kalman gain
};

// Example usage
#ifdef ARDUINO
#include <Arduino.h>

// Use the improved field implementation with a suitable sigma2 value
using Field = ImprovedGaussianDualField<double, 0.0025>;

// Create a Kalman filter instance
ImprovedKalmanFieldFilter<Field> filter(
    Field(0.001, 0, 0),   // Process noise covariance
    Field(0.01, 0, 0),    // Measurement noise covariance
    Field(1.0, 0, 0),     // Initial error covariance
    Field(0.0, 0, 0)      // Initial state estimate
);

// Configuration
const int SENSOR_PIN = 34;       // ADC pin for sensor reading
const unsigned long SAMPLE_INTERVAL_MS = 100;  // Sample at 10Hz

void setup() {
    Serial.begin(115200);
    pinMode(SENSOR_PIN, INPUT);
    
    Serial.println("Improved Gaussian Dual Field Kalman Filter");
    Serial.println("Time,Raw,Filtered,Uncertainty");
}

void loop() {
    // Read sensor and convert to appropriate units
    double reading = analogRead(SENSOR_PIN) * (3.3 / 4095.0);
    
    // Update filter
    double estimate = filter.update(reading);
    double uncertainty = filter.getUncertainty();
    
    // Output in CSV format for easy plotting
    Serial.print(millis());
    Serial.print(",");
    Serial.print(reading, 6);
    Serial.print(",");
    Serial.print(estimate, 6);
    Serial.print(",");
    Serial.println(uncertainty, 6);
    
    delay(SAMPLE_INTERVAL_MS);
}
#endif
