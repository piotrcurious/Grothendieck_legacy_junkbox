/**
 * GaussianDualField - A numerically stable field implementation for Kalman filtering
 * with built-in Kahan summation and dual number differentiation support.
 */
#include <cmath>

template<typename T = double, T sigma2 = 0.01>
class GaussianDualField {
public:
    T nominal;  // Main value
    T noise;    // First-order noise term
    T delta;    // Derivative term

    // Kahan compensation terms for numerical stability
    T c_nom = 0, c_noise = 0, c_delta = 0;

    /**
     * Constructor with default initialization to zero
     */
    constexpr GaussianDualField(T a = 0, T b = 0, T c = 0)
        : nominal(a), noise(b), delta(c) {}

    /**
     * Perform numerically stable addition using Kahan summation algorithm
     * @param sum The accumulator value
     * @param comp The compensation term
     * @param value The value to add
     */
    static inline void kahanAdd(T& sum, T& comp, T value) {
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;  // New compensation term
        sum = t;
    }

    /**
     * Addition operator with Kahan compensation
     */
    GaussianDualField operator+(const GaussianDualField& o) const {
        GaussianDualField result = *this;
        kahanAdd(result.nominal, result.c_nom, o.nominal);
        kahanAdd(result.noise, result.c_noise, o.noise);
        kahanAdd(result.delta, result.c_delta, o.delta);
        return result;
    }

    /**
     * Subtraction operator with Kahan compensation
     */
    GaussianDualField operator-(const GaussianDualField& o) const {
        GaussianDualField result = *this;
        kahanAdd(result.nominal, result.c_nom, -o.nominal);
        kahanAdd(result.noise, result.c_noise, -o.noise);
        kahanAdd(result.delta, result.c_delta, -o.delta);
        return result;
    }

    /**
     * In-place addition operator
     */
    GaussianDualField& operator+=(const GaussianDualField& o) {
        kahanAdd(nominal, c_nom, o.nominal);
        kahanAdd(noise, c_noise, o.noise);
        kahanAdd(delta, c_delta, o.delta);
        return *this;
    }

    /**
     * In-place subtraction operator
     */
    GaussianDualField& operator-=(const GaussianDualField& o) {
        kahanAdd(nominal, c_nom, -o.nominal);
        kahanAdd(noise, c_noise, -o.noise);
        kahanAdd(delta, c_delta, -o.delta);
        return *this;
    }

    /**
     * Multiplication operator with FMA stabilization
     */
    GaussianDualField operator*(const GaussianDualField& o) const {
        // FMA-stabilized multiplication for better numerical precision
        T a = fma(nominal, o.nominal, sigma2 * noise * o.noise);
        T b = fma(nominal, o.noise, noise * o.nominal);
        T c = fma(nominal, o.delta, delta * o.nominal);
        return {a, b, c};
    }

    /**
     * In-place multiplication operator
     */
    GaussianDualField& operator*=(const GaussianDualField& o) {
        *this = *this * o;
        return *this;
    }

    /**
     * Calculate inverse with Newton-Raphson refinement
     * @return The inverse field
     */
    GaussianDualField inverse() const {
        // Handle potential division by zero
        if (std::abs(nominal) < std::numeric_limits<T>::epsilon() * 10) {
            // Return a safely large value with proper sign
            T sign = (nominal >= 0) ? 1 : -1;
            return {sign / std::numeric_limits<T>::epsilon(), 0, 0};
        }

        T denom = nominal * nominal - sigma2 * noise * noise;

        // Newton-Raphson refinement for inverse (more accurate than direct division)
        T inv_nom = 1.0 / denom;
        // Two iterations are typically sufficient for double precision
        for (int i = 0; i < 2; ++i) {
            inv_nom = inv_nom * (2.0 - denom * inv_nom);
        }

        T a = (nominal * nominal + sigma2 * noise * noise) * inv_nom * inv_nom;
        T b = -2 * nominal * noise * inv_nom * inv_nom;
        T c = -delta * nominal * inv_nom;

        return {a, b, c};
    }

    /**
     * Division operator (implemented as multiplication by inverse)
     */
    GaussianDualField operator/(const GaussianDualField& o) const {
        return (*this) * o.inverse();
    }

    /**
     * In-place division operator
     */
    GaussianDualField& operator/=(const GaussianDualField& o) {
        *this = *this / o;
        return *this;
    }

    /**
     * Scalar multiplication operator
     */
    GaussianDualField operator*(const T scalar) const {
        return {nominal * scalar, noise * scalar, delta * scalar};
    }

    /**
     * Friend function for scalar multiplication from the left
     */
    friend GaussianDualField operator*(const T scalar, const GaussianDualField& field) {
        return field * scalar;
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
};

/**
 * Kalman filter implementation using GaussianDualField
 * @tparam Field The field type used (default: GaussianDualField)
 */
template<typename Field>
class KalmanFieldFilter {
public:
    /**
     * Constructor with filter parameters
     * @param q_ Process noise
     * @param r_ Measurement noise
     * @param p0_ Initial covariance
     * @param x0_ Initial estimate
     */
    KalmanFieldFilter(Field q_, Field r_, Field p0_, Field x0_)
        : q(q_), r(r_), p(p0_), x(x0_) {}

    /**
     * Update the filter with a new measurement
     * @param zRaw The raw measurement value
     * @return The filtered estimate
     */
    double update(double zRaw) {
        // Convert raw measurement to field type
        Field z(zRaw, 0.0, 0.0);

        // Prediction step
        p += q;
        
        // Update step
        k = p / (p + r);
        x += k * (z - x);
        p = (Field(1.0, 0.0, 0.0) - k) * p;

        return x.getValue();
    }

    /**
     * Get the current estimate
     * @return Current estimate value
     */
    double getEstimate() const {
        return x.getValue();
    }

    /**
     * Get the current covariance
     * @return Current covariance value
     */
    double getCovariance() const {
        return p.getValue();
    }

    /**
     * Get the current uncertainty (standard deviation)
     * @return Current uncertainty value
     */
    double getUncertainty() const {
        return std::sqrt(p.getValue());
    }

    /**
     * Reset the filter to initial values
     * @param initialValue The new initial value
     */
    void reset(double initialValue = 0.0) {
        x = Field(initialValue, 0.0, 0.0);
        p = Field(1.0, 0.0, 0.0);  // Reset to default covariance
    }

private:
    Field q;  // Process noise
    Field r;  // Measurement noise
    Field p;  // Estimate covariance
    Field x;  // Current estimate
    Field k;  // Kalman gain
};

// Main implementation for Arduino
#ifdef ARDUINO
#include <Arduino.h>

// Use a smaller sigma2 value for better stability
using Field = GaussianDualField<double, 0.0025>;

// Global filter instance
KalmanFieldFilter<Field> filter(
    Field(0.001),   // Process noise - how quickly the true value can change
    Field(0.01),    // Measurement noise - how noisy the sensor readings are
    Field(1.0),     // Initial covariance - starting uncertainty
    Field(0.0)      // Initial estimate - starting value
);

// Pin definitions
const int SENSOR_PIN = 34;  // Analog pin for input
const int LED_PIN = 13;     // Optional indicator LED

void setup() {
    Serial.begin(115200);
    pinMode(SENSOR_PIN, INPUT);
    pinMode(LED_PIN, OUTPUT);
    
    Serial.println("Gaussian Dual Field Kalman Filter");
    Serial.println("Time,Raw,Filtered");
}

void loop() {
    // Read analog value and convert to voltage
    double reading = analogRead(SENSOR_PIN) * (3.3 / 4095.0);
    
    // Update filter
    double estimate = filter.update(reading);
    
    // Optional: activate LED if estimate exceeds threshold
    digitalWrite(LED_PIN, estimate > 1.65 ? HIGH : LOW);
    
    // Output data in CSV format
    Serial.print(millis());
    Serial.print(",");
    Serial.print(reading, 6);
    Serial.print(",");
    Serial.println(estimate, 6);
    
    delay(100);  // 10Hz sampling rate
}
#endif
