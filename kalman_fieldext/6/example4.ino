#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept> // For potential exceptions
#include <concepts>  // For concepts if using C++20

// Concept to check if a type is a floating point type (C++20)
// template<typename T>
// concept FloatingPoint = std::is_floating_point_v<T>;

/**
 * @brief UncertaintyAlgebra - Represents a number with uncertainty and derivative components.
 *
 * This class defines an algebraic structure over triples (nominal, noise, delta).
 * NOTE: This structure is a commutative R-algebra but is NOT a field because
 * multiplication is NOT associative. The name reflects its components rather
 * than implying strict field properties.
 *
 * The multiplication rule is defined as:
 * (a, b, c) * (a', b', c') = (a*a' + sigma2*b*b', a*b' + a'*b, a*c' + a'*c)
 *
 * - 'nominal': The base value (like a real number).
 * - 'noise': A component related to uncertainty propagation. Its exact interpretation
 * depends on the application context.
 * - 'delta': A component behaving like the infinitesimal part of a dual number,
 * useful for automatic differentiation.
 * - 'sigma2': A parameter influencing the interaction between 'noise' components
 * during multiplication. Must be specified at compile time.
 */
template<typename T = double, T sigma2 = 0.01>
// requires FloatingPoint<T> // C++20 constraint
class UncertaintyAlgebra {

    // Static assertion to ensure T is a floating-point type
    static_assert(std::is_floating_point_v<T>, "Template parameter T must be a floating-point type (float, double, long double).");

public:
    // --- Member Variables ---
    T nominal; // Base field value (like R)
    T noise;   // Uncertainty-related component
    T delta;   // Derivative component (like dual number part)

    // Kahan summation compensation terms (for += and -=)
    T c_nom = 0;
    T c_noise = 0;
    T c_delta = 0;

    // --- Constructors ---

    /**
     * @brief Default constructor. Initializes to additive identity (0, 0, 0).
     */
    constexpr UncertaintyAlgebra(T a = 0, T b = 0, T c = 0)
        : nominal(a), noise(b), delta(c) {}

    // --- Static Factory Methods ---

    /**
     * @brief Creates the multiplicative identity element (1, 0, 0).
     */
    static constexpr UncertaintyAlgebra one() {
        return UncertaintyAlgebra(1, 0, 0);
    }

    /**
     * @brief Creates the additive identity element (0, 0, 0).
     */
    static constexpr UncertaintyAlgebra zero() {
        return UncertaintyAlgebra(0, 0, 0);
    }

    /**
     * @brief Creates an element from a scalar value (value, 0, 0).
     */
    static constexpr UncertaintyAlgebra fromScalar(T value) {
        return UncertaintyAlgebra(value, 0, 0);
    }

    // --- Kahan Summation Helper ---

    /**
     * @brief Performs numerically stable addition using Kahan summation.
     * @param sum Reference to the sum accumulator.
     * @param comp Reference to the compensation term for the sum.
     * @param value The value to add.
     */
    static inline void kahanAdd(T& sum, T& comp, T value) {
        // Note: This implementation assumes standard IEEE 754 arithmetic.
        T y = value - comp; // comp corrects for lost low-order bits in previous additions
        T t = sum + y;      // Add corrected value; low-order bits of y might be lost
        // Calculate new compensation:
        // (t - sum) recovers the high-order part of y that was added
        // ((t - sum) - y) isolates the negative of the low-order bits lost in (sum + y)
        comp = (t - sum) - y;
        sum = t;            // Update the sum
    }

    // --- Arithmetic Operators ---

    /**
     * @brief Addition operator. Returns a new object.
     * Note: Kahan summation is NOT used here as it provides minimal benefit
     * for a single operation on temporary objects. It's used in +=.
     */
    UncertaintyAlgebra operator+(const UncertaintyAlgebra& o) const {
        return UncertaintyAlgebra(nominal + o.nominal,
                                  noise + o.noise,
                                  delta + o.delta);
    }

    /**
     * @brief Subtraction operator. Returns a new object.
     * Note: Kahan summation is NOT used here. It's used in -=.
     */
    UncertaintyAlgebra operator-(const UncertaintyAlgebra& o) const {
        return UncertaintyAlgebra(nominal - o.nominal,
                                  noise - o.noise,
                                  delta - o.delta);
    }

    /**
     * @brief In-place addition operator with Kahan compensation.
     */
    UncertaintyAlgebra& operator+=(const UncertaintyAlgebra& o) {
        kahanAdd(nominal, c_nom, o.nominal);
        kahanAdd(noise, c_noise, o.noise);
        kahanAdd(delta, c_delta, o.delta);
        return *this;
    }

    /**
     * @brief In-place subtraction operator with Kahan compensation.
     */
    UncertaintyAlgebra& operator-=(const UncertaintyAlgebra& o) {
        kahanAdd(nominal, c_nom, -o.nominal);
        kahanAdd(noise, c_noise, -o.noise);
        kahanAdd(delta, c_delta, -o.delta);
        return *this;
    }

    /**
     * @brief Multiplication operator.
     * Uses FMA (fused multiply-add) for potentially better numerical stability
     * and performance if available and T is float or double.
     */
    UncertaintyAlgebra operator*(const UncertaintyAlgebra& o) const {
        T res_nominal, res_noise, res_delta;

        // Multiplication rules:
        // nominal = a*a' + sigma2*b*b'
        // noise   = a*b' + a'*b
        // delta   = a*c' + a'*c

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // Use FMA for float and double types
            res_nominal = std::fma(nominal, o.nominal, sigma2 * noise * o.noise);
            res_noise   = std::fma(nominal, o.noise, o.nominal * noise);
            res_delta   = std::fma(nominal, o.delta, o.nominal * delta);
        } else {
            // Fall back to standard operations for other types (e.g., long double)
            // or if FMA is not desired/available.
            res_nominal = nominal * o.nominal + sigma2 * noise * o.noise;
            res_noise   = nominal * o.noise + o.nominal * noise;
            res_delta   = nominal * o.delta + o.nominal * delta;
        }
        return {res_nominal, res_noise, res_delta};
    }

    /**
     * @brief In-place multiplication operator.
     */
    UncertaintyAlgebra& operator*=(const UncertaintyAlgebra& o) {
        // Calculate result first to avoid using updated values prematurely
        UncertaintyAlgebra result = (*this) * o;
        *this = result; // Assign the computed result
        // Note: Kahan compensation terms are reset by the assignment
        //       if the multiplication significantly changes the scale.
        //       This is generally acceptable for multiplication.
        c_nom = 0; c_noise = 0; c_delta = 0;
        return *this;
    }

    /**
     * @brief Additive inverse (negation).
     */
    constexpr UncertaintyAlgebra operator-() const {
        return {-nominal, -noise, -delta};
    }

    /**
     * @brief Calculates the multiplicative inverse.
     * Throws std::runtime_error if the element is not invertible (determinant is near zero).
     * @return The multiplicative inverse.
     * @throws std::runtime_error If the determinant `nominal^2 - sigma2 * noise^2` is close to zero.
     */
    UncertaintyAlgebra inverse() const {
        // Determinant for invertibility: det = a^2 - sigma2 * b^2
        T det = nominal * nominal - sigma2 * noise * noise;

        // Use a relative tolerance for checking near-zero determinant
        const T relative_epsilon = std::numeric_limits<T>::epsilon() * 100;
        // Check against the scale of the terms involved
        T tolerance = relative_epsilon * (std::abs(nominal * nominal) + std::abs(sigma2 * noise * noise));
        // Ensure tolerance is at least epsilon for very small numbers
        tolerance = std::max(tolerance, std::numeric_limits<T>::epsilon());


        if (std::abs(det) < tolerance) {
            // Element is not invertible (or numerically very close to singular)
            throw std::runtime_error("UncertaintyAlgebra: Division by zero or near-zero determinant in inverse().");
            // Alternative: Return a signaling NaN or a specific error state object
            // return {std::numeric_limits<T>::quiet_NaN(),
            //         std::numeric_limits<T>::quiet_NaN(),
            //         std::numeric_limits<T>::quiet_NaN()};
        }

        // Calculate inverse determinant (1/det)
        T inv_det = T(1.0) / det;

        // Optional: Apply Newton-Raphson refinement for potentially better precision of inv_det
        // This can help especially if |det| is very small but non-zero.
        // Be cautious as it adds computation cost.
        // for (int i = 0; i < 2; ++i) { // 1-2 iterations usually suffice
        //     inv_det = inv_det * (T(2.0) - det * inv_det);
        // }

        // Inverse components based on derivation: (a/det, -b/det, -c/det)
        T inv_nominal = nominal * inv_det;
        T inv_noise   = -noise * inv_det;
        // *** CORRECTED DELTA INVERSE CALCULATION ***
        T inv_delta   = -delta * inv_det;

        return {inv_nominal, inv_noise, inv_delta};
    }

    /**
     * @brief Division operator (implemented as multiplication by inverse).
     * @throws std::runtime_error If the divisor `o` is not invertible.
     */
    UncertaintyAlgebra operator/(const UncertaintyAlgebra& o) const {
        // Calculate inverse of the divisor first
        UncertaintyAlgebra o_inv = o.inverse();
        return (*this) * o_inv; // Multiply by the inverse
    }

    /**
     * @brief In-place division operator.
     * @throws std::runtime_error If the divisor `o` is not invertible.
     */
    UncertaintyAlgebra& operator/=(const UncertaintyAlgebra& o) {
        UncertaintyAlgebra o_inv = o.inverse();
        *this *= o_inv; // Use in-place multiplication
        return *this;
    }

    /**
     * @brief Scalar multiplication operator (element * scalar).
     */
    UncertaintyAlgebra operator*(const T scalar) const {
        return {nominal * scalar, noise * scalar, delta * scalar};
    }

    /**
     * @brief Friend function for scalar multiplication from the left (scalar * element).
     */
    friend UncertaintyAlgebra operator*(const T scalar, const UncertaintyAlgebra& field) {
        // Simply reuses the element * scalar operator
        return field * scalar;
    }

    /**
     * @brief Scalar division operator (element / scalar).
     * @throws std::runtime_error If scalar is zero.
     */
    UncertaintyAlgebra operator/(const T scalar) const {
        if (scalar == T(0.0)) {
             throw std::runtime_error("UncertaintyAlgebra: Division by zero scalar.");
        }
        T inv_scalar = T(1.0) / scalar; // Compute inverse once
        return {nominal * inv_scalar, noise * inv_scalar, delta * inv_scalar};
    }

    /**
      * @brief In-place scalar division operator (element /= scalar).
      * @throws std::runtime_error If scalar is zero.
      */
     UncertaintyAlgebra& operator/=(const T scalar) {
         if (scalar == T(0.0)) {
             throw std::runtime_error("UncertaintyAlgebra: Division by zero scalar.");
         }
         T inv_scalar = T(1.0) / scalar;
         nominal *= inv_scalar;
         noise *= inv_scalar;
         delta *= inv_scalar;
         // Reset Kahan terms as scale might change significantly
         c_nom = 0; c_noise = 0; c_delta = 0;
         return *this;
     }


    // --- Comparison Operators ---

    /**
     * @brief Equality comparison operator. Uses a tolerance based on epsilon.
     */
    bool operator==(const UncertaintyAlgebra& o) const {
        // Use a relative tolerance for comparison
        const T epsilon = std::numeric_limits<T>::epsilon() * 100; // Adjust factor as needed

        // Check relative difference or absolute difference if numbers are small
        auto check_close = [&](T v1, T v2) {
            T diff = std::abs(v1 - v2);
            T max_val = std::max({std::abs(v1), std::abs(v2), T(1.0)}); // Avoid issues near zero
            return diff < epsilon * max_val;
            // Alternative: Check absolute difference if values are very small
            // return diff <= epsilon || diff < epsilon * std::abs(v1 + v2) / 2.0;
        };

        return check_close(nominal, o.nominal) &&
               check_close(noise, o.noise) &&
               check_close(delta, o.delta);
    }

    /**
     * @brief Inequality comparison operator.
     */
    bool operator!=(const UncertaintyAlgebra& o) const {
        return !(*this == o);
    }

    // --- Accessors ---

    /**
     * @brief Explicit conversion operator to the base scalar type (returns nominal value).
     */
    explicit operator T() const {
        return nominal;
    }

    /**
     * @brief Gets the nominal value component.
     */
    T getValue() const {
        return nominal;
    }

    /**
     * @brief Gets the noise component.
     * The interpretation of this value as "uncertainty" (e.g., std dev)
     * depends on the specific application and how sigma2 is used.
     * This returns the raw 'noise' component 'b'.
     */
    T getNoiseComponent() const {
        return noise;
    }

    /**
     * @brief Gets the derivative component.
     */
    T getDerivative() const {
        return delta;
    }

    // --- Verification ---

    /**
     * @brief Checks key algebraic properties (useful for debugging).
     * Note: This structure is NOT associative under multiplication.
     * @param other Another element to use for testing properties like distributivity.
     * @return True if tested properties hold within tolerance, False otherwise.
     */
    bool verifyAlgebraicProperties(const UncertaintyAlgebra& other) const {
        const T epsilon = std::numeric_limits<T>::epsilon() * 1000; // Wider tolerance for complex checks

        // Create identity elements
        auto one = UncertaintyAlgebra::one();
        auto zero = UncertaintyAlgebra::zero();

        bool properties_hold = true;
        std::string error_msg = "";

        // 1. Distributivity: this * (other + one) == this * other + this * one
        try {
            auto left_dist = (*this) * (other + one);
            auto right_dist = (*this) * other + (*this) * one;
            if (left_dist != right_dist) {
                 properties_hold = false;
                 error_msg += "Distributivity FAILED. ";
            }
        } catch (const std::exception& e) {
             properties_hold = false;
             error_msg += "Exception during distributivity check. ";
        }


        // 2. Multiplicative Inverse: this * this.inverse() == one (if invertible)
        T det = nominal * nominal - sigma2 * noise * noise;
        T tolerance = epsilon * (std::abs(nominal * nominal) + std::abs(sigma2 * noise * noise));
        tolerance = std::max(tolerance, std::numeric_limits<T>::epsilon());

        if (std::abs(det) >= tolerance) { // Check if likely invertible
             try {
                auto inv = this->inverse();
                auto prod = (*this) * inv;
                if (prod != one) {
                    properties_hold = false;
                    error_msg += "Multiplicative Inverse FAILED (prod != one). ";
                }
             } catch (const std::exception& e) {
                 properties_hold = false;
                 error_msg += "Exception during inverse check: " + std::string(e.what()) + ". ";
             }
        } else {
             // Cannot check inverse if determinant is near zero. This is expected.
             // error_msg += "Skipped inverse check (near-zero determinant). ";
        }

        // 3. Additive Inverse: this + (-this) == zero
        try {
            if ((*this) + (-*this) != zero) {
                properties_hold = false;
                error_msg += "Additive Inverse FAILED. ";
            }
        } catch (const std::exception& e) {
             properties_hold = false;
             error_msg += "Exception during additive inverse check. ";
        }


        // 4. Associativity (FAILURE EXPECTED): (this * other) * one != this * (other * one)
        // We expect this to fail, demonstrating non-associativity.
        try {
            auto left_assoc = ((*this) * other) * one; // Should simplify to (*this) * other
            auto right_assoc = (*this) * (other * one); // Should simplify to (*this) * other
            // In theory, if 'one' is identity, both sides should be equal.
            // Let's test with a non-identity element too.
             auto third = UncertaintyAlgebra(0.5, 0.2, 0.1); // Another element
             left_assoc = ((*this) * other) * third;
             right_assoc = (*this) * (other * third);
             if (left_assoc != right_assoc) {
                 // This is EXPECTED TO FAIL unless sigma2=0 or noise components are zero.
                 // We don't set properties_hold = false here, just note it.
                 // error_msg += "Associativity Check: As expected, (a*b)*c != a*(b*c). ";
             }
        } catch (const std::exception& e) {
             // Handle potential exceptions during associativity check if needed
             error_msg += "Exception during associativity check. ";
        }


        // Optional: Output error message if verification fails
        // if (!properties_hold) {
        //     std::cerr << "Verification failed: " << error_msg << std::endl;
        // }

        return properties_hold;
    }
};


/**
 * @brief Simple 1D Kalman filter using the UncertaintyAlgebra class.
 *
 * WARNING: The use of UncertaintyAlgebra for state (x), covariance (p),
 * and noise parameters (q, r) is highly non-standard. A standard Kalman filter
 * uses real numbers/matrices. The interpretation and validity of propagating
 * the 'noise' and 'delta' components through the KF equations in this manner
 * requires careful justification based on the specific problem domain.
 * This implementation assumes Identity state transition (F=1) and measurement (H=1) models.
 */
template<typename Field> // Field should be an instantiation of UncertaintyAlgebra
class KalmanFilterWithAlgebra {
public:
    /**
     * @brief Constructor.
     * @param q_ Process noise variance represented as a Field object.
     * @param r_ Measurement noise variance represented as a Field object.
     * @param p0_ Initial estimate covariance represented as a Field object.
     * @param x0_ Initial state estimate represented as a Field object.
     */
    KalmanFilterWithAlgebra(Field q_, Field r_, Field p0_, Field x0_)
        : q(q_), r(r_), p(p0_), x(x0_), is_initialized(true)
    {
        // Initial validation (optional)
        if (r.getValue() <= 0) {
             // Warning or error: Measurement noise variance should be positive
        }
    }

    /**
     * @brief Updates the filter with a new measurement.
     * @param zRaw The raw scalar measurement value.
     * @return The nominal value of the updated state estimate.
     * @throws std::runtime_error If division by zero occurs (e.g., p+r is non-invertible).
     */
    T update(T zRaw) {
         if (!is_initialized) {
             // Handle case where filter wasn't properly initialized
             // Maybe initialize here based on the first measurement?
             // Or throw an error.
             throw std::runtime_error("KalmanFilterWithAlgebra not initialized.");
         }

        // Convert raw measurement to the Field type.
        // Assumes measurement itself has zero 'noise' and 'delta' components
        // *within the algebra's structure*. The measurement noise is handled by 'r'.
        Field z = Field::fromScalar(zRaw);

        // --- Prediction Step ---
        // Predict state (simple identity model: x_k^- = x_{k-1}^+)
        // No change to x in this simple model. x = x;

        // Predict covariance (simple identity model: p_k^- = p_{k-1}^+ + q)
        p += q; // Use Kahan-compensated addition

        // --- Update Step ---
        // Calculate innovation (measurement residual)
        Field innovation = z - x; // Uses non-Kahan subtraction

        // Calculate innovation covariance (S = H*P*H' + R -> S = p + r for H=1)
        Field s = p + r; // Uses non-Kahan addition

        // Calculate optimal Kalman gain (K = P*H'/S -> K = p / s for H=1)
        // This is the most likely place for exceptions if s is not invertible
        try {
            k = p / s; // Division uses inverse() internally
        } catch (const std::runtime_error& e) {
            // Handle non-invertible s (e.g., p+r is near zero).
            // This might indicate a problem with noise parameters or filter divergence.
            // Options: Skip update, reset filter, clamp gain, re-throw, etc.
            // For now, re-throw to signal the issue.
            throw std::runtime_error("Kalman gain calculation failed: " + std::string(e.what()));
        }


        // Update state estimate (x_k^+ = x_k^- + K * innovation)
        x += k * innovation; // Use Kahan-compensated addition

        // Update error covariance (Joseph form for stability: p_k^+ = (I-KH)P(I-KH)' + KRK')
        // Simplifies to p = (1 - k) * p * (1 - k) + k * r * k for H=1
        Field one_minus_k = Field::one() - k;
        // Note: These multiplications do not use Kahan summation.
        p = one_minus_k * p * one_minus_k + k * r * k;

        // Return the nominal value of the updated state
        return x.getValue();
    }

    /**
     * @brief Gets the nominal value of the current state estimate.
     */
    T getEstimateValue() const {
        return x.getValue();
    }

    /**
     * @brief Gets the 'noise' component of the current state estimate.
     * Interpretation depends on application context.
     */
    T getEstimateNoiseComponent() const {
        return x.getNoiseComponent();
    }

     /**
      * @brief Gets the 'derivative' component of the current state estimate.
      */
     T getEstimateDerivative() const {
         return x.getDerivative();
     }


    /**
     * @brief Gets an estimate of uncertainty, typically sqrt of the nominal covariance.
     * WARNING: This interprets only the 'nominal' part of the Field 'p' as the variance.
     * It ignores the 'noise' and 'delta' components of 'p'. The validity of this
     * depends heavily on whether the 'nominal' component correctly tracks the variance
     * despite the custom algebra.
     * @return Standard deviation estimate (sqrt(p.nominal)). Returns NaN if p.nominal is negative.
     */
    T getUncertaintyEstimate() const {
        T nominal_p = p.getValue();
        if (nominal_p < 0) {
            // Covariance should not be negative. Indicates numerical instability or model issues.
            // Return NaN or handle as an error.
            return std::numeric_limits<T>::quiet_NaN();
        }
        return std::sqrt(nominal_p);
    }

    /**
     * @brief Resets the filter state and covariance.
     * @param initialValue The nominal value to reset the state estimate to.
     * @param initialCovariance The Field object representing the initial covariance (e.g., Field(1.0)).
     */
    void reset(T initialValue, Field initialCovariance) {
        x = Field::fromScalar(initialValue);
        p = initialCovariance;
        // Reset Kahan terms for x and p
        x.c_nom = x.c_noise = x.c_delta = 0;
        p.c_nom = p.c_noise = p.c_delta = 0;
        // Gain k will be recalculated on the next update.
        is_initialized = true;
    }

    /**
     * @brief Gets the full state estimate as a Field object.
     */
    Field getState() const {
        return x;
    }

    /**
      * @brief Gets the full covariance estimate as a Field object.
      */
    Field getCovariance() const {
        return p;
    }


private:
    using T = typename Field::value_type; // Assuming Field exposes its underlying type T

    Field q; // Process noise covariance (as Field object)
    Field r; // Measurement noise covariance (as Field object)
    Field p; // Estimate error covariance (as Field object)
    Field x; // State estimate (as Field object)
    Field k; // Kalman gain (as Field object)
    bool is_initialized = false;
};


// --- Example Usage (Conditional Compilation for Arduino) ---
#ifdef ARDUINO
#include <Arduino.h>

// Define the underlying type and sigma2 parameter
using DataType = double;
constexpr DataType SIGMA2 = 0.0025; // Example value for sigma^2

// Use the UncertaintyAlgebra with the chosen parameters
using AlgebraField = UncertaintyAlgebra<DataType, SIGMA2>;

// Create a Kalman filter instance using the algebra
// NOTE: Justification needed for representing Q, R, P0 as AlgebraField objects.
// Standard KF uses scalar variances. Here we initialize noise/delta to 0.
KalmanFilterWithAlgebra<AlgebraField> filter(
    AlgebraField(0.001, 0, 0),  // Process noise variance Q (nominal=0.001)
    AlgebraField(0.01, 0, 0),   // Measurement noise variance R (nominal=0.01)
    AlgebraField(1.0, 0, 0),    // Initial error covariance P0 (nominal=1.0)
    AlgebraField(0.0, 0, 0)     // Initial state estimate x0 (nominal=0.0)
);

// Configuration
const int SENSOR_PIN = 34; // ADC pin (adjust for your board, e.g., A0 for Uno)
const double ADC_MAX_VALUE = 4095.0; // For ESP32 ADC
const double VOLTAGE_REF = 3.3;      // ESP32 voltage reference
const unsigned long SAMPLE_INTERVAL_MS = 100; // Sample at 10Hz
unsigned long lastSampleTime = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for Serial connection

    // Note: ESP32 ADC pins don't need pinMode set explicitly for INPUT.
    // For other boards (like Arduino Uno), you might need:
    // pinMode(SENSOR_PIN, INPUT);

    Serial.println("--- Uncertainty Algebra Kalman Filter ---");
    Serial.println("Note: Using a non-standard algebraic structure for KF.");
    Serial.println("Time(ms),RawVoltage,FilteredVoltage,EstUncertainty(sqrt(P_nominal))");

    lastSampleTime = millis();
}

void loop() {
    unsigned long currentTime = millis();
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;

        // Read sensor and convert to voltage
        // Add error handling for analogRead if necessary
        int rawValue = analogRead(SENSOR_PIN);
        double voltage = static_cast<DataType>(rawValue) * (VOLTAGE_REF / ADC_MAX_VALUE);

        // Update filter
        try {
            double estimate = filter.update(voltage);
            double uncertainty = filter.getUncertaintyEstimate(); // sqrt(p.nominal)

            // Output in CSV format
            Serial.print(currentTime);
            Serial.print(",");
            Serial.print(voltage, 6); // Print with 6 decimal places
            Serial.print(",");
            Serial.print(estimate, 6);
            Serial.print(",");
            Serial.println(uncertainty, 6);

        } catch (const std::runtime_error& e) {
            Serial.print("ERROR at Time(ms) ");
            Serial.print(currentTime);
            Serial.print(": ");
            Serial.println(e.what());
            // Optional: Reset filter or take other corrective action
            // filter.reset(0.0, AlgebraField(1.0, 0, 0)); // Example reset
        }
    }
    // Allow other tasks to run (important in more complex sketches)
    delay(1); // Small delay
}
#endif // ARDUINO

