#include <cmath>
#include <limits>
#include <type_traits>
#include <stdexcept>   // For runtime_error
#include <concepts>    // For concepts if using C++20
#include <ostream>     // For operator<<
#include <iomanip>     // For std::setprecision in operator<<
#include <string>      // For exception messages
#include <algorithm>   // For std::max

// Concept to check if a type is a floating point type (C++20)
// template<typename T>
// concept FloatingPoint = std::is_floating_point_v<T>;

/**
 * @brief UncertaintyAlgebra - Represents a number with uncertainty and derivative components.
 *
 * Defines a commutative R-algebra structure over triples (nominal, noise, delta).
 * WARNING: This structure is NOT a field because multiplication is NOT associative
 * due to the sigma2 term unless sigma2 is zero or noise components are zero.
 * (a * b) * c != a * (b * c) in general.
 *
 * The multiplication rule is:
 * (a, b, c) * (a', b', c') = (a*a' + sigma2*b*b', a*b' + a'*b, a*c' + a'*c)
 *
 * Components:
 * - 'nominal': The base value (analogous to a real number).
 * - 'noise': A component intended for uncertainty propagation. Its precise meaning
 * (e.g., related to variance or standard deviation) depends on the
 * application and the interpretation of sigma2.
 * - 'delta': A component behaving like the infinitesimal part of a dual number
 * (epsilon where epsilon^2 = 0), useful for automatic differentiation.
 * - 'sigma2': A non-negative compile-time parameter scaling the noise interaction
 * in multiplication.
 */
template<typename T = double, T sigma2_param = 0.01>
// requires FloatingPoint<T> // C++20 constraint
class UncertaintyAlgebra {

    // --- Static Assertions ---
    static_assert(std::is_floating_point_v<T>,
                  "Template parameter T must be a floating-point type (float, double, long double).");
    static_assert(sigma2_param >= T(0.0),
                  "Template parameter sigma2_param must be non-negative.");

public:
    // --- Type Alias ---
    using value_type = T;

    // --- Public Constants ---
    static constexpr T sigma2 = sigma2_param; // Make sigma2 accessible

    // --- Member Variables ---
    T nominal; // Base value
    T noise;   // Uncertainty-related component
    T delta;   // Derivative component (dual part)

    // Kahan summation compensation terms (only used for += and -=)
    // These help mitigate precision loss in repeated additions/subtractions.
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

    /** @brief Creates the multiplicative identity element (1, 0, 0). */
    static constexpr UncertaintyAlgebra one() {
        return UncertaintyAlgebra(1, 0, 0);
    }

    /** @brief Creates the additive identity element (0, 0, 0). */
    static constexpr UncertaintyAlgebra zero() {
        return UncertaintyAlgebra(0, 0, 0);
    }

    /** @brief Creates an element from a scalar value (value, 0, 0). */
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
        T y = value - comp;
        T t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }

    // --- Arithmetic Operators ---

    /**
     * @brief Addition operator. Returns a new object.
     * Note: Does not use Kahan summation (minimal benefit on temporaries).
     */
    UncertaintyAlgebra operator+(const UncertaintyAlgebra& o) const {
        // Beware of potential intermediate overflow/underflow if components are large.
        return UncertaintyAlgebra(nominal + o.nominal,
                                  noise + o.noise,
                                  delta + o.delta);
    }

    /**
     * @brief Subtraction operator. Returns a new object.
     * Note: Does not use Kahan summation.
     */
    UncertaintyAlgebra operator-(const UncertaintyAlgebra& o) const {
        return UncertaintyAlgebra(nominal - o.nominal,
                                  noise - o.noise,
                                  delta - o.delta);
    }

    /** @brief In-place addition operator with Kahan compensation. */
    UncertaintyAlgebra& operator+=(const UncertaintyAlgebra& o) {
        kahanAdd(nominal, c_nom, o.nominal);
        kahanAdd(noise, c_noise, o.noise);
        kahanAdd(delta, c_delta, o.delta);
        return *this;
    }

    /** @brief In-place subtraction operator with Kahan compensation. */
    UncertaintyAlgebra& operator-=(const UncertaintyAlgebra& o) {
        kahanAdd(nominal, c_nom, -o.nominal);
        kahanAdd(noise, c_noise, -o.noise);
        kahanAdd(delta, c_delta, -o.delta);
        return *this;
    }

    /**
     * @brief Multiplication operator.
     * Uses FMA (fused multiply-add) if available for float/double.
     * WARNING: Multiplication is NOT associative if sigma2 != 0 and noise != 0.
     * WARNING: Potential for overflow/underflow, especially with large sigma2 or component values.
     */
    UncertaintyAlgebra operator*(const UncertaintyAlgebra& o) const {
        T res_nominal, res_noise, res_delta;

        // Multiplication rules:
        // nominal = a*a' + sigma2*b*b'
        // noise   = a*b' + a'*b
        // delta   = a*c' + a'*c

        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            // FMA can improve precision and performance
            res_nominal = std::fma(nominal, o.nominal, sigma2 * noise * o.noise);
            res_noise   = std::fma(nominal, o.noise, o.nominal * noise);
            res_delta   = std::fma(nominal, o.delta, o.nominal * delta);
        } else {
            // Standard operations for other types (e.g., long double)
            res_nominal = nominal * o.nominal + sigma2 * noise * o.noise;
            res_noise   = nominal * o.noise + o.nominal * noise;
            res_delta   = nominal * o.delta + o.nominal * delta;
        }
        return {res_nominal, res_noise, res_delta};
    }

    /**
     * @brief In-place multiplication operator.
     * Note: Resets Kahan compensation terms.
     */
    UncertaintyAlgebra& operator*=(const UncertaintyAlgebra& o) {
        UncertaintyAlgebra result = (*this) * o; // Calculate result first
        *this = result; // Assign (copies members, resets Kahan terms implicitly)
        // Explicitly clear Kahan terms after assignment for clarity
        c_nom = 0; c_noise = 0; c_delta = 0;
        return *this;
    }

    /** @brief Additive inverse (negation). */
    constexpr UncertaintyAlgebra operator-() const {
        return {-nominal, -noise, -delta};
    }

    /**
     * @brief Calculates the multiplicative inverse: `inv(X)` such that `X * inv(X) = 1`.
     * Throws std::runtime_error if the element is not invertible (determinant is near zero).
     * WARNING: Potential for overflow/underflow if determinant is very small or components are large.
     * @return The multiplicative inverse.
     * @throws std::runtime_error If the determinant `nominal^2 - sigma2 * noise^2` is too close to zero.
     */
    UncertaintyAlgebra inverse() const {
        // Determinant for invertibility: det = a^2 - sigma2 * b^2
        // This arises from solving the system:
        // ax + sigma2*by = 1
        // ay + bx = 0
        // az + cx = 0  (where (x,y,z) is the inverse)
        T term1 = nominal * nominal;
        T term2 = sigma2 * noise * noise;
        T det = term1 - term2;

        // Check for near-zero determinant using relative and absolute tolerance
        const T abs_epsilon = std::numeric_limits<T>::epsilon();
        const T rel_epsilon_factor = 100.0; // Factor for relative check, adjust if needed
        // Tolerance depends on the magnitude of the terms involved
        T tolerance = std::max(abs_epsilon, rel_epsilon_factor * abs_epsilon * (std::abs(term1) + std::abs(term2)));

        if (std::abs(det) < tolerance) {
            throw std::runtime_error("UncertaintyAlgebra::inverse() - Division by near-zero determinant: "
                                     "det = " + std::to_string(det) + ", tolerance = " + std::to_string(tolerance));
        }

        // Calculate inverse determinant (1/det)
        T inv_det = T(1.0) / det;

        // Optional: Newton-Raphson refinement for inv_det (can improve precision)
        // inv_det = inv_det * (T(2.0) - det * inv_det); // One iteration
        // inv_det = inv_det * (T(2.0) - det * inv_det); // Second iteration

        // Inverse components derived from solving the system: (a/det, -b/det, -c/det)
        T inv_nominal = nominal * inv_det;
        T inv_noise   = -noise * inv_det;
        T inv_delta   = -delta * inv_det; // Corrected formula

        return {inv_nominal, inv_noise, inv_delta};
    }

    /**
     * @brief Division operator (X / Y = X * inv(Y)).
     * @throws std::runtime_error If the divisor `o` is not invertible.
     */
    UncertaintyAlgebra operator/(const UncertaintyAlgebra& o) const {
        return (*this) * o.inverse(); // Multiplication handles potential FMA usage
    }

    /**
     * @brief In-place division operator.
     * @throws std::runtime_error If the divisor `o` is not invertible.
     */
    UncertaintyAlgebra& operator/=(const UncertaintyAlgebra& o) {
        *this *= o.inverse(); // Reuses *= which resets Kahan terms
        return *this;
    }

    /** @brief Scalar multiplication operator (element * scalar). */
    UncertaintyAlgebra operator*(const T scalar) const {
        return {nominal * scalar, noise * scalar, delta * scalar};
    }

    /** @brief Friend function for scalar multiplication from the left (scalar * element). */
    friend UncertaintyAlgebra operator*(const T scalar, const UncertaintyAlgebra& field) {
        return field * scalar; // Reuse element * scalar operator
    }

    /**
     * @brief Scalar division operator (element / scalar).
     * @throws std::runtime_error If scalar is zero.
     */
    UncertaintyAlgebra operator/(const T scalar) const {
        if (scalar == T(0.0)) {
             throw std::runtime_error("UncertaintyAlgebra::operator/ - Division by zero scalar.");
        }
        // Compute inverse once to potentially save a division operation
        T inv_scalar = T(1.0) / scalar;
        return {nominal * inv_scalar, noise * inv_scalar, delta * inv_scalar};
    }

    /**
      * @brief In-place scalar division operator (element /= scalar).
      * @throws std::runtime_error If scalar is zero.
      */
     UncertaintyAlgebra& operator/=(const T scalar) {
         if (scalar == T(0.0)) {
             throw std::runtime_error("UncertaintyAlgebra::operator/= - Division by zero scalar.");
         }
         T inv_scalar = T(1.0) / scalar;
         nominal *= inv_scalar;
         noise   *= inv_scalar;
         delta   *= inv_scalar;
         // Reset Kahan terms as scale changes
         c_nom = 0; c_noise = 0; c_delta = 0;
         return *this;
     }

    // --- Comparison Operators ---

    /**
     * @brief Equality comparison operator. Uses relative tolerance.
     * Compares all three components (nominal, noise, delta).
     */
    bool operator==(const UncertaintyAlgebra& o) const {
        const T abs_epsilon = std::numeric_limits<T>::epsilon();
        const T rel_epsilon_factor = 100.0; // Adjust as needed

        auto check_close = [&](T v1, T v2) {
            T diff = std::abs(v1 - v2);
            // Use absolute tolerance for values near zero, relative otherwise
            T tolerance = std::max(abs_epsilon, rel_epsilon_factor * abs_epsilon * std::max(std::abs(v1), std::abs(v2)));
            // Alternative simpler check:
            // T max_val = std::max({std::abs(v1), std::abs(v2), T(1.0)}); // Avoid division by zero
            // T tolerance = rel_epsilon_factor * abs_epsilon * max_val;
            return diff <= tolerance;
        };

        return check_close(nominal, o.nominal) &&
               check_close(noise, o.noise) &&
               check_close(delta, o.delta);
    }

    /** @brief Inequality comparison operator. */
    bool operator!=(const UncertaintyAlgebra& o) const {
        return !(*this == o);
    }

    // --- Accessors ---

    /** @brief Explicit conversion operator to the base scalar type (returns nominal value). */
    explicit operator T() const {
        return nominal;
    }

    /** @brief Gets the nominal value component. */
    T getValue() const { return nominal; }

    /** @brief Gets the raw noise component 'b'. Interpretation depends on context. */
    T getNoiseComponent() const { return noise; }

    /** @brief Gets the derivative component 'c'. */
    T getDerivative() const { return delta; }

    // --- Stream Output ---

    /**
     * @brief Stream insertion operator for easy printing.
     * @param os Output stream.
     * @param ua UncertaintyAlgebra object.
     * @return Reference to the output stream.
     */
    friend std::ostream& operator<<(std::ostream& os, const UncertaintyAlgebra& ua) {
        // Store previous precision and set a new one for consistent output
        std::streamsize old_precision = os.precision();
        os << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10);
        os << "(" << ua.nominal << ", " << ua.noise << ", " << ua.delta << ")";
        // Restore previous precision
        os.precision(old_precision);
        os.unsetf(std::ios_base::floatfield); // Unset fixed format
        return os;
    }

    // --- Verification (for debugging) ---
    // (verifyAlgebraicProperties method remains largely the same as before)
    // ... see previous version for the implementation ...
    bool verifyAlgebraicProperties(const UncertaintyAlgebra& other) const {
        const T epsilon = std::numeric_limits<T>::epsilon() * 1000; // Wider tolerance

        auto one = UncertaintyAlgebra::one();
        auto zero = UncertaintyAlgebra::zero();
        bool properties_hold = true;

        // Simplified check structure
        auto check_property = [&](bool condition, const std::string& name) {
            if (!condition) {
                properties_hold = false;
                // std::cerr << "Verification FAILED: " << name << std::endl; // Optional debug output
            }
        };

        try { check_property((*this) * (other + one) == (*this) * other + (*this) * one, "Distributivity"); }
        catch (...) { check_property(false, "Distributivity (Exception)"); }

        T det = nominal * nominal - sigma2 * noise * noise;
        T tolerance = epsilon * (std::abs(nominal * nominal) + std::abs(sigma2 * noise * noise));
        if (std::abs(det) > std::max(tolerance, std::numeric_limits<T>::epsilon())) {
             try { check_property((*this) * this->inverse() == one, "Multiplicative Inverse"); }
             catch (...) { check_property(false, "Multiplicative Inverse (Exception)"); }
        }

        try { check_property((*this) + (-*this) == zero, "Additive Inverse"); }
        catch (...) { check_property(false, "Additive Inverse (Exception)"); }

        // Associativity (EXPECTED TO FAIL) - uncomment to observe
        // try {
        //     auto third = UncertaintyAlgebra(0.5, 0.2, 0.1);
        //     check_property(((*this) * other) * third == (*this) * (other * third), "Associativity");
        // } catch (...) { check_property(false, "Associativity (Exception)"); }

        return properties_hold;
    }

};


/**
 * @brief Simple 1D Kalman filter using an UncertaintyAlgebra type.
 *
 * WARNING: This implementation makes VERY STRONG assumptions:
 * 1.  System is 1-Dimensional.
 * 2.  State Transition Model is Identity (F=1, x_k = x_{k-1} + noise).
 * 3.  Measurement Model is Identity (H=1, z_k = x_k + noise).
 * 4.  The use of UncertaintyAlgebra for state (x), covariance (p), and noise
 * parameters (q, r) is NON-STANDARD. Standard KFs use real numbers/matrices.
 *
 * INTERPRETATION WARNING:
 * - The validity of propagating the 'noise' and 'delta' components through the
 * KF equations in this manner depends entirely on the specific problem domain
 * and a clear mathematical justification for this custom algebra's application.
 * - The 'delta' component (for derivatives) is NOT USED for linearization as in
 * an Extended Kalman Filter (EKF) because the F=1, H=1 models are linear.
 * Its propagation here follows the algebra rules but may not correspond to a
 * standard derivative propagation in a filtering context.
 * - The `getUncertaintyEstimate()` method returns `sqrt(p.nominal)`, IGNORING
 * the `p.noise` and `p.delta` components. This assumes `p.nominal` adequately
 * represents the variance, which requires careful validation.
 */
template<typename Field> // Field should be an instantiation of UncertaintyAlgebra
class KalmanFilterWithAlgebra {
public:
    // --- Type Alias ---
    using T = typename Field::value_type;

    // --- Constructors ---

    /** @brief Default constructor. Filter must be initialized before use. */
    KalmanFilterWithAlgebra() : is_initialized(false) {}

    /**
     * @brief Constructor with immediate initialization.
     * @param q_ Process noise variance (as Field object, nominal part should be > 0).
     * @param r_ Measurement noise variance (as Field object, nominal part should be > 0).
     * @param p0_ Initial estimate covariance (as Field object, nominal part should be >= 0).
     * @param x0_ Initial state estimate (as Field object).
     * @throws std::invalid_argument if nominal parts of q or r are non-positive, or p0 is negative.
     */
    KalmanFilterWithAlgebra(Field q_, Field r_, Field p0_, Field x0_)
        : is_initialized(false) // Initialize sets this to true
    {
        initialize(q_, r_, p0_, x0_);
    }

    // --- Initialization ---

    /**
     * @brief Initializes or re-initializes the filter state and parameters.
     * @param q_ Process noise variance (as Field object, nominal part should be > 0).
     * @param r_ Measurement noise variance (as Field object, nominal part should be > 0).
     * @param p0_ Initial estimate covariance (as Field object, nominal part should be >= 0).
     * @param x0_ Initial state estimate (as Field object).
     * @throws std::invalid_argument if nominal parts of q or r are non-positive, or p0 is negative.
     */
    void initialize(Field q_, Field r_, Field p0_, Field x0_) {
        if (q_.getValue() <= T(0.0)) {
            throw std::invalid_argument("KalmanFilterWithAlgebra::initialize - Process noise variance (q.nominal) must be positive.");
        }
        if (r_.getValue() <= T(0.0)) {
            throw std::invalid_argument("KalmanFilterWithAlgebra::initialize - Measurement noise variance (r.nominal) must be positive.");
        }
        if (p0_.getValue() < T(0.0)) {
            throw std::invalid_argument("KalmanFilterWithAlgebra::initialize - Initial covariance (p0.nominal) cannot be negative.");
        }

        q = q_;
        r = r_;
        p = p0_;
        x = x0_;

        // Reset Kahan terms explicitly on initialization/reset
        x.c_nom = x.c_noise = x.c_delta = 0;
        p.c_nom = p.c_noise = p.c_delta = 0;
        // q and r are typically constant, Kahan terms not relevant unless they are updated.
        // k (gain) is calculated fresh each time.

        is_initialized = true;
    }


    // --- Update Step ---

    /**
     * @brief Updates the filter with a new scalar measurement.
     * @param zRaw The raw scalar measurement value.
     * @return The nominal value of the updated state estimate.
     * @throws std::runtime_error If the filter is not initialized or if division by zero
     * occurs during gain calculation (e.g., p+r is non-invertible).
     */
    T update(T zRaw) {
         if (!is_initialized) {
             throw std::runtime_error("KalmanFilterWithAlgebra::update - Filter not initialized.");
         }

        // Convert measurement to Field type (assuming zero noise/delta for the measurement itself)
        Field z = Field::fromScalar(zRaw);

        // --- Prediction Step (Assumes F=1) ---
        // State prediction: x_k^- = x_{k-1}^+ (no change)
        // Covariance prediction: p_k^- = p_{k-1}^+ + q
        p += q; // Uses Kahan-compensated addition

        // --- Update Step (Assumes H=1) ---
        // Innovation (residual): y = z - H*x_k^- = z - x
        Field innovation = z - x; // Uses non-Kahan subtraction

        // Innovation covariance: S = H*P*H' + R = p + r
        Field s = p + r; // Uses non-Kahan addition

        // Optimal Kalman gain: K = P*H'/S = p / s
        try {
            k = p / s; // Division uses inverse() internally
        } catch (const std::runtime_error& e) {
            // Catch potential division by zero from p / s
            throw std::runtime_error("KalmanFilterWithAlgebra::update - Kalman gain calculation failed: " + std::string(e.what()));
        }

        // Updated state estimate: x_k^+ = x_k^- + K * y
        x += k * innovation; // Uses Kahan-compensated addition

        // Updated error covariance (Joseph form): p_k^+ = (I-KH)P(I-KH)' + KRK'
        // Simplifies to p = (1 - k) * p * (1 - k) + k * r * k for H=1
        Field one_minus_k = Field::one() - k;
        // Note: These multiplications do not use Kahan summation.
        // Potential precision loss here, especially if k is close to 1 or 0.
        p = one_minus_k * p * one_minus_k + k * r * k;

        // Return the nominal value of the updated state
        return x.getValue();
    }

    // --- Accessors ---

    /** @brief Gets the nominal value of the current state estimate. */
    T getEstimateValue() const {
        if (!is_initialized) return std::numeric_limits<T>::quiet_NaN(); // Or throw
        return x.getValue();
    }

    /** @brief Gets the 'noise' component of the current state estimate. Interpretation depends on context. */
    T getEstimateNoiseComponent() const {
        if (!is_initialized) return std::numeric_limits<T>::quiet_NaN(); // Or throw
        return x.getNoiseComponent();
    }

     /** @brief Gets the 'derivative' component of the current state estimate. */
     T getEstimateDerivative() const {
         if (!is_initialized) return std::numeric_limits<T>::quiet_NaN(); // Or throw
         return x.getDerivative();
     }

    /**
     * @brief Gets an estimate of uncertainty, typically sqrt of the nominal covariance.
     * WARNING: Interprets ONLY the 'nominal' part of the Field 'p' as the variance.
     * Ignores `p.noise` and `p.delta`. Validity requires careful justification.
     * @return Standard deviation estimate (sqrt(p.nominal)). Returns NaN if not initialized or p.nominal < 0.
     */
    T getUncertaintyEstimate() const {
        if (!is_initialized) return std::numeric_limits<T>::quiet_NaN();
        T nominal_p = p.getValue();
        if (nominal_p < 0) {
            // Should not happen with Joseph form if Q, R >= 0, but check anyway.
            return std::numeric_limits<T>::quiet_NaN();
        }
        return std::sqrt(nominal_p);
    }

    /** @brief Gets the full state estimate as a Field object. */
    Field getState() const {
        if (!is_initialized) throw std::runtime_error("Filter not initialized");
        return x;
    }

    /** @brief Gets the full covariance estimate as a Field object. */
    Field getCovariance() const {
         if (!is_initialized) throw std::runtime_error("Filter not initialized");
        return p;
    }

    /** @brief Checks if the filter has been initialized. */
    bool isInitialized() const {
        return is_initialized;
    }

private:
    Field q {}; // Process noise covariance (as Field object)
    Field r {}; // Measurement noise covariance (as Field object)
    Field p {}; // Estimate error covariance (as Field object)
    Field x {}; // State estimate (as Field object)
    Field k {}; // Kalman gain (as Field object)
    bool is_initialized = false;
};


// --- Example Usage (Conditional Compilation for Arduino) ---
#ifdef ARDUINO
#include <Arduino.h>
#include <limits> // Required for numeric_limits in example

// --- Configuration ---
using DataType = double;
constexpr DataType SIGMA2 = 0.0025; // Example value for sigma^2 in the algebra
const int SENSOR_PIN = 34;          // ADC pin (e.g., ESP32) - ADJUST FOR YOUR BOARD
const double ADC_MAX_VALUE = 4095.0; // Max ADC reading (e.g., 12-bit ESP32)
const double VOLTAGE_REF = 3.3;      // Reference voltage (e.g., ESP32)
const unsigned long SAMPLE_INTERVAL_MS = 100; // Sample at 10Hz

// Filter Noise Parameters (Nominal values)
// These need careful tuning based on the sensor and system dynamics.
const DataType PROCESS_NOISE_VARIANCE = 0.0001; // Q: How much the true value drifts between samples
const DataType MEASUREMENT_NOISE_VARIANCE = 0.01; // R: How much noise is in the sensor reading
const DataType INITIAL_COVARIANCE = 1.0;        // P0: Initial uncertainty about the state

// Use the UncertaintyAlgebra with the chosen parameters
using AlgebraField = UncertaintyAlgebra<DataType, SIGMA2>;

// Create a Kalman filter instance using the algebra
// Initialize with nominal values for Q, R, P0. Noise/Delta components set to 0 initially.
KalmanFilterWithAlgebra<AlgebraField> filter(
    AlgebraField(PROCESS_NOISE_VARIANCE, 0, 0),
    AlgebraField(MEASUREMENT_NOISE_VARIANCE, 0, 0),
    AlgebraField(INITIAL_COVARIANCE, 0, 0),
    AlgebraField(0.0, 0, 0) // Initial state estimate (e.g., 0 Volts)
);

unsigned long lastSampleTime = 0;

void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for Serial Monitor

    // Note: ESP32 ADC pins often don't need pinMode set. Check for your board.
    // pinMode(SENSOR_PIN, INPUT);

    Serial.println("--- Uncertainty Algebra Kalman Filter ---");
    Serial.println("System: 1D, F=1, H=1");
    Serial.println("Algebra: sigma2 = " + String(SIGMA2, 6));
    Serial.println("Filter Params: Qn=" + String(PROCESS_NOISE_VARIANCE, 6) +
                     ", Rn=" + String(MEASUREMENT_NOISE_VARIANCE, 6) +
                     ", P0n=" + String(INITIAL_COVARIANCE, 6));
    Serial.println("WARNING: Non-standard KF approach. Validate results carefully.");
    Serial.println("Time(ms),RawVoltage,FilteredVoltage,EstUncertainty(sqrt(P_nominal))");

    lastSampleTime = millis();
}

void loop() {
    unsigned long currentTime = millis();
    if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
        lastSampleTime = currentTime;

        // Read sensor and convert to voltage
        int rawValue = analogRead(SENSOR_PIN);
        // Basic check for read errors if analogRead could return < 0
        if (rawValue < 0) {
            Serial.println("Error reading sensor!");
            delay(SAMPLE_INTERVAL_MS); // Avoid busy-looping on error
            return;
        }
        double voltage = static_cast<DataType>(rawValue) * (VOLTAGE_REF / ADC_MAX_VALUE);

        // Update filter
        try {
            double estimate = filter.update(voltage);
            double uncertainty = filter.getUncertaintyEstimate();

            // Output in CSV format
            Serial.print(currentTime);
            Serial.print(",");
            Serial.print(voltage, 6); // Print with 6 decimal places
            Serial.print(",");
            Serial.print(estimate, 6);
            Serial.print(",");
            // Handle potential NaN in uncertainty estimate
            if (std::isnan(uncertainty)) {
                 Serial.print("NaN");
            } else {
                 Serial.print(uncertainty, 6);
            }
            Serial.println();

        } catch (const std::exception& e) { // Catch std::runtime_error and others
            Serial.print("ERROR at Time(ms) ");
            Serial.print(currentTime);
            Serial.print(": ");
            Serial.println(e.what());
            // Consider resetting the filter or halting on error
            // filter.initialize(AlgebraField(PROCESS_NOISE_VARIANCE, 0, 0), ...); // Example reset
            delay(1000); // Pause briefly after error
        }
    }
    // Small delay to prevent tight loop if sample interval is short
    // or if other tasks need CPU time. Adjust as needed.
    delay(1);
}
#endif // ARDUINO

