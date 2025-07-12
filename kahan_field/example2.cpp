#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <numeric> // For std::gcd in modular inverse

// Forward declarations
template<typename Field>
class AffinePoint;

template<typename Field>
class RationalMap;

// --- Field Concept and Implementations ---

// Helper for comparing Field types, especially for floating point vs exact
template<typename Field>
struct FieldComparator {
    // Default to exact comparison for non-floating point types
    static bool are_equal(Field a, Field b, double epsilon) {
        return a == b;
    }
};

// Specialization for double to use epsilon comparison
template<>
struct FieldComparator<double> {
    static bool are_equal(double a, double b, double epsilon) {
        return std::abs(a - b) < epsilon;
    }
};

/**
 * @brief Represents a generic Field element.
 * This concept requires:
 * - Default constructor (e.g., Field(0))
 * - Constructor from long long
 * - Operators: +, -, *, /
 * - Comparison operators: ==, !=
 * - Stream insertion operator: <<
 */
template<int P> // P must be a prime number for GF(P) to be a field
class GF_p {
private:
    long long val_;

    // Private helper for modular exponentiation (for modular inverse)
    long long power(long long base, long long exp) const {
        long long res = 1;
        base %= P;
        while (exp > 0) {
            if (exp % 2 == 1) res = (res * base) % P;
            base = (base * base) % P;
            exp /= 2;
        }
        return res;
    }

    // Private helper for modular inverse using Fermat's Little Theorem
    // a^(P-2) mod P is the inverse of a mod P (for prime P)
    long long modInverse(long long n) const {
        assert(n != 0 && "Division by zero in GF_p");
        return power(n, P - 2);
    }

public:
    // Default constructor (represents 0 in the field)
    GF_p(long long val = 0) : val_((val % P + P) % P) {}

    // Arithmetic operators
    GF_p operator+(const GF_p& other) const { return GF_p((val_ + other.val_) % P); }
    GF_p operator-(const GF_p& other) const { return GF_p((val_ - other.val_ + P) % P); }
    GF_p operator*(const GF_p& other) const { return GF_p((val_ * other.val_) % P); }
    GF_p operator/(const GF_p& other) const {
        return GF_p((val_ * modInverse(other.val_)) % P);
    }

    // Compound assignment operators
    GF_p& operator+=(const GF_p& other) { *this = *this + other; return *this; }
    GF_p& operator-=(const GF_p& other) { *this = *this - other; return *this; }
    GF_p& operator*=(const GF_p& other) { *this = *this * other; return *this; }
    GF_p& operator/=(const GF_p& other) { *this = *this / other; return *this; }

    // Comparison operators
    bool operator==(const GF_p& other) const { return val_ == other.val_; }
    bool operator!=(const GF_p& other) const { return val_ != other.val_; }

    // Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const GF_p& f) {
        return os << f.val_;
    }
};


// --- Algebraic Geometry Framework Components ---

/**
 * @brief Represents a point in affine space A^2 over a field.
 * In our Kahan summation context, coordinates are (sum, correction).
 */
template<typename Field>
class AffinePoint {
private:
    Field sum_;
    Field correction_;
    
public:
    // Constructor with default values for sum and correction
    AffinePoint(Field sum = Field(0), Field correction = Field(0)) 
        : sum_(sum), correction_(correction) {}
    
    // Accessors for sum and correction components
    Field sum() const { return sum_; }
    Field correction() const { return correction_; }
    
    // Setters for sum and correction components
    void set_sum(Field s) { sum_ = s; }
    void set_correction(Field c) { correction_ = c; }
    
    /**
     * @brief Calculates the algebraic invariant (the true sum represented by this point).
     * For Kahan summation, this is sum + correction.
     * @return The true sum.
     */
    Field algebraic_invariant() const {
        return sum_ + correction_;
    }
    
    /**
     * @brief Checks if the point's algebraic invariant satisfies an expected sum.
     * Uses FieldComparator for robust comparison.
     * @param expected_sum The sum to compare against.
     * @param epsilon Tolerance for floating-point comparisons.
     * @return True if the invariant is equal to the expected sum within tolerance.
     */
    bool satisfies_invariant(Field expected_sum, double epsilon = 1e-15) const {
        return FieldComparator<Field>::are_equal(algebraic_invariant(), expected_sum, epsilon);
    }
    
    // Stream insertion operator for AffinePoint
    friend std::ostream& operator<<(std::ostream& os, const AffinePoint& p) {
        return os << "(" << p.sum_ << ", " << p.correction_ << ")";
    }
};

/**
 * @brief Represents the rational map φ: A^2 × R → A^2 that defines Kahan summation.
 * This is the core algebraic structure that preserves the summation invariant.
 */
template<typename Field>
class KahanRationalMap {
public:
    /**
     * @brief Applies the rational map: (s, c, x) ↦ (s', c').
     * This map is designed to preserve the algebraic invariant s + c + x = s' + c'.
     * The logic adapts based on whether Field is double (floating-point) or another type (exact).
     * @param point The current affine point (s, c).
     * @param x The value to add.
     * @return The new affine point (s', c').
     */
    AffinePoint<Field> apply(const AffinePoint<Field>& point, Field x) const {
        Field s = point.sum();
        Field c = point.correction();
        
        // Kahan summation steps, adapted for generic Field:
        // In exact arithmetic (like GF_p), the correction c_new will always be 0.
        // This highlights that Kahan's mechanism is specifically for floating-point errors.
        
        Field y = x - c;
        Field t = s + y;
        
        // This is the core of Kahan's compensation: (t - s) - y captures the rounding error.
        // In exact fields, (t - s) - y will always be 0.
        Field c_new = (t - s) - y;
        
        Field s_new = t;
        
        return AffinePoint<Field>(s_new, c_new);
    }
    
    /**
     * @brief Verifies that the rational map preserves the algebraic invariant.
     * @param point The input affine point.
     * @param x The value added.
     * @param epsilon Tolerance for floating-point comparisons.
     * @return True if the invariant is preserved within tolerance.
     */
    bool preserves_invariant(const AffinePoint<Field>& point, Field x, double epsilon = 1e-15) const {
        Field original_invariant = point.algebraic_invariant() + x;
        AffinePoint<Field> result = apply(point, x);
        Field new_invariant = result.algebraic_invariant();
        
        return FieldComparator<Field>::are_equal(original_invariant, new_invariant, epsilon);
    }
};

/**
 * @brief Algebraic Kahan Summation Engine.
 * Uses the geometric framework to perform compensated summation.
 */
template<typename Field>
class AlgebraicKahanSummator {
private:
    AffinePoint<Field> state_;
    KahanRationalMap<Field> rational_map_;
    Field exact_sum_;  // Track the true algebraic sum for verification
    double comparison_epsilon_; // Epsilon for comparisons, specific to this summator
    
public:
    // Constructor initializes state and sets comparison epsilon
    AlgebraicKahanSummator(double epsilon = 1e-15) 
        : state_(Field(0), Field(0)), exact_sum_(Field(0)), comparison_epsilon_(epsilon) {}
    
    /**
     * @brief Add a value using the algebraic geometry framework.
     * @param value The value to add.
     */
    void add(Field value) {
        // Verify invariant before transformation
        // This assertion checks if the current state (s,c) correctly represents the exact_sum_
        assert(state_.satisfies_invariant(exact_sum_, comparison_epsilon_));
        
        // Apply the rational map φ: A^2 × R → A^2
        state_ = rational_map_.apply(state_, value);
        
        // Update exact sum for verification (in exact arithmetic)
        exact_sum_ += value;
        
        // Verify invariant after transformation
        // This assertion checks if the rational map itself preserved the invariant
        // (s_old + c_old + x == s_new + c_new)
        // To do this, we reconstruct the 'previous' state for the map's verification.
        // Note: This assert might be tricky for floating points if 'value' itself
        // was subject to prior rounding errors. For demonstration, it's illustrative.
        assert(rational_map_.preserves_invariant(
            AffinePoint<Field>(state_.sum() - value, state_.correction()), value, comparison_epsilon_));
    }
    
    /**
     * @brief Get the current sum (the sum coordinate of our affine point).
     * @return The sum component.
     */
    Field sum() const { return state_.sum(); }
    
    /**
     * @brief Get the correction term (the correction coordinate).
     * @return The correction component.
     */
    Field correction() const { return state_.correction(); }
    
    /**
     * @brief Get the algebraic invariant (true sum).
     * @return The algebraic invariant (sum + correction).
     */
    Field algebraic_sum() const { return state_.algebraic_invariant(); }
    
    /**
     * @brief Get the current state as an affine point.
     * @return The current AffinePoint.
     */
    AffinePoint<Field> current_state() const { return state_; }
    
    /**
     * @brief Verify that our geometric representation maintains the algebraic invariant.
     * @return True if the invariant holds.
     */
    bool verify_invariant() const {
        return state_.satisfies_invariant(exact_sum_, comparison_epsilon_);
    }
};

/**
 * @brief Cohomology-inspired error analysis.
 * Measures how much floating-point arithmetic deviates from field axioms.
 * This class is specifically relevant for floating-point numbers.
 */
template<typename Field>
class FloatingPointCohomology {
public:
    /**
     * @brief Compute the "differential" of floating-point addition.
     * This measures non-associativity: (a + b) + c vs a + (b + c).
     * @return The associativity defect.
     */
    static Field associativity_defect(Field a, Field b, Field c) {
        Field left_assoc = (a + b) + c;
        Field right_assoc = a + (b + c);
        return left_assoc - right_assoc;
    }
    
    /**
     * @brief Measure the floating-point error in a single addition.
     * @return The addition error.
     */
    static Field addition_error(Field a, Field b) {
        Field computed = a + b;
        // In exact arithmetic, this would be zero
        // In floating-point, it captures the rounding error
        return computed - a - b;
    }
};

// --- Demonstration Functions ---

/**
 * @brief Demonstrates the Algebraic Kahan Summation framework with `double`.
 * This shows the traditional Kahan summation's error compensation.
 */
void demonstrate_with_double() {
    std::cout << std::fixed << std::setprecision(16);
    
    std::cout << "=== Algebraic Geometry Framework for Kahan Summation (Field = double) ===\n\n";
    
    // Create our geometric summation engine for doubles
    AlgebraicKahanSummator<double> kahan_double(1e-15); // Use a standard epsilon for double
    
    // Test case: summing numbers of very different magnitudes
    std::vector<double> values = {1.0, 1e-15, 1e-15, 1e-15, 1e-15, 1e-15};
    
    std::cout << "Adding values with vastly different magnitudes:\n";
    for (double val : values) {
        std::cout << "Adding: " << val << std::endl;
        kahan_double.add(val);
        
        std::cout << "  Current affine point: " << kahan_double.current_state() << std::endl;
        std::cout << "  Algebraic invariant (s+c): " << kahan_double.algebraic_sum() << std::endl;
        std::cout << "  Invariant verified: " << (kahan_double.verify_invariant() ? "✓" : "✗") << std::endl;
        std::cout << std::endl;
    }
    
    // Compare with naive summation
    double naive_sum = 0.0;
    for (double val : values) {
        naive_sum += val;
    }
    
    double expected_exact_sum = 1.0 + 5.0 * 1e-15;
    
    std::cout << "=== Results Comparison (double) ===\n";
    std::cout << "Naive sum:             " << naive_sum << std::endl;
    std::cout << "Kahan sum (s):         " << kahan_double.sum() << std::endl;
    std::cout << "Kahan correction (c):  " << kahan_double.correction() << std::endl;
    std::cout << "Algebraic invariant (s+c): " << kahan_double.algebraic_sum() << std::endl;
    std::cout << "Expected exact sum:    " << expected_exact_sum << std::endl;
    
    std::cout << "Difference (Kahan vs Exact): " << std::abs(kahan_double.algebraic_sum() - expected_exact_sum) << std::endl;
    std::cout << "Difference (Naive vs Exact): " << std::abs(naive_sum - expected_exact_sum) << std::endl;
    
    // Demonstrate cohomological error analysis for doubles
    std::cout << "\n=== Cohomological Error Analysis (double) ===\n";
    double a = 0.1, b = 0.2, c = 0.3; // Example where floating point errors might occur
    double defect = FloatingPointCohomology<double>::associativity_defect(a, b, c);
    std::cout << "Associativity defect for (0.1 + 0.2) + 0.3 vs 0.1 + (0.2 + 0.3): " << defect << std::endl;
    
    double error_add = FloatingPointCohomology<double>::addition_error(0.1, 0.2);
    std::cout << "Addition error for 0.1 + 0.2: " << error_add << std::endl;
}

/**
 * @brief Demonstrates the Algebraic Kahan Summation framework with `GF_p<5>`.
 * This highlights that Kahan's error compensation is not needed in exact fields.
 */
void demonstrate_with_gf_p() {
    std::cout << "\n\n=== Algebraic Geometry Framework (Field = GF_p<5>) ===\n\n";
    
    // Create our geometric summation engine for GF_p<5>
    // Epsilon is irrelevant for exact fields, but we pass 0.0 for consistency
    AlgebraicKahanSummator<GF_p<5>> kahan_gf(0.0); 
    
    // Test case: summing values in GF(5)
    // Values are integers, but they will be mapped to GF(5) elements
    std::vector<long long> raw_values = {1, 2, 3, 4, 1, 2};
    
    std::cout << "Adding values in GF(5):\n";
    GF_p<5> expected_exact_sum_gf(0);
    for (long long raw_val : raw_values) {
        GF_p<5> val(raw_val);
        std::cout << "Adding: " << val << std::endl;
        kahan_gf.add(val);
        
        expected_exact_sum_gf += val; // Track exact sum in GF(5)
        
        std::cout << "  Current affine point: " << kahan_gf.current_state() << std::endl;
        std::cout << "  Algebraic invariant (s+c): " << kahan_gf.algebraic_sum() << std::endl;
        std::cout << "  Invariant verified: " << (kahan_gf.verify_invariant() ? "✓" : "✗") << std::endl;
        std::cout << "  Expected exact sum (GF_p): " << expected_exact_sum_gf << std::endl;
        std::cout << std::endl;
    }
    
    // Compare with naive summation in GF(5)
    GF_p<5> naive_sum_gf(0);
    for (long long raw_val : raw_values) {
        naive_sum_gf += GF_p<5>(raw_val);
    }
    
    std::cout << "=== Results Comparison (GF_p<5>) ===\n";
    std::cout << "Naive sum (GF_p):      " << naive_sum_gf << std::endl;
    std::cout << "Kahan sum (s) (GF_p):  " << kahan_gf.sum() << std::endl;
    std::cout << "Kahan correction (c) (GF_p): " << kahan_gf.correction() << " (always 0 in exact fields)\n";
    std::cout << "Algebraic invariant (s+c) (GF_p): " << kahan_gf.algebraic_sum() << std::endl;
    std::cout << "Expected exact sum (GF_p): " << expected_exact_sum_gf << std::endl;
    
    std::cout << "Difference (Kahan vs Exact): " << (kahan_gf.algebraic_sum() - expected_exact_sum_gf) << " (should be 0)\n";
    std::cout << "Difference (Naive vs Exact): " << (naive_sum_gf - expected_exact_sum_gf) << " (should be 0)\n";
    
    // Cohomological error analysis is not applicable to exact fields, so we don't demonstrate it here.
}


int main() {
    demonstrate_with_double();
    demonstrate_with_gf_p();
    
    return 0;
}
