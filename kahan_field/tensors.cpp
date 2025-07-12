#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <numeric> // For std::gcd in modular inverse
#include <stdexcept> // For exceptions

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
        if (n == 0) {
            throw std::runtime_error("Division by zero in GF_p: attempting to invert 0.");
        }
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

// --- New: Tensor Class ---

/**
 * @brief Represents a 2D Tensor (Matrix) over a generic Field.
 * All operations are element-wise.
 */
template<typename Field>
class Tensor {
private:
    std::vector<std::vector<Field>> data_;
    size_t rows_;
    size_t cols_;

public:
    // Default constructor for empty tensor
    Tensor() : rows_(0), cols_(0) {}

    // Constructor for a tensor of specified dimensions, initialized to Field(0)
    Tensor(size_t rows, size_t cols) : rows_(rows), cols_(cols) {
        data_.resize(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            data_[i].resize(cols_, Field(0));
        }
    }

    // Constructor from a 2D vector of Field elements
    Tensor(const std::vector<std::vector<Field>>& data) {
        if (data.empty()) {
            rows_ = 0;
            cols_ = 0;
        } else {
            rows_ = data.size();
            cols_ = data[0].size();
            for (const auto& row : data) {
                if (row.size() != cols_) {
                    throw std::invalid_argument("All rows in Tensor data must have the same number of columns.");
                }
            }
            data_ = data;
        }
    }

    // Accessor for elements
    Field& operator()(size_t r, size_t c) {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Tensor element access out of bounds.");
        }
        return data_[r][c];
    }

    const Field& operator()(size_t r, size_t c) const {
        if (r >= rows_ || c >= cols_) {
            throw std::out_of_range("Tensor element access out of bounds.");
        }
        return data_[r][c];
    }

    // Get dimensions
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    // Check if dimensions match for element-wise operations
    bool has_same_dimensions(const Tensor& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    // Element-wise arithmetic operators
    Tensor operator+(const Tensor& other) const {
        if (!has_same_dimensions(other)) {
            throw std::invalid_argument("Tensor dimensions must match for addition.");
        }
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.data_[i][j] = data_[i][j] + other.data_[i][j];
            }
        }
        return result;
    }

    Tensor operator-(const Tensor& other) const {
        if (!has_same_dimensions(other)) {
            throw std::invalid_argument("Tensor dimensions must match for subtraction.");
        }
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.data_[i][j] = data_[i][j] - other.data_[i][j];
            }
        }
        return result;
    }

    Tensor operator*(const Tensor& other) const {
        if (!has_same_dimensions(other)) {
            throw std::invalid_argument("Tensor dimensions must match for element-wise multiplication.");
        }
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.data_[i][j] = data_[i][j] * other.data_[i][j];
            }
        }
        return result;
    }

    Tensor operator/(const Tensor& other) const {
        if (!has_same_dimensions(other)) {
            throw std::invalid_argument("Tensor dimensions must match for element-wise division.");
        }
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                result.data_[i][j] = data_[i][j] / other.data_[i][j];
            }
        }
        return result;
    }

    // Compound assignment operators
    Tensor& operator+=(const Tensor& other) { *this = *this + other; return *this; }
    Tensor& operator-=(const Tensor& other) { *this = *this - other; return *this; }
    Tensor& operator*=(const Tensor& other) { *this = *this * other; return *this; }
    Tensor& operator/=(const Tensor& other) { *this = *this / other; return *this; }

    // Comparison operator (element-wise equality)
    bool operator==(const Tensor& other) const {
        if (!has_same_dimensions(other)) {
            return false;
        }
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < cols_; ++j) {
                if (!(data_[i][j] == other.data_[i][j])) { // Uses Field's operator==
                    return false;
                }
            }
        }
        return true;
    }

    bool operator!=(const Tensor& other) const {
        return !(*this == other);
    }

    // Stream insertion operator
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "[";
        for (size_t i = 0; i < t.rows_; ++i) {
            if (i > 0) os << " ";
            os << "[";
            for (size_t j = 0; j < t.cols_; ++j) {
                os << t.data_[i][j];
                if (j < t.cols_ - 1) os << ", ";
            }
            os << "]";
            if (i < t.rows_ - 1) os << "\n";
        }
        os << "]";
        return os;
    }
};

// Specialization for FieldComparator for Tensor<double>
template<>
struct FieldComparator<Tensor<double>> {
    static bool are_equal(const Tensor<double>& a, const Tensor<double>& b, double epsilon) {
        if (!a.has_same_dimensions(b)) {
            return false;
        }
        for (size_t i = 0; i < a.rows(); ++i) {
            for (size_t j = 0; j < a.cols(); ++j) {
                if (!FieldComparator<double>::are_equal(a(i, j), b(i, j), epsilon)) {
                    return false;
                }
            }
        }
        return true;
    }
};


// --- Algebraic Geometry Framework Components (adapted for Tensor) ---

/**
 * @brief Represents a point in affine space A^2 over a Field (now potentially a Tensor).
 * In our Kahan summation context, coordinates are (sum_tensor, correction_tensor).
 */
template<typename Field> // Field can now be Tensor<double> or Tensor<GF_p<P>>
class AffinePoint {
private:
    Field sum_;
    Field correction_;
    
public:
    // Constructor with default values for sum and correction
    // Requires Field to have a default constructor that initializes to 'zero'
    AffinePoint(Field sum = Field(), Field correction = Field()) 
        : sum_(sum), correction_(correction) {}

    // Constructor for specific dimensions (useful when Field is a Tensor)
    AffinePoint(size_t rows, size_t cols)
        : sum_(rows, cols), correction_(rows, cols) {}
    
    // Accessors for sum and correction components
    Field sum() const { return sum_; }
    Field correction() const { return correction_; }
    
    // Setters for sum and correction components
    void set_sum(Field s) { sum_ = s; }
    void set_correction(Field c) { correction_ = c; }
    
    /**
     * @brief Calculates the algebraic invariant (the true sum represented by this point).
     * For Kahan summation, this is sum + correction (element-wise for Tensors).
     * @return The true sum (a Field, which could be a Tensor).
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
        os << "Sum: " << p.sum_ << "\nCorrection: " << p.correction_;
        return os;
    }
};

/**
 * @brief Represents the rational map φ: A^2 × R → A^2 that defines Kahan summation.
 * This is the core algebraic structure that preserves the summation invariant.
 * Field can now be a Tensor type.
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
        
        // Kahan summation steps, adapted for generic Field (now potentially Tensor):
        // All operations here are element-wise if Field is a Tensor.
        
        Field y = x - c;
        Field t = s + y;
        
        // This is the core of Kahan's compensation: (t - s) - y captures the rounding error.
        // In exact fields (or Tensors of exact fields), (t - s) - y will always be 0 (element-wise).
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
 * Now works with any Field, including Tensor<double> or Tensor<GF_p<P>>.
 */
template<typename Field>
class AlgebraicKahanSummator {
private:
    AffinePoint<Field> state_;
    KahanRationalMap<Field> rational_map_;
    Field exact_sum_;  // Track the true algebraic sum for verification
    double comparison_epsilon_; // Epsilon for comparisons, specific to this summator
    
public:
    // Constructor for scalar fields (Field has default constructor Field())
    AlgebraicKahanSummator(double epsilon = 1e-15) 
        : state_(), exact_sum_(), comparison_epsilon_(epsilon) {}

    // Constructor for tensor fields (Field is a Tensor, needs dimensions)
    AlgebraicKahanSummator(size_t rows, size_t cols, double epsilon = 1e-15)
        : state_(rows, cols), exact_sum_(rows, cols), comparison_epsilon_(epsilon) {}
    
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
 * This class is specifically relevant for floating-point numbers or Tensors of them.
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
 * @brief Demonstrates the Algebraic Kahan Summation framework with `Tensor<double>`.
 * This shows Kahan summation applied element-wise to matrices.
 */
void demonstrate_with_tensor_double() {
    std::cout << std::fixed << std::setprecision(16);
    
    std::cout << "=== Algebraic Geometry Framework for Kahan Summation (Field = Tensor<double>) ===\n\n";
    
    // Create our geometric summation engine for Tensor<double>
    size_t rows = 2, cols = 2;
    AlgebraicKahanSummator<Tensor<double>> kahan_tensor_double(rows, cols, 1e-15);
    
    // Test case: summing tensors with very different magnitudes
    std::vector<Tensor<double>> values = {
        Tensor<double>({{100.0, 1.0}, {0.1, 0.001}}),
        Tensor<double>({{1e-15, 2e-15}, {3e-15, 4e-15}}),
        Tensor<double>({{5e-15, 6e-15}, {7e-15, 8e-15}})
    };
    
    std::cout << "Adding tensors with vastly different magnitudes:\n";
    for (const auto& val_tensor : values) {
        std::cout << "Adding:\n" << val_tensor << std::endl;
        kahan_tensor_double.add(val_tensor);
        
        std::cout << "  Current affine point:\n" << kahan_tensor_double.current_state() << std::endl;
        std::cout << "  Algebraic invariant (s+c):\n" << kahan_tensor_double.algebraic_sum() << std::endl;
        std::cout << "  Invariant verified: " << (kahan_tensor_double.verify_invariant() ? "✓" : "✗") << std::endl;
        std::cout << std::endl;
    }
    
    // Compare with naive summation
    Tensor<double> naive_sum_tensor(rows, cols);
    for (const auto& val_tensor : values) {
        naive_sum_tensor += val_tensor;
    }
    
    // Calculate expected exact sum manually for verification
    Tensor<double> expected_exact_sum_tensor(rows, cols);
    expected_exact_sum_tensor(0,0) = 100.0 + 1e-15 + 5e-15; // 100.000000000000006
    expected_exact_sum_tensor(0,1) = 1.0 + 2e-15 + 6e-15;   // 1.000000000000008
    expected_exact_sum_tensor(1,0) = 0.1 + 3e-15 + 7e-15;   // 0.100000000000010
    expected_exact_sum_tensor(1,1) = 0.001 + 4e-15 + 8e-15; // 0.001000000000012
    
    std::cout << "=== Results Comparison (Tensor<double>) ===\n";
    std::cout << "Naive sum:\n" << naive_sum_tensor << std::endl;
    std::cout << "Kahan sum (s):\n" << kahan_tensor_double.sum() << std::endl;
    std::cout << "Kahan correction (c):\n" << kahan_tensor_double.correction() << std::endl;
    std::cout << "Algebraic invariant (s+c):\n" << kahan_tensor_double.algebraic_sum() << std::endl;
    std::cout << "Expected exact sum:\n" << expected_exact_sum_tensor << std::endl;
    
    std::cout << "Difference (Kahan vs Exact): " << (kahan_tensor_double.algebraic_sum() - expected_exact_sum_tensor) << std::endl;
    std::cout << "Difference (Naive vs Exact): " << (naive_sum_tensor - expected_exact_sum_tensor) << std::endl;
    
    // Demonstrate cohomological error analysis for Tensor<double>
    std::cout << "\n=== Cohomological Error Analysis (Tensor<double>) ===\n";
    Tensor<double> ta({{0.1, 0.2}, {0.3, 0.4}});
    Tensor<double> tb({{0.0000000000000001, 0.0000000000000002}, {0.0000000000000003, 0.0000000000000004}});
    Tensor<double> tc({{0.0000000000000005, 0.0000000000000006}, {0.0000000000000007, 0.0000000000000008}});
    
    Tensor<double> defect_tensor = FloatingPointCohomology<Tensor<double>>::associativity_defect(ta, tb, tc);
    std::cout << "Associativity defect tensor:\n" << defect_tensor << std::endl;
    
    Tensor<double> error_add_tensor = FloatingPointCohomology<Tensor<double>>::addition_error(ta, tb);
    std::cout << "Addition error tensor:\n" << error_add_tensor << std::endl;
}

/**
 * @brief Demonstrates the Algebraic Kahan Summation framework with `Tensor<GF_p<5>>`.
 * This highlights that Kahan's error compensation is not needed in exact fields, even for tensors.
 */
void demonstrate_with_tensor_gf_p() {
    std::cout << "\n\n=== Algebraic Geometry Framework (Field = Tensor<GF_p<5>>) ===\n\n";
    
    size_t rows = 2, cols = 2;
    AlgebraicKahanSummator<Tensor<GF_p<5>>> kahan_tensor_gf(rows, cols, 0.0);
    
    // Test case: summing tensors in GF(5)
    std::vector<Tensor<GF_p<5>>> values = {
        Tensor<GF_p<5>>({{GF_p<5>(1), GF_p<5>(2)}, {GF_p<5>(3), GF_p<5>(4)}}),
        Tensor<GF_p<5>>({{GF_p<5>(1), GF_p<5>(1)}, {GF_p<5>(1), GF_p<5>(1)}}),
        Tensor<GF_p<5>>({{GF_p<5>(2), GF_p<5>(3)}, {GF_p<5>(4), GF_p<5>(0)}})
    };
    
    std::cout << "Adding tensors in GF(5):\n";
    Tensor<GF_p<5>> expected_exact_sum_tensor_gf(rows, cols);
    for (const auto& val_tensor : values) {
        std::cout << "Adding:\n" << val_tensor << std::endl;
        kahan_tensor_gf.add(val_tensor);
        
        expected_exact_sum_tensor_gf += val_tensor; // Track exact sum in GF(5)
        
        std::cout << "  Current affine point:\n" << kahan_tensor_gf.current_state() << std::endl;
        std::cout << "  Algebraic invariant (s+c):\n" << kahan_tensor_gf.algebraic_sum() << std::endl;
        std::cout << "  Invariant verified: " << (kahan_tensor_gf.verify_invariant() ? "✓" : "✗") << std::endl;
        std::cout << "  Expected exact sum (GF_p):\n" << expected_exact_sum_tensor_gf << std::endl;
        std::cout << std::endl;
    }
    
    // Compare with naive summation in GF(5)
    Tensor<GF_p<5>> naive_sum_tensor_gf(rows, cols);
    for (const auto& val_tensor : values) {
        naive_sum_tensor_gf += val_tensor;
    }
    
    std::cout << "=== Results Comparison (Tensor<GF_p<5>>) ===\n";
    std::cout << "Naive sum (GF_p):\n" << naive_sum_tensor_gf << std::endl;
    std::cout << "Kahan sum (s) (GF_p):\n" << kahan_tensor_gf.sum() << std::endl;
    std::cout << "Kahan correction (c) (GF_p):\n" << kahan_tensor_gf.correction() << " (always 0-tensor in exact fields)\n";
    std::cout << "Algebraic invariant (s+c) (GF_p):\n" << kahan_tensor_gf.algebraic_sum() << std::endl;
    std::cout << "Expected exact sum (GF_p):\n" << expected_exact_sum_tensor_gf << std::endl;
    
    std::cout << "Difference (Kahan vs Exact):\n" << (kahan_tensor_gf.algebraic_sum() - expected_exact_sum_tensor_gf) << " (should be 0-tensor)\n";
    std::cout << "Difference (Naive vs Exact):\n" << (naive_sum_tensor_gf - expected_exact_sum_tensor_gf) << " (should be 0-tensor)\n";
    
    // Cohomological error analysis is not applicable to exact fields, so we don't demonstrate it here.
}


int main() {
    try {
        demonstrate_with_tensor_double();
        demonstrate_with_tensor_gf_p();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
