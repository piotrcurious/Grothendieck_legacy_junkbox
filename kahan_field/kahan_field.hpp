#ifndef KAHAN_FIELD_HPP
#define KAHAN_FIELD_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <numeric>
#include <stdexcept>

/**
 * @file kahan_field.hpp
 * @brief An algebraic framework for Kahan summation and finite field arithmetic.
 *
 * This library treats floating-point errors not just as noise, but as geometric
 * objects that can be tracked using rational maps on affine spaces.
 */

// --- Field Concept and Implementations ---

/**
 * @brief Helper for robust comparison of field elements.
 */
template<typename Field>
struct FieldComparator {
    static bool are_equal(const Field& a, const Field& b, double epsilon) {
        return a == b;
    }
};

template<>
struct FieldComparator<double> {
    static bool are_equal(double a, double b, double epsilon) {
        return std::abs(a - b) < epsilon;
    }
};

template<>
struct FieldComparator<float> {
    static bool are_equal(float a, float b, double epsilon) {
        return std::abs(a - b) < static_cast<float>(epsilon);
    }
};

/**
 * @brief Represents a Finite Field element GF(P).
 * P should be a prime number for this to be a field.
 */
template<int P>
class GF_p {
private:
    long long val_;

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

    long long modInverse(long long n) const {
        if (n == 0) {
            throw std::runtime_error("Division by zero in GF_p: attempting to invert 0.");
        }
        return power(n, P - 2);
    }

public:
    GF_p(long long val = 0) : val_((val % P + P) % P) {}

    GF_p operator+(const GF_p& other) const { return GF_p((val_ + other.val_) % P); }
    GF_p operator-(const GF_p& other) const { return GF_p((val_ - other.val_ + P) % P); }
    GF_p operator*(const GF_p& other) const { return GF_p((val_ * other.val_) % P); }
    GF_p operator/(const GF_p& other) const {
        return GF_p((val_ * modInverse(other.val_)) % P);
    }

    GF_p& operator+=(const GF_p& other) { *this = *this + other; return *this; }
    GF_p& operator-=(const GF_p& other) { *this = *this - other; return *this; }
    GF_p& operator*=(const GF_p& other) { *this = *this * other; return *this; }
    GF_p& operator/=(const GF_p& other) { *this = *this / other; return *this; }

    bool operator==(const GF_p& other) const { return val_ == other.val_; }
    bool operator!=(const GF_p& other) const { return val_ != other.val_; }

    long long value() const { return val_; }

    friend std::ostream& operator<<(std::ostream& os, const GF_p& f) {
        return os << f.val_;
    }
};

/**
 * @brief Represents a 2D Tensor (Matrix) over a generic Field.
 * Supports both element-wise operations and matrix multiplication.
 */
template<typename Field>
class Tensor {
private:
    std::vector<std::vector<Field>> data_;
    size_t rows_;
    size_t cols_;

public:
    Tensor() : rows_(0), cols_(0) {}

    Tensor(size_t rows, size_t cols, Field initial = Field(0)) : rows_(rows), cols_(cols) {
        data_.assign(rows_, std::vector<Field>(cols_, initial));
    }

    Tensor(const std::vector<std::vector<Field>>& data) {
        if (data.empty()) {
            rows_ = 0;
            cols_ = 0;
        } else {
            rows_ = data.size();
            cols_ = data[0].size();
            for (const auto& row : data) {
                if (row.size() != cols_) {
                    throw std::invalid_argument("Inconsistent column counts in Tensor initialization.");
                }
            }
            data_ = data;
        }
    }

    Field& operator()(size_t r, size_t c) {
        return data_.at(r).at(c);
    }

    const Field& operator()(size_t r, size_t c) const {
        return data_.at(r).at(c);
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    bool has_same_dimensions(const Tensor& other) const {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    // Element-wise addition
    Tensor operator+(const Tensor& other) const {
        if (!has_same_dimensions(other)) throw std::invalid_argument("Dimension mismatch for addition.");
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result.data_[i][j] = data_[i][j] + other.data_[i][j];
        return result;
    }

    // Element-wise subtraction
    Tensor operator-(const Tensor& other) const {
        if (!has_same_dimensions(other)) throw std::invalid_argument("Dimension mismatch for subtraction.");
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result.data_[i][j] = data_[i][j] - other.data_[i][j];
        return result;
    }

    // Element-wise multiplication (Hadamard product)
    Tensor multiply_elementwise(const Tensor& other) const {
        if (!has_same_dimensions(other)) throw std::invalid_argument("Dimension mismatch for element-wise multiplication.");
        Tensor result(rows_, cols_);
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                result.data_[i][j] = data_[i][j] * other.data_[i][j];
        return result;
    }

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const {
        if (cols_ != other.rows_) throw std::invalid_argument("Dimension mismatch for matrix multiplication.");
        Tensor result(rows_, other.cols_);
        for (size_t i = 0; i < rows_; ++i) {
            for (size_t j = 0; j < other.cols_; ++j) {
                Field sum = Field(0);
                for (size_t k = 0; k < cols_; ++k) {
                    sum += data_[i][k] * other.data_[k][j];
                }
                result.data_[i][j] = sum;
            }
        }
        return result;
    }

    // Overload * for element-wise multiplication by default to match Kahan pattern
    Tensor operator*(const Tensor& other) const {
        return multiply_elementwise(other);
    }

    Tensor& operator+=(const Tensor& other) { *this = *this + other; return *this; }
    Tensor& operator-=(const Tensor& other) { *this = *this - other; return *this; }
    Tensor& operator*=(const Tensor& other) { *this = multiply_elementwise(other); return *this; }

    bool operator==(const Tensor& other) const {
        if (!has_same_dimensions(other)) return false;
        for (size_t i = 0; i < rows_; ++i)
            for (size_t j = 0; j < cols_; ++j)
                if (!(data_[i][j] == other.data_[i][j])) return false;
        return true;
    }

    bool operator!=(const Tensor& other) const { return !(*this == other); }

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

template<typename Field>
struct FieldComparator<Tensor<Field>> {
    static bool are_equal(const Tensor<Field>& a, const Tensor<Field>& b, double epsilon) {
        if (!a.has_same_dimensions(b)) return false;
        for (size_t i = 0; i < a.rows(); ++i)
            for (size_t j = 0; j < a.cols(); ++j)
                if (!FieldComparator<Field>::are_equal(a(i, j), b(i, j), epsilon)) return false;
        return true;
    }
};

// --- Algebraic Geometry Framework Components ---

/**
 * @brief Represents a point in affine space A^2 over a Field.
 * Coordinates are (sum, correction).
 * Algebraic invariant: true_sum = sum + correction.
 */
template<typename Field>
class AffinePoint {
private:
    Field sum_;
    Field correction_;

public:
    AffinePoint(Field sum = Field(), Field correction = Field())
        : sum_(sum), correction_(correction) {}

    Field sum() const { return sum_; }
    Field correction() const { return correction_; }

    void set_sum(const Field& s) { sum_ = s; }
    void set_correction(const Field& c) { correction_ = c; }

    Field algebraic_invariant() const {
        return sum_ + correction_;
    }

    bool satisfies_invariant(const Field& expected_sum, double epsilon = 1e-15) const {
        return FieldComparator<Field>::are_equal(algebraic_invariant(), expected_sum, epsilon);
    }

    friend std::ostream& operator<<(std::ostream& os, const AffinePoint& p) {
        os << "Sum: " << p.sum_ << "\nCorrection: " << p.correction_;
        return os;
    }
};

/**
 * @brief Represents the rational map phi: A^2 x Field -> A^2 that defines Kahan summation.
 */
template<typename Field>
class KahanRationalMap {
public:
    /**
     * @brief Apply the map: (s, c, x) |-> (s', c')
     */
    AffinePoint<Field> apply(const AffinePoint<Field>& point, const Field& x) const {
        Field s = point.sum();
        Field c = point.correction();

        Field y = x - c;
        Field t = s + y;
        Field c_new = (t - s) - y;
        Field s_new = t;

        return AffinePoint<Field>(s_new, c_new);
    }

    /**
     * @brief Verify that the map preserves the algebraic invariant (s + c + x = s' + c').
     */
    bool preserves_invariant(const AffinePoint<Field>& point, const Field& x, double epsilon = 1e-15) const {
        Field original_invariant = point.algebraic_invariant() + x;
        AffinePoint<Field> result = apply(point, x);
        Field new_invariant = result.algebraic_invariant();

        return FieldComparator<Field>::are_equal(original_invariant, new_invariant, epsilon);
    }
};

/**
 * @brief Algebraic Kahan Summation Engine.
 * Implements compensated summation using the rational map on affine space.
 */
template<typename Field>
class AlgebraicKahanSummator {
private:
    AffinePoint<Field> state_;
    KahanRationalMap<Field> rational_map_;

public:
    AlgebraicKahanSummator()
        : state_() {}

    /**
     * @brief Constructor for Tensors or types needing specific zero initialization.
     */
    AlgebraicKahanSummator(const Field& initial_zero)
        : state_(initial_zero, initial_zero) {}

    /**
     * @brief Add a value to the running sum using compensated summation.
     */
    void add(const Field& value) {
        state_ = rational_map_.apply(state_, value);
    }

    /**
     * @brief Get the running sum (leading term).
     */
    Field sum() const { return state_.sum(); }

    /**
     * @brief Get the current correction term.
     */
    Field correction() const { return state_.correction(); }

    /**
     * @brief Get the full algebraic sum (sum + correction).
     */
    Field algebraic_sum() const { return state_.algebraic_invariant(); }

    /**
     * @brief Get the current state in A^2.
     */
    AffinePoint<Field> current_state() const { return state_; }

    /**
     * @brief Verify the invariant preservation for a given step.
     */
    bool verify_step(const Field& x, double epsilon = 1e-15) const {
        return rational_map_.preserves_invariant(state_, x, epsilon);
    }
};

/**
 * @brief Cohomology-inspired error analysis.
 * Measures deviations from ideal field behavior.
 */
template<typename Field>
class FloatingPointCohomology {
public:
    /**
     * @brief d(a,b,c) = (a+b)+c - (a+(b+c))
     */
    static Field associativity_defect(Field a, Field b, Field c) {
        Field left_assoc = (a + b) + c;
        Field right_assoc = a + (b + c);
        return left_assoc - right_assoc;
    }

    /**
     * @brief Error of a single addition.
     */
    static Field addition_error(Field a, Field b) {
        Field computed = a + b;
        return computed - a - b;
    }
};

#endif // KAHAN_FIELD_HPP
