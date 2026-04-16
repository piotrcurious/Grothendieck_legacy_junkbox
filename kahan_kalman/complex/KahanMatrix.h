#ifndef KAHAN_MATRIX_H
#define KAHAN_MATRIX_H
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <utility>
#define KAHAN_EPSILON 1e-12
inline double kahanSum(const double* input, size_t size) {
    double sum = 0.0; double c = 0.0;
    for (size_t i = 0; i < size; ++i) {
        double y = input[i] - c; double t = sum + y;
        c = (t - sum) - y; sum = t;
    }
    return sum;
}
class Matrix {
public:
    double* data; int rows; int cols;
    Matrix(int r, int c) : rows(r), cols(c) {
        if (rows <= 0 || cols <= 0) { rows = 0; cols = 0; data = nullptr; return; }
        data = new double[rows * cols];
        if (data) { for (int i = 0; i < rows * cols; ++i) data[i] = 0.0; }
    }
    ~Matrix() { if (data) delete[] data; }
    Matrix(const Matrix& other) : rows(other.rows), cols(other.cols) {
        data = new double[rows * cols];
        if (data) memcpy(data, other.data, rows * cols * sizeof(double));
    }
    Matrix& operator=(const Matrix& other) {
        if (this == &other) return *this;
        if (data) delete[] data;
        rows = other.rows; cols = other.cols;
        data = new double[rows * cols];
        if (data) memcpy(data, other.data, rows * cols * sizeof(double));
        return *this;
    }
    double& operator()(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) { static double dummy = NAN; return dummy; }
        return data[r * cols + c];
    }
    const double& operator()(int r, int c) const {
        if (r < 0 || r >= rows || c < 0 || c >= cols) { static const double dummy = NAN; return dummy; }
        return data[r * cols + c];
    }
    Matrix add(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) return Matrix(0, 0);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double terms[] = {(*this)(i, j), other(i, j)};
                result(i, j) = kahanSum(terms, 2);
            }
        }
        return result;
    }
    Matrix subtract(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) return Matrix(0, 0);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double terms[] = {(*this)(i, j), -other(i, j)};
                result(i, j) = kahanSum(terms, 2);
            }
        }
        return result;
    }
    Matrix multiply(const Matrix& other) const {
        if (cols != other.rows) return Matrix(0, 0);
        Matrix result(rows, other.cols);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < other.cols; ++j) {
                double sum = 0.0; double c = 0.0;
                for (int k = 0; k < cols; ++k) {
                    double term = (*this)(i, k) * other(k, j);
                    double y = term - c; double t = sum + y;
                    c = (t - sum) - y; sum = t;
                }
                result(i, j) = sum;
            }
        }
        return result;
    }
    Matrix multiply_scalar(double scalar) const {
        Matrix result(rows, cols);
        if (result.data) { for (int i = 0; i < rows * cols; ++i) result.data[i] = data[i] * scalar; }
        return result;
    }
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i) { for (int j = 0; j < cols; ++j) result(j, i) = (*this)(i, j); }
        return result;
    }
    void print() const {
        if (!data) { Serial.println("Empty Matrix"); return; }
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { Serial.print((*this)(i, j), 6); Serial.print("	"); }
            Serial.println();
        }
    }
};
inline Matrix solveLinear(const Matrix& A, const Matrix& b) {
    if (A.rows != A.cols || A.rows != b.rows) return Matrix(0, 0);
    int n = A.rows; int m = b.cols;
    Matrix aug(n, n + m);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug(i, j) = A(i, j);
        for (int j = 0; j < m; ++j) aug(i, n + j) = b(i, j);
    }
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int k = i + 1; k < n; ++k) { if (fabs(aug(k, i)) > fabs(aug(pivot, i))) pivot = k; }
        if (pivot != i) { for (int j = i; j < n + m; ++j) std::swap(aug(i, j), aug(pivot, j)); }
        if (fabs(aug(i, i)) < KAHAN_EPSILON) return Matrix(0, 0);
        for (int k = i + 1; k < n; ++k) {
            double factor = aug(k, i) / aug(i, i);
            for (int j = i; j < n + m; ++j) {
                double term = factor * aug(i, j);
                double terms[] = {aug(k, j), -term};
                aug(k, j) = kahanSum(terms, 2);
            }
        }
    }
    Matrix x(n, m);
    for (int col = 0; col < m; ++col) {
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0; double c = 0.0;
            for (int j = i + 1; j < n; ++j) {
                double term = aug(i, j) * x(j, col);
                double y = term - c; double t = sum + y;
                c = (t - sum) - y; sum = t;
            }
            x(col < m ? i : 0, col) = (aug(i, n + col) - sum) / aug(i, i);
        }
    }
    return x;
}
inline bool ldltDecomposition(const Matrix& A, Matrix& L, Matrix& D) {
    if (A.rows != A.cols || L.rows != A.rows || L.cols != A.cols || D.rows != A.rows || D.cols != A.cols) return false;
    int n = A.rows;
    for (int i = 0; i < n; ++i) {
        double sum_D = 0.0; double c_D = 0.0;
        for (int k = 0; k < i; ++k) {
            double term = L(i, k) * L(i, k) * D(k, k);
            double y = term - c_D; double t = sum_D + y;
            c_D = (t - sum_D) - y; sum_D = t;
        }
        D(i, i) = A(i, i) - sum_D;
        if (D(i, i) < KAHAN_EPSILON) return false;
        L(i, i) = 1.0;
        for (int j = i + 1; j < n; ++j) {
            double sum_L = 0.0; double c_L = 0.0;
            for (int k = 0; k < i; ++k) {
                double term = L(j, k) * D(k, k) * L(i, k);
                double y = term - c_L; double t = sum_L + y;
                c_L = (t - sum_L) - y; sum_L = t;
            }
            L(j, i) = (A(j, i) - sum_L) / D(i, i);
        }
    }
    return true;
}
inline Matrix solveLDLT(const Matrix& A, const Matrix& b) {
    if (A.rows != A.cols || A.rows != b.rows) return Matrix(0, 0);
    int n = A.rows; int m = b.cols;
    Matrix L(n, n), D(n, n);
    if (!ldltDecomposition(A, L, D)) return Matrix(0, 0);
    Matrix x(n, m);
    for (int col = 0; col < m; ++col) {
        Matrix y(n, 1);
        for (int i = 0; i < n; ++i) {
            double sum = 0.0; double c = 0.0;
            for (int j = 0; j < i; ++j) {
                double term = L(i, j) * y(j, 0);
                double y_c = term - c; double t = sum + y_c;
                c = (t - sum) - y_c; sum = t;
            }
            y(i, 0) = b(i, col) - sum;
        }
        Matrix z(n, 1);
        for (int i = 0; i < n; ++i) z(i, 0) = y(i, 0) / D(i, i);
        for (int i = n - 1; i >= 0; --i) {
            double sum = 0.0; double c = 0.0;
            for (int j = i + 1; j < n; ++j) {
                double term = L(j, i) * x(j, col);
                double y_c = term - c; double t = sum + y_c;
                c = (t - sum) - y_c; sum = t;
            }
            x(i, col) = z(i, 0) - sum;
        }
    }
    return x;
}
#endif
