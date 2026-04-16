#ifndef FITTING_UTILS_H
#define FITTING_UTILS_H

#include <math.h>

// Common buffer size
const int FITTING_BUFFER_SIZE = 30;

// Linear model: y = mx + b
inline float linearModel(float x, float m, float b) {
    return m * x + b;
}

// Exponential model: y = a * e^(bx)
inline float exponentialModel(float x, float a, float b) {
    return a * exp(b * x);
}

// Function to calculate linear fit (y = mx + b)
inline void linearFit(float* x, float* y, int n, float& m, float& b) {
    float sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    for (int i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumX2 += x[i] * x[i];
    }
    float denominator = (n * sumX2 - sumX * sumX);
    if (fabs(denominator) < 1e-6) {
        m = 0;
        b = sumY / n;
    } else {
        m = (n * sumXY - sumX * sumY) / denominator;
        b = (sumY - m * sumX) / n;
    }
}

// Function to calculate exponential fit (y = a * e^(bx))
// Uses the Volterra integral equation approach: y(t) = y(0) + b * integral(y(s)ds) from 0 to t
inline void exponentialFitFredholm(float* x, float* y, int n, float& a, float& b) {
    float integral[FITTING_BUFFER_SIZE];
    integral[0] = 0;
    for (int i = 1; i < n && i < FITTING_BUFFER_SIZE; i++) {
        float dt = x[i] - x[i-1];
        integral[i] = integral[i-1] + (y[i] + y[i-1]) * 0.5 * dt;
    }

    // Linear fit: y = b * integral + y0
    float sumI = 0, sumY = 0, sumIY = 0, sumI2 = 0;
    for (int i = 0; i < n; i++) {
        sumI += integral[i];
        sumY += y[i];
        sumIY += integral[i] * y[i];
        sumI2 += integral[i] * integral[i];
    }

    float denominator = (n * sumI2 - sumI * sumI);
    if (fabs(denominator) < 1e-6) {
        b = 0;
        a = sumY / n;
    } else {
        b = (n * sumIY - sumI * sumY) / denominator;
        float y_start = (sumY - b * sumI) / n;
        // y(t) = a * exp(b*t). At t = x[0], y(x[0]) = a * exp(b*x[0]) = y_start
        // So a = y_start * exp(-b * x[0])
        a = y_start * exp(-b * x[0]);
    }
}

// Function to calculate the goodness of fit (R^2)
inline float goodnessOfFit(float* x, float* y, int n, float(*model)(float, float, float), float param1, float param2) {
    float ssTotal = 0, ssResidual = 0;
    float meanY = 0;
    for (int i = 0; i < n; i++) {
        meanY += y[i];
    }
    meanY /= n;
    for (int i = 0; i < n; i++) {
        float yi = y[i];
        float fi = model(x[i], param1, param2);
        ssTotal += (yi - meanY) * (yi - meanY);
        ssResidual += (yi - fi) * (yi - fi);
    }
    if (ssTotal < 1e-6) return 0;
    return 1 - (ssResidual / ssTotal);
}

// Function to calculate polynomial coefficients (least squares method)
// Max degree supported is 4 to keep buffer sizes fixed and small
inline void polynomialFit(float* x, float* y, int n, int degree, float* coeffs) {
    if (degree > 4) degree = 4;

    float X[2 * 4 + 1];
    int maxPowers = 2 * degree + 1;
    for (int i = 0; i < maxPowers; i++) X[i] = 0;

    float Y[5];
    for (int i = 0; i <= degree; i++) Y[i] = 0;

    for (int j = 0; j < n; j++) {
        float x_pow = 1.0;
        for (int i = 0; i < maxPowers; i++) {
            X[i] += x_pow;
            if (i <= degree) {
                Y[i] += x_pow * y[j];
            }
            x_pow *= x[j];
        }
    }

    float B[5][6], a[5];
    for (int i = 0; i <= degree; i++) {
        for (int j = 0; j <= degree; j++) {
            B[i][j] = X[i + j];
        }
    }

    for (int i = 0; i <= degree; i++) {
        B[i][degree + 1] = Y[i];
    }

    int systemSize = degree + 1;
    for (int i = 0; i < systemSize; i++) {
        for (int k = i + 1; k < systemSize; k++) {
            if (fabs(B[i][i]) < fabs(B[k][i])) {
                for (int j = 0; j <= systemSize; j++) {
                    float temp = B[i][j];
                    B[i][j] = B[k][j];
                    B[k][j] = temp;
                }
            }
        }
    }

    for (int i = 0; i < systemSize - 1; i++) {
        for (int k = i + 1; k < systemSize; k++) {
            if (fabs(B[i][i]) < 1e-9) continue;
            float t = B[k][i] / B[i][i];
            for (int j = 0; j <= systemSize; j++) {
                B[k][j] -= t * B[i][j];
            }
        }
    }

    for (int i = systemSize - 1; i >= 0; i--) {
        a[i] = B[i][systemSize];
        for (int j = i + 1; j < systemSize; j++) {
            a[i] -= B[i][j] * a[j];
        }
        if (fabs(B[i][i]) > 1e-9)
            a[i] /= B[i][i];
        else
            a[i] = 0;
    }

    for (int i = 0; i < systemSize; i++) {
        coeffs[i] = a[i];
    }
}

// Function to calculate the first derivative of a polynomial at a given point
inline float polynomialDerivative(float* coeffs, int degree, float x) {
    float derivative = 0;
    for (int i = 1; i <= degree; i++) {
        derivative += i * coeffs[i] * pow(x, i - 1);
    }
    return derivative;
}

#endif
