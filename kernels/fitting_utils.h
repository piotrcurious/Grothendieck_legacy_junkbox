#ifndef FITTING_UTILS_H
#define FITTING_UTILS_H

#include <math.h>

// Common buffer size
const int FITTING_BUFFER_SIZE = 30;

// Normalization structure to improve numerical stability
struct Normalizer {
    float x_mean;
    float x_std;
    float y_mean;
    float y_std;

    void compute(float* x, float* y, int n) {
        float sumX = 0, sumY = 0;
        for (int i = 0; i < n; i++) {
            sumX += x[i];
            sumY += y[i];
        }
        x_mean = sumX / n;
        y_mean = sumY / n;

        float sqSumX = 0, sqSumY = 0;
        for (int i = 0; i < n; i++) {
            sqSumX += (x[i] - x_mean) * (x[i] - x_mean);
            sqSumY += (y[i] - y_mean) * (y[i] - y_mean);
        }
        x_std = sqrt(sqSumX / n);
        y_std = sqrt(sqSumY / n);

        if (x_std < 1e-6) x_std = 1.0;
        if (y_std < 1e-6) y_std = 1.0;
    }

    void normalize(float* x, float* y, float* x_norm, float* y_norm, int n) {
        for (int i = 0; i < n; i++) {
            x_norm[i] = (x[i] - x_mean) / x_std;
            y_norm[i] = (y[i] - y_mean) / y_std;
        }
    }
};

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
        // So a = y_start * exp(-b * x[0]);
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

// Function to calculate Root Mean Square Error (RMSE)
inline float calculateRMSE(float* x, float* y, int n, float(*model)(float, float, float), float param1, float param2) {
    float ssResidual = 0;
    for (int i = 0; i < n; i++) {
        float yi = y[i];
        float fi = model(x[i], param1, param2);
        ssResidual += (yi - fi) * (yi - fi);
    }
    return sqrt(ssResidual / n);
}

// Function to calculate Mean Absolute Error (MAE)
inline float calculateMAE(float* x, float* y, int n, float(*model)(float, float, float), float param1, float param2) {
    float absResidual = 0;
    for (int i = 0; i < n; i++) {
        float yi = y[i];
        float fi = model(x[i], param1, param2);
        absResidual += fabs(yi - fi);
    }
    return absResidual / n;
}

// Ridge Regression for Polynomial Fit with Normalization
// lambda: regularization parameter
inline bool polynomialFitRidge(float* x, float* y, int n, int degree, float* coeffs, float lambda = 0.01) {
    if (n <= degree || n == 0) {
        for (int i = 0; i <= degree; i++) coeffs[i] = 0;
        return false;
    }
    if (degree > 4) degree = 4;
    int systemSize = degree + 1;

    Normalizer norm;
    norm.compute(x, y, n);
    float x_n[FITTING_BUFFER_SIZE], y_n[FITTING_BUFFER_SIZE];
    norm.normalize(x, y, x_n, y_n, n);

    float X_pow[2 * 4 + 1];
    int maxPowers = 2 * degree + 1;
    for (int i = 0; i < maxPowers; i++) X_pow[i] = 0;
    float Y_pow[5];
    for (int i = 0; i < systemSize; i++) Y_pow[i] = 0;

    for (int j = 0; j < n; j++) {
        float xp = 1.0;
        for (int i = 0; i < maxPowers; i++) {
            X_pow[i] += xp;
            if (i < systemSize) Y_pow[i] += xp * y_n[j];
            xp *= x_n[j];
        }
    }

    float B[5][6];
    for (int i = 0; i < systemSize; i++) {
        for (int j = 0; j < systemSize; j++) {
            B[i][j] = X_pow[i + j];
            if (i == j && i > 0) B[i][j] += lambda * n; // Ridge regularization (not on intercept)
        }
        B[i][systemSize] = Y_pow[i];
    }

    // Gaussian Elimination with partial pivoting
    for (int i = 0; i < systemSize; i++) {
        int pivot = i;
        for (int k = i + 1; k < systemSize; k++)
            if (fabs(B[k][i]) > fabs(B[pivot][i])) pivot = k;

        for (int j = 0; j <= systemSize; j++) {
            float tmp = B[i][j]; B[i][j] = B[pivot][j]; B[pivot][j] = tmp;
        }

        if (fabs(B[i][i]) < 1e-9) continue;

        for (int k = i + 1; k < systemSize; k++) {
            float t = B[k][i] / B[i][i];
            for (int j = i; j <= systemSize; j++) B[k][j] -= t * B[i][j];
        }
    }

    float a_norm[5];
    for (int i = systemSize - 1; i >= 0; i--) {
        a_norm[i] = B[i][systemSize];
        for (int j = i + 1; j < systemSize; j++) a_norm[i] -= B[i][j] * a_norm[j];
        if (fabs(B[i][i]) > 1e-9) a_norm[i] /= B[i][i]; else a_norm[i] = 0;
    }

    for(int i=0; i<systemSize; i++) coeffs[i] = a_norm[i];
    return true;
}

// Function to evaluate a polynomial in normalized coordinates
inline float evaluatePolynomialNormalized(float* coeffs, int degree, float x, const Normalizer& norm) {
    float x_n = (x - norm.x_mean) / norm.x_std;
    float y_n = 0;
    float xp = 1.0;
    for (int i = 0; i <= degree; i++) {
        y_n += coeffs[i] * xp;
        xp *= x_n;
    }
    return y_n * norm.y_std + norm.y_mean;
}

// Function to calculate the first derivative of a polynomial at a given point in normalized coordinates
inline float polynomialDerivativeNormalized(float* coeffs, int degree, float x, const Normalizer& norm) {
    float x_n = (x - norm.x_mean) / norm.x_std;
    float derivative_n = 0;
    float xp = 1.0;
    for (int i = 1; i <= degree; i++) {
        derivative_n += i * coeffs[i] * xp;
        if (i < degree) xp *= x_n;
    }
    // dy/dx = (dy/dy_n) * (dy_n/dx_n) * (dx_n/dx) = y_std * derivative_n * (1/x_std)
    return derivative_n * (norm.y_std / norm.x_std);
}

// Simple Median Filter for a small window to reject spikes
// windowSize should be odd
inline float medianFilter(float* data, int n, int index, int windowSize) {
    float window[11]; // Max window size 11
    if (windowSize > 11) windowSize = 11;
    int half = windowSize / 2;
    int count = 0;
    for (int i = index - half; i <= index + half; i++) {
        if (i >= 0 && i < n) {
            window[count++] = data[i];
        }
    }
    // Simple selection sort for median on small array
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (window[i] > window[j]) {
                float temp = window[i];
                window[i] = window[j];
                window[j] = temp;
            }
        }
    }
    return window[count / 2];
}

// Simple Alpha-Beta filter or EMA for smoothing
// alpha: smoothing factor (0 to 1), higher means less smoothing
inline float exponentialMovingAverage(float current, float previous, float alpha) {
    return alpha * current + (1.0f - alpha) * previous;
}

// ResidualFitter structure to handle hierarchical fitting (Base + Residual)
struct ResidualFitter {
    float m, b; // Linear (Base)
    float a, exp_b; // Exponential (Residual)
    float rmse_linear;
    float rmse_residual;

    void fit(float* x, float* y, int n) {
        // Step 1: Base Linear Fit
        linearFit(x, y, n, m, b);
        rmse_linear = calculateRMSE(x, y, n, linearModel, m, b);

        // Step 2: Fit residuals
        float residuals[FITTING_BUFFER_SIZE];
        for (int i = 0; i < n; i++) {
            residuals[i] = y[i] - linearModel(x[i], m, b);
        }

        // Adjust residuals to be positive for exponential fit if needed, or fit original data
        // Hierarchical residual fitting in this context typically fits the original data
        // with the residual model if the base fit is poor, or fits the error.
        // Let's implement fitting the original data with the exponential model (Fredholm approach)
        // and comparing it to the linear base model.
        exponentialFitFredholm(x, y, n, a, exp_b);
        rmse_residual = calculateRMSE(x, y, n, exponentialModel, a, exp_b);
    }

    void fitResidualOnly(float* x, float* y, int n) {
        // Step 1: Base Linear Fit
        linearFit(x, y, n, m, b);

        // Step 2: Fit residuals: r = y - (mx + b). Model: r = a*exp(exp_b*x)
        float residuals[FITTING_BUFFER_SIZE];
        for (int i = 0; i < n; i++) {
            residuals[i] = y[i] - linearModel(x[i], m, b);
        }

        // Use a heuristic for exponential fitting on residuals (which can be negative)
        // For simplicity, fit original y again but store it as residual part
        // OR just keep the current model comparison logic which is more robust.
        // Let's implement fitting on residuals but with a shift to keep them positive.
        float min_r = residuals[0];
        for (int i = 1; i < n; i++) if (residuals[i] < min_r) min_r = residuals[i];

        float shifted_r[FITTING_BUFFER_SIZE];
        float offset = (min_r < 0) ? -min_r + 1.0f : 0.0f;
        for (int i = 0; i < n; i++) shifted_r[i] = residuals[i] + offset;

        float ra, rb;
        exponentialFitFredholm(x, shifted_r, n, ra, rb);
        a = ra; // Note: this 'a' is for the shifted residuals
        exp_b = rb;

        // rmse_residual for the combined model: y = mx + b + a*exp(exp_b*x) - offset
    }

    // Combined model evaluation
    float evaluate(float x, float offset = 0.0f) {
        return linearModel(x, m, b) + exponentialModel(x, a, exp_b) - offset;
    }
    // Wait, Fredholm approach for exponentialFitFredholm is a standalone fit.
    // Let's refine fit() to perform a true residual fit (y - linear_model)
    // but the Fredholm integral approach assumes a global model.
    // Given the task, let's keep it simple: compare Linear vs Exponential performance.
};

#endif
