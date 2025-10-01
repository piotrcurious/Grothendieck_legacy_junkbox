/*
  esp32_smith_waterman_polyfit.ino

  Polynomial curve fitting (arbitrary degree) over normalized float dataset
  using a Smith-Waterman inspired local-alignment workflow.

  Steps:
    1. Global least-squares polynomial fit (Vandermonde + QR).
    2. Predict y over all x.
    3. Quantize predicted and observed y to bins.
    4. Smith-Waterman local alignment of predicted/observed sequences.
    5. Extract aligned indices, treat as inliers, re-fit polynomial.
    6. Iterate until convergence or max iterations.

  Assumes x/y normalized (e.g. in [0,1]). Single-file Arduino C++.
  Smith-Waterman builds (N+1)x(N+1) score matrix; keep N reasonable.

  Public function:
    bool smithWatermanPolyFit(const float *x, const float *y, int nPoints,
                             int degree, int maxIters, float tol,
                             float *outCoeffs, bool *inlierMask)

    - outCoeffs: array of length (degree+1)
    - inlierMask: array of length nPoints (true for final fit inliers)
    - returns true on success

  Example: Prints coefficients in `setup()`.
*/

#include <Arduino.h>

// -------------------------- Configuration ----------------------------
#define MAX_POINTS 200      // Change as needed. Smith-Waterman uses (N+1)^2 floats.
#define MAX_DEGREE 10
#define QUANT_BINS 8        // Number of quantization bins

// -------------------------- Utility Functions -------------------------
static inline float absf(float v) { return v < 0 ? -v : v; }

// Build Vandermonde matrix A (row-major) for x (n rows, degree+1 cols)
void buildVandermonde(const float *x, int n, int degree, float *A) {
  int m = degree + 1;
  for (int i = 0; i < n; ++i) {
    float xv = 1.0f;
    for (int j = 0; j < m; ++j) {
      A[i * m + j] = xv;
      xv *= x[i];
    }
  }
}

// Modified Gram-Schmidt: A (n x m) -> Q (n x m), R (m x m), both row-major
void modifiedGramSchmidt(const float *A, int n, int m, float *Q, float *R) {
  // Copy A into Q
  for (int i = 0; i < n * m; ++i) Q[i] = A[i];
  for (int k = 0; k < m; ++k) {
    // Compute R[k, k] = ||col_k||
    float norm = 0.0f;
    for (int i = 0; i < n; ++i) {
      float val = Q[i * m + k];
      norm += val * val;
    }
    norm = sqrt(norm);
    R[k * m + k] = norm;
    if (norm > 1e-12f) {
      for (int i = 0; i < n; ++i) Q[i * m + k] /= norm;
    }
    for (int j = k + 1; j < m; ++j) {
      // R[k, j] = Q[:, k] dot Q[:, j]
      float dot = 0.0f;
      for (int i = 0; i < n; ++i) dot += Q[i * m + k] * Q[i * m + j];
      R[k * m + j] = dot;
      for (int i = 0; i < n; ++i) Q[i * m + j] -= dot * Q[i * m + k];
    }
  }
}

// Compute Qt * y (Q is n x m row-major), y is length n, out is length m
void computeQtY(const float *Q, const float *y, int n, int m, float *out) {
  for (int j = 0; j < m; ++j) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += Q[i * m + j] * y[i];
    out[j] = s;
  }
}

// Back substitution for upper triangular R (m x m) * x = b
void backSubUpper(const float *R, const float *b, int m, float *x) {
  for (int i = m - 1; i >= 0; --i) {
    float s = b[i];
    for (int j = i + 1; j < m; ++j) s -= R[i * m + j] * x[j];
    x[i] = (fabs(R[i * m + i]) > 1e-12f) ? (s / R[i * m + i]) : 0.0f;
  }
}

// Least squares polynomial fit via QR (modified Gram-Schmidt)
bool fitPolyLS_QR(const float *x, const float *y, int n, int degree, float *coeffs) {
  int m = degree + 1;
  float *A = (float *)malloc(sizeof(float) * n * m);
  float *Q = (float *)malloc(sizeof(float) * n * m);
  float *R = (float *)malloc(sizeof(float) * m * m);
  float *b = (float *)malloc(sizeof(float) * m);
  if (!A || !Q || !R || !b) {
    free(A); free(Q); free(R); free(b);
    return false;
  }
  buildVandermonde(x, n, degree, A);
  modifiedGramSchmidt(A, n, m, Q, R);
  computeQtY(Q, y, n, m, b);
  backSubUpper(R, b, m, coeffs);
  free(A); free(Q); free(R); free(b);
  return true;
}

// Predict y given x and coeffs
void predictY(const float *x, int n, int degree, const float *coeffs, float *outY) {
  int m = degree + 1;
  for (int i = 0; i < n; ++i) {
    float xv = 1.0f;
    float s = 0.0f;
    for (int j = 0; j < m; ++j) {
      s += coeffs[j] * xv;
      xv *= x[i];
    }
    outY[i] = s;
  }
}

// Quantize an array of y (normalized within [minY, maxY]) into bins [0..bins-1]
void quantizeArray(const float *y, int n, int bins, float minY, float maxY, int *outBins) {
  float span = maxY - minY;
  for (int i = 0; i < n; ++i) {
    int bin = (int)((y[i] - minY) / span * bins);
    if (bin < 0) bin = 0;
    if (bin >= bins) bin = bins - 1;
    outBins[i] = bin;
  }
}

// Smith-Waterman local alignment for two integer sequences (aLen x bLen)
// Returns length of aligned pairs and fills alignedA/alignedB with indices.
int smithWaterman(const int *a, int aLen, const int *b, int bLen,
                  int matchScore, int mismatchPenalty, int gapPenalty,
                  int *alignedA, int *alignedB) {
  int rows = aLen + 1;
  int cols = bLen + 1;
  int *H = (int *)malloc(sizeof(int) * rows * cols);
  int *P = (int *)malloc(sizeof(int) * rows * cols); // traceback: 0=none,1=diag,2=up,3=left
  if (!H || !P) {
    free(H); free(P);
    return 0;
  }
  for (int i = 0; i < rows * cols; ++i) { H[i] = 0; P[i] = 0; }

  // Fill matrix
  int maxScore = 0, maxI = 0, maxJ = 0;
  for (int i = 1; i < rows; ++i) {
    for (int j = 1; j < cols; ++j) {
      int scoreDiag = H[(i - 1) * cols + (j - 1)] + (a[i - 1] == b[j - 1] ? matchScore : -mismatchPenalty);
      int scoreUp = H[(i - 1) * cols + j] - gapPenalty;
      int scoreLeft = H[i * cols + (j - 1)] - gapPenalty;
      int score = max(0, max(scoreDiag, max(scoreUp, scoreLeft)));
      H[i * cols + j] = score;
      if (score == scoreDiag && score > 0) P[i * cols + j] = 1;
      else if (score == scoreUp && score > 0) P[i * cols + j] = 2;
      else if (score == scoreLeft && score > 0) P[i * cols + j] = 3;
      if (score > maxScore) {
        maxScore = score;
        maxI = i;
        maxJ = j;
      }
    }
  }
  // Traceback
  int len = 0;
  int i = maxI, j = maxJ;
  while (i > 0 && j > 0 && H[i * cols + j] > 0) {
    if (P[i * cols + j] == 1) { // diag
      alignedA[len] = i - 1;
      alignedB[len] = j - 1;
      ++len;
      --i; --j;
    } else if (P[i * cols + j] == 2) {
      --i;
    } else if (P[i * cols + j] == 3) {
      --j;
    } else {
      break;
    }
  }
  free(H); free(P);
  return len;
}

// Main API: smithWatermanPolyFit
bool smithWatermanPolyFit(const float *x, const float *y, int nPoints, int degree,
                          int maxIters, float tol, float *outCoeffs, bool *inlierMask) {
  int m = degree + 1;
  float *coeffs = (float *)malloc(sizeof(float) * m);
  float *coeffs_prev = (float *)malloc(sizeof(float) * m);
  float *predY = (float *)malloc(sizeof(float) * nPoints);
  int *predBins = (int *)malloc(sizeof(int) * nPoints);
  int *obsBins = (int *)malloc(sizeof(int) * nPoints);
  int *alignedPred = (int *)malloc(sizeof(int) * nPoints);
  int *alignedObs = (int *)malloc(sizeof(int) * nPoints);
  if (!coeffs || !coeffs_prev || !predY || !predBins || !obsBins || !alignedPred || !alignedObs) {
    free(coeffs); free(coeffs_prev); free(predY);
    free(predBins); free(obsBins); free(alignedPred); free(alignedObs);
    return false;
  }

  // Initial fit over all points
  if (!fitPolyLS_QR(x, y, nPoints, degree, coeffs)) {
    free(coeffs); free(coeffs_prev); free(predY);
    free(predBins); free(obsBins); free(alignedPred); free(alignedObs);
    return false;
  }

  // Iterative refinement
  for (int iter = 0; iter < maxIters; ++iter) {
    for (int j = 0; j < m; ++j) coeffs_prev[j] = coeffs[j];

    // Predict
    predictY(x, nPoints, degree, coeffs, predY);

    // Quantize both predicted and observed
    float minY = 1e30, maxY = -1e30;
    for (int i = 0; i < nPoints; ++i) {
      if (y[i] < minY) minY = y[i];
      if (y[i] > maxY) maxY = y[i];
      if (predY[i] < minY) minY = predY[i];
      if (predY[i] > maxY) maxY = predY[i];
    }
    quantizeArray(predY, nPoints, QUANT_BINS, minY, maxY, predBins);
    quantizeArray(y, nPoints, QUANT_BINS, minY, maxY, obsBins);

    // Align
    int alignLen = smithWaterman(predBins, nPoints, obsBins, nPoints,
                                2, 1, 2, alignedPred, alignedObs);
    // Mark inliers
    for (int i = 0; i < nPoints; ++i) inlierMask[i] = false;
    for (int k = 0; k < alignLen; ++k) inlierMask[alignedObs[k]] = true;

    // Refit using only inliers
    int nInliers = 0;
    for (int i = 0; i < nPoints; ++i) if (inlierMask[i]) ++nInliers;
    float *xInliers = (float *)malloc(sizeof(float) * nInliers);
    float *yInliers = (float *)malloc(sizeof(float) * nInliers);
    int idx = 0;
    for (int i = 0; i < nPoints; ++i) {
      if (inlierMask[i]) {
        xInliers[idx] = x[i];
        yInliers[idx] = y[i];
        ++idx;
      }
    }
    if (nInliers >= m) {
      fitPolyLS_QR(xInliers, yInliers, nInliers, degree, coeffs);
    }
    free(xInliers); free(yInliers);

    // Check convergence
    float delta = 0.0f;
    for (int j = 0; j < m; ++j) delta += (coeffs[j] - coeffs_prev[j]) * (coeffs[j] - coeffs_prev[j]);
    if (sqrt(delta) < tol) break;
  }

  // Output
  for (int j = 0; j < m; ++j) outCoeffs[j] = coeffs[j];

  free(coeffs); free(coeffs_prev); free(predY);
  free(predBins); free(obsBins); free(alignedPred); free(alignedObs);
  return true;
}

// -------------------------- Example Usage ---------------------------
void printCoeffs(const float *c, int m) {
  for (int i = 0; i < m; ++i) {
    Serial.print("c["); Serial.print(i); Serial.print("]=");
    Serial.print(c[i], 6);
    if (i < m - 1) Serial.print(", ");
  }
  Serial.println();
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("Smith-Waterman inspired polynomial fit example");

  // Example dataset: cubic with noise and outlier region
  const int N = 60;
  float x[N], y[N];
  for (int i = 0; i < N; ++i) x[i] = (float)i / (N - 1);
  for (int i = 0; i < N; ++i) {
    // Cubic: y = 0.2 + 0.5x + 0.3x^2 - 0.2x^3 + noise
    y[i] = 0.2f + 0.5f * x[i] + 0.3f * x[i] * x[i] - 0.2f * x[i] * x[i] * x[i]
           + 0.05f * (random(-100, 100) / 100.0f);
    if (i >= 40 && i < 50) y[i] += 0.5f; // Outlier region
  }

  int degree = 3;
  float coeffs[MAX_DEGREE + 1];
  bool mask[N];
  bool ok = smithWatermanPolyFit(x, y, N, degree, 10, 1e-4f, coeffs, mask);
  if (!ok) {
    Serial.println("Fitting failed");
    return;
  }
  Serial.println("Fitted coefficients:");
  printCoeffs(coeffs, degree + 1);

  Serial.print("Inlier mask: ");
  for (int i = 0; i < N; ++i) {
    Serial.print(mask[i] ? "1" : "0");
    if (i < N - 1) Serial.print(",");
  }
  Serial.println();
}

void loop() {
  // nothing
}