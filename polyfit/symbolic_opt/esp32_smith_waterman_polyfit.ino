/* esp32_smith_waterman_polyfit.ino

Polynomial curve fitting (arbitrary degree) over normalized float dataset using a Smith-Waterman inspired local-alignment workflow.

Approach (practical interpretation of the user's request):

1. Do a global least-squares polynomial fit (Vandermonde matrix + Modified Gram-Schmidt QR).


2. Use the fitted polynomial to predict y over all x.


3. Quantize predicted and observed y to discrete symbols (bins).


4. Run Smith-Waterman local alignment between predicted-symbol sequence and observed-symbol sequence. This finds the best local aligned region where predicted and observed behaviours match.


5. Extract the aligned index pairs, treat those as inliers and re-fit the polynomial using only them.


6. Iterate until convergence or max iterations.



Notes:

Input x is assumed normalized (e.g. in [0,1]) and y normalized likewise.

This code is single-file Arduino-style C++ and is suitable for ESP32 builds.

Memory: Smith-Waterman builds an (N+1)x(N+1) score matrix; limit N to a reasonable size.


Public function: bool smithWatermanPolyFit(const float *x, const float *y, int nPoints, int degree, int maxIters, float tol, float *outCoeffs, bool *inlierMask)

- outCoeffs should point to array of length (degree+1)
- inlierMask should point to array of length nPoints (will be true for points included in final fit)
- returns true on success

Example usage in setup() prints coefficients. */

#include <Arduino.h>

// -------------------------- Configuration ---------------------------- #define MAX_POINTS 200   // change as needed but watch memory for SW (matrix ~ (N+1)^2 floats) #define MAX_DEGREE 10 #define Q_BINS 128       // number of quantization bins for y-values used by Smith-Waterman

// -------------------------- Utility functions ----------------------- static inline float absf(float v) { return v < 0 ? -v : v; }

// Build Vandermonde matrix A (row-major) for x (n rows, m=degree+1 cols) // A[i*m + j] = x[i]^j void buildVandermonde(const float *x, int n, int degree, float A){ int m = degree + 1; for(int i=0;i<n;++i){ float xv = 1.0f; for(int j=0;j<m;++j){ A[im + j] = xv; xv *= x[i]; } } }

// Modified Gram-Schmidt: A (n x m) -> Q (n x m) and R (m x m) // A is row-major; Q stored row-major; R stored row-major void modifiedGramSchmidt(const float *A, int n, int m, float Q, float R){ // Copy A into Q initially for(int i=0;i<nm;++i) Q[i] = A[i]; // zero R for(int i=0;i<mm;++i) R[i]=0.0f;

for(int k=0;k<m;++k){ // compute R[k,k] = ||col_k|| float norm = 0.0f; for(int i=0;i<n;++i){ float val = Q[im + k]; norm += valval; } norm = sqrt(norm); R[km + k] = norm; if(norm > 1e-12f){ for(int i=0;i<n;++i) Q[im + k] /= norm; } // orthogonalize subsequent columns for(int j=k+1;j<m;++j){ float dot = 0.0f; for(int i=0;i<n;++i) dot += Q[im + k] * Q[im + j]; R[km + j] = dot; for(int i=0;i<n;++i) Q[im + j] -= dot * Q[i*m + k]; } } }

// Compute Qt * y  (Q is n x m row-major), y is length n, out is length m void computeQtY(const float *Q, const float *y, int n, int m, float out){ for(int j=0;j<m;++j){ float s=0.0f; for(int i=0;i<n;++i) s += Q[im + j] * y[i]; out[j] = s; } }

// Back substitution solving R (m x m, upper triangular) * x = b void backSubUpper(const float *R, const float b, int m, float x){ for(int i=m-1;i>=0;--i){ float s = b[i]; for(int j=i+1;j<m;++j) s -= R[im + j] * x[j]; float rii = R[im + i]; if(fabs(rii) < 1e-12f) x[i] = 0.0f; else x[i] = s / rii; } }

// Least squares polynomial fit via QR (modified Gram-Schmidt) // x (n), y (n), degree -> coeffs length degree+1 // returns true on success bool fitPolyLS_QR(const float *x, const float *y, int n, int degree, float *coeffs){ int m = degree + 1; if(n < m) return false; // allocate float A = (float)malloc(sizeof(float)nm); float Q = (float)malloc(sizeof(float)nm); float R = (float)malloc(sizeof(float)mm); float QtY = (float)malloc(sizeof(float)*m); if(!A || !Q || !R || !QtY){ free(A); free(Q); free(R); free(QtY); return false; } buildVandermonde(x,n,degree,A); modifiedGramSchmidt(A,n,m,Q,R); computeQtY(Q,y,n,m,QtY); backSubUpper(R,QtY,m,coeffs); free(A); free(Q); free(R); free(QtY); return true; }

// Predict y given x and coeffs void predictY(const float *x, int n, int degree, const float *coeffs, float *outY){ int m = degree + 1; for(int i=0;i<n;++i){ float xv = 1.0f; float s = 0.0f; for(int j=0;j<m;++j){ s += coeffs[j] * xv; xv *= x[i]; } outY[i] = s; } }

// Quantize an array of y (assumed normalized within [minY,maxY]) into bins [0..bins-1] void quantizeArray(const float *y, int n, int bins, float minY, float maxY, int *outBins){ float span = maxY - minY; if(span <= 0.0f) span = 1.0f; for(int i=0;i<n;++i){ float v = (y[i] - minY) / span; int b = (int)floor(v * (bins-1) + 0.5f); if(b < 0) b = 0; if(b >= bins) b = bins-1; outBins[i] = b; } }

// Smith-Waterman local alignment for two integer sequences (aLen x bLen) // scoring: match -> +matchScore, mismatch -> -mismatchPenalty, gap -> -gapPenalty // returns length of aligned pairs and fills aligned index pairs (predIdx[], obsIdx[]) in order from left->right // aligned arrays should be at least min(aLen,bLen) long int smithWatermanAlign(const int *a, int aLen, const int *b, int bLen, int matchScore, int mismatchPenalty, int gapPenalty, int *outPredIdx, int *outObsIdx){ // Build DP matrix H of size (aLen+1)x(bLen+1) // For moderate sizes, build full matrix on the heap int rows = aLen + 1; int cols = bLen + 1; // avoid huge allocations if((long)rows * cols > 20000){ // too large; fallback to simple local-correlation (Kadane on per-point score) // We'll instead do simple best contiguous subsequence on equal-length sequences int bestS = 0, curS = 0, bestStart=0, curStart=0, bestEnd=-1; int minLen = min(aLen,bLen); for(int i=0;i<minLen;++i){ int sc = (a[i]==b[i]) ? matchScore : -mismatchPenalty; curS += sc; if(curS <= 0){ curS = 0; curStart = i+1; } else if(curS > bestS){ bestS = curS; bestStart = curStart; bestEnd = i; } } if(bestEnd < 0) return 0; int L = bestEnd - bestStart + 1; for(int k=0;k<L;++k){ outPredIdx[k] = bestStart + k; outObsIdx[k] = bestStart + k; } return L; }

// allocate matrix int H = (int)malloc(sizeof(int)rowscols); int P = (int)malloc(sizeof(int)rowscols); // traceback: 0=none,1=diag,2=up,3=left if(!H || !P){ free(H); free(P); return 0; } for(int i=0;i<rowscols;++i){ H[i]=0; P[i]=0; } int maxI=0, maxJ=0, maxVal=0; for(int i=1;i<rows;++i){ for(int j=1;j<cols;++j){ int sc = (a[i-1]==b[j-1]) ? matchScore : -mismatchPenalty; int diag = H[(i-1)cols + (j-1)] + sc; int up   = H[(i-1)cols + j] - gapPenalty; int left = H[icols + (j-1)] - gapPenalty; int h = diag; int p = 1; if(up > h){ h = up; p = 2; } if(left > h){ h = left; p = 3; } if(h < 0) { h = 0; p = 0; } H[icols + j] = h; P[icols + j] = p; if(h > maxVal){ maxVal=h; maxI=i; maxJ=j; } } } if(maxVal == 0){ free(H); free(P); return 0; } // Traceback from maxI,maxJ int i = maxI, j = maxJ; int outLen = 0; while(i>0 && j>0){ int p = P[i*cols + j]; if(p==0) break; if(p==1){ // diag -> aligned pair outPredIdx[outLen] = i-1; outObsIdx[outLen]  = j-1; outLen++; i--; j--; } else if(p==2){ i--; } else if(p==3){ j--; } } // currently reversed order (right->left). reverse to left->right and return for(int k=0;k<outLen/2;++k){ int t1 = outPredIdx[k]; int t2 = outPredIdx[outLen-1-k]; outPredIdx[k]=t2; outPredIdx[outLen-1-k]=t1; } for(int k=0;k<outLen/2;++k){ int t1 = outObsIdx[k]; int t2 = outObsIdx[outLen-1-k]; outObsIdx[k]=t2; outObsIdx[outLen-1-k]=t1; } free(H); free(P); return outLen; }

// Main API: smithWatermanPolyFit // x,y: length nPoints (normalized floats) // degree: polynomial degree // maxIters: iterative refinements using SW // tol: stop if coefficient change L2 norm < tol // outCoeffs: length degree+1 // inlierMask: length nPoints (bool) bool smithWatermanPolyFit(const float *x, const float *y, int nPoints, int degree, int maxIters, float tol, float *outCoeffs, bool *inlierMask){ if(nPoints <= 0 || degree < 0 || degree > MAX_DEGREE || nPoints > MAX_POINTS) return false; int m = degree + 1; // initial fit on all points float coeffs = (float)malloc(sizeof(float)*m); float coeffs_prev = (float)malloc(sizeof(float)*m); float predY = (float)malloc(sizeof(float)*nPoints); int predBins = (int)malloc(sizeof(int)*nPoints); int obsBins  = (int)malloc(sizeof(int)*nPoints); int alignedPred = (int)malloc(sizeof(int)*nPoints); int alignedObs  = (int)malloc(sizeof(int)*nPoints); if(!coeffs||!coeffs_prev||!predY||!predBins||!obsBins||!alignedPred||!alignedObs){ free(coeffs); free(coeffs_prev); free(predY); free(predBins); free(obsBins); free(alignedPred); free(alignedObs); return false; } if(!fitPolyLS_QR(x,y,nPoints,degree,coeffs)){ free(coeffs); free(coeffs_prev); free(predY); free(predBins); free(obsBins); free(alignedPred); free(alignedObs); return false; } for(int i=0;i<nPoints;++i) inlierMask[i] = true;

for(int iter=0; iter<maxIters; ++iter){ // store previous for(int j=0;j<m;++j) coeffs_prev[j]=coeffs[j]; // predict predictY(x,nPoints,degree,coeffs,predY); // quantize both predicted and observed using their common min/max float minY = y[0], maxY = y[0]; for(int i=1;i<nPoints;++i){ if(y[i]<minY) minY=y[i]; if(y[i]>maxY) maxY=y[i]; } // expand min/max slightly to allow predicted outside float pad = (maxY - minY) * 0.05f + 1e-6f; minY -= pad; maxY += pad; quantizeArray(predY,nPoints,Q_BINS,minY,maxY,predBins); quantizeArray(y,nPoints,Q_BINS,minY,maxY,obsBins); // Smith-Waterman alignment (predicted vs observed) int alignedLen = smithWatermanAlign(predBins,nPoints, obsBins,nPoints, 3, 2, 2, alignedPred, alignedObs); if(alignedLen < m){ // alignment too small; stop and return current coeffs break; } // Build arrays of x,y for aligned pairs (use observed index as source of real y) float xa = (float)malloc(sizeof(float)*alignedLen); float ya = (float)malloc(sizeof(float)*alignedLen); if(!xa||!ya){ free(xa); free(ya); break; } for(int k=0;k<alignedLen;++k){ xa[k] = x[alignedObs[k]]; // use observed indices to get original x,y mapping ya[k] = y[alignedObs[k]]; } // Fit on aligned pairs float newCoeffs = (float)malloc(sizeof(float)m); bool ok = fitPolyLS_QR(xa,ya,alignedLen,degree,newCoeffs); free(xa); free(ya); if(!ok){ free(newCoeffs); break; } // compute change float delta2 = 0.0f; for(int j=0;j<m;++j){ float d = newCoeffs[j] - coeffs[j]; delta2 += dd; coeffs[j]=newCoeffs[j]; } free(newCoeffs); // update inlier mask from alignment (mark aligned observed indices true, others false) for(int i=0;i<nPoints;++i) inlierMask[i] = false; for(int k=0;k<alignedLen;++k) inlierMask[ alignedObs[k] ] = true; if(sqrt(delta2) < tol) break; }

// copy out for(int j=0;j<m;++j) outCoeffs[j]=coeffs[j];

free(coeffs); free(coeffs_prev); free(predY); free(predBins); free(obsBins); free(alignedPred); free(alignedObs); return true; }

// -------------------------- Example usage ---------------------------

void printCoeffs(const float *c, int m){ for(int i=0;i<m;++i){ Serial.print("c["); Serial.print(i); Serial.print("]="); Serial.print(c[i],6); if(i < m-1) Serial.print(", "); } Serial.println(); }

void setup(){ Serial.begin(115200); delay(500); Serial.println("Smith-Waterman inspired polynomial fit example");

// Example dataset (normalized x,y). We'll synthesize a cubic with noise plus an outlier region const int N = 60; float x[N]; float y[N]; for(int i=0;i<N;++i){ x[i] = (float)i / (N-1); } // ground truth: y = 0.5 - 1.2x + 0.9x^2 - 0.3x^3 for(int i=0;i<N;++i){ float xv = x[i]; float yy = 0.5f -1.2fxv + 0.9fxvxv -0.3fxvxv*xv; // base // add noise float noise = (float)random(-50,51)/1000.0f; // +/-0.05 y[i] = yy + noise; } // inject an outlier segment for(int i=20;i<30;++i) y[i] += 0.6f;

int degree = 3; float coeffs[MAX_DEGREE+1]; bool mask[N]; bool ok = smithWatermanPolyFit(x,y,N,degree,10,1e-4f,coeffs,mask); if(!ok){ Serial.println("Fitting failed"); return; } Serial.println("Fitted coefficients:"); printCoeffs(coeffs,degree+1); Serial.println("Inlier mask (index:included)"); for(int i=0;i<N;++i) if(mask[i]){ Serial.print(i); Serial.print(' ');} Serial.println(); }

void loop(){ // nothing }

