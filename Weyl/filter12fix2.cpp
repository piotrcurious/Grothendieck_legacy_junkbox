// ***************************************************************************
// Frequency‑Domain Processing Using Fixed‑Point Algebra
// ***************************************************************************
//
// For each frequency bin we do the following:
//   1. Convert the FFT output (double) to a fixed‑point complex number.
//   2. Compute a finite-difference derivative D using forward, central,
//      or backward differences.
//   3. Form the Weyl operator: P(F) = F - D + F·D.
//   4. Compute a normalized frequency x = i / BUFFER_SIZE, and then a
//      constraint value g(x) = x² - x + λ.
//   5. Define an attenuation (filter) function H(x) = 1 / (1 + [g(x)]²).
//   6. Multiply the Weyl operator result by H(x) to obtain the final filtered
//      coefficient.
void processFrequencyDomain() {
  // Convert FFT results to fixed‑point.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }
  
  // Normalized frequency spacing Δx ≈ 1/BUFFER_SIZE (in fixed‑point).
  int32_t deltaNorm = double_to_fixed(1.0 / BUFFER_SIZE);
  
  // Compute the finite-difference derivative D for each frequency bin.
  fixed_complex D[BUFFER_SIZE];
  
  // Bin 0: forward difference.
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[1], fixedF[0]);
    D[0].real = q_div(diff.real, deltaNorm);
    D[0].imag = q_div(diff.imag, deltaNorm);
  } else {
    D[0].real = 0;
    D[0].imag = 0;
  }
  
  // Bins 1 to BUFFER_SIZE-2: central difference.
  for (int i = 1; i < BUFFER_SIZE - 1; i++) {
    fixed_complex diff = fc_sub(fixedF[i+1], fixedF[i-1]);
    // Denominator is 2*deltaNorm.
    int32_t twoDelta = q_mul(double_to_fixed(2.0), deltaNorm);
    D[i].real = q_div(diff.real, twoDelta);
    D[i].imag = q_div(diff.imag, twoDelta);
  }
  
  // Last bin: backward difference.
  if (BUFFER_SIZE > 1) {
    fixed_complex diff = fc_sub(fixedF[BUFFER_SIZE-1], fixedF[BUFFER_SIZE-2]);
    D[BUFFER_SIZE-1].real = q_div(diff.real, deltaNorm);
    D[BUFFER_SIZE-1].imag = q_div(diff.imag, deltaNorm);
  } else {
    D[BUFFER_SIZE-1].real = 0;
    D[BUFFER_SIZE-1].imag = 0;
  }
  
  // Process each frequency bin.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Apply the Weyl operator: P(F) = F - D + F·D.
    fixed_complex term1 = fc_sub(fixedF[i], D[i]);
    fixed_complex term2 = fc_mul(fixedF[i], D[i]);
    fixed_complex P = fc_add(term1, term2);
    
    // Compute normalized frequency x = i / BUFFER_SIZE in fixed‑point.
    int32_t x = double_to_fixed((double)i / (double)BUFFER_SIZE);
    int32_t x_sq = q_mul(x, x);
    int32_t lambda = double_to_fixed(0.3); // Tuning parameter.
    
    // Compute the constraint value: g(x) = x² - x + λ.
    // The subtraction needs to be in fixed-point, so x needs to be scaled properly
    int32_t g_val = x_sq - x + lambda;
    
    // Define a filter (attenuation) function:
    //    H(x) = 1 / (1 + (g(x))²)
    int32_t g_sq = q_mul(g_val, g_val);
    int32_t onePlus_g_sq = Q_ONE + g_sq;
    int32_t attenuation = q_div(Q_ONE, onePlus_g_sq);
    
    // Final filtered coefficient: Q(F) = P(F) * H(x).
    fixed_complex filtered = fc_mul_scalar(P, attenuation);
    
    // Store the filtered coefficient.
    fixedF[i] = filtered;
  }
  
  // Convert the fixed‑point frequency-domain data back to doubles for the inverse FFT.
  for (int i = 0; i < BUFFER_SIZE; i++) {
    double r, im;
    fc_to_double(fixedF[i], r, im);
    realBuffer[i] = r;
    imagBuffer[i] = im;
  }
}
