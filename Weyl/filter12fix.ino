void processFrequencyDomain() {
  // Convert FFT results to fixed‑point
  for (int i = 0; i < BUFFER_SIZE; i++) {
    fixedF[i] = fc_from_double(realBuffer[i], imagBuffer[i]);
  }
  
  // Normalized frequency spacing and field constants
  int32_t deltaNorm = double_to_fixed(1.0 / BUFFER_SIZE);
  int32_t lambda = double_to_fixed(0.3);  // Field strength parameter
  int32_t eps = double_to_fixed(0.01);    // Small regularization constant
  
  // Compute the field gradient D for each frequency bin
  fixed_complex D[BUFFER_SIZE];
  
  // Zero frequency (DC) component requires special handling
  D[0].real = 0;
  D[0].imag = 0;
  if (BUFFER_SIZE > 1) {
    // Use one-sided difference for DC component
    fixed_complex diff = fc_sub(fixedF[1], fixedF[0]);
    int32_t scale = q_div(Q_ONE, deltaNorm);
    D[0] = fc_mul_scalar(diff, scale);
  }
  
  // Interior points use central difference
  for (int i = 1; i < BUFFER_SIZE - 1; i++) {
    // Central difference formula: (f[i+1] - f[i-1])/(2Δx)
    fixed_complex diff = fc_sub(fixedF[i+1], fixedF[i-1]);
    int32_t scale = q_div(Q_ONE, q_mul(deltaNorm, double_to_fixed(2.0)));
    D[i] = fc_mul_scalar(diff, scale);
  }
  
  // Nyquist frequency component also needs special handling
  D[BUFFER_SIZE-1].real = 0;
  D[BUFFER_SIZE-1].imag = 0;
  if (BUFFER_SIZE > 1) {
    // Use one-sided difference for Nyquist component
    fixed_complex diff = fc_sub(fixedF[BUFFER_SIZE-1], fixedF[BUFFER_SIZE-2]);
    int32_t scale = q_div(Q_ONE, deltaNorm);
    D[BUFFER_SIZE-1] = fc_mul_scalar(diff, scale);
  }
  
  // Process each frequency bin with corrected field equations
  for (int i = 0; i < BUFFER_SIZE; i++) {
    // Normalized frequency x = i/(N/2) for i up to N/2
    int32_t x;
    if (i <= BUFFER_SIZE/2) {
      x = double_to_fixed((double)i / (BUFFER_SIZE/2.0));
    } else {
      // Mirror frequencies for i > N/2
      x = double_to_fixed((double)(BUFFER_SIZE - i) / (BUFFER_SIZE/2.0));
    }
    
    // Compute squared term with better numerical stability
    int32_t x_sq = q_mul(x, x);
    
    // Field potential: V(x) = x² - x + λ
    int32_t V = x_sq - x + lambda;
    
    // Field gradient: ∇V = 2x - 1
    int32_t grad_V = q_mul(double_to_fixed(2.0), x) - Q_ONE;
    
    // Compute regularized field operator: F - D + F·(∇V)
    fixed_complex grad_term = fc_mul_scalar(fixedF[i], grad_V);
    fixed_complex P = fc_sub(fixedF[i], D[i]);
    P = fc_add(P, grad_term);
    
    // Regularized attenuation function: H(x) = 1/(1 + V² + ε)
    int32_t V_sq = q_mul(V, V);
    int32_t denom = Q_ONE + V_sq + eps;
    int32_t attenuation = q_div(Q_ONE, denom);
    
    // Apply attenuation to field operator
    fixedF[i] = fc_mul_scalar(P, attenuation);
  }
  
  // Convert back to double precision for inverse FFT
  for (int i = 0; i < BUFFER_SIZE; i++) {
    double r, im;
    fc_to_double(fixedF[i], r, im);
    realBuffer[i] = r;
    imagBuffer[i] = im;
  }
}
