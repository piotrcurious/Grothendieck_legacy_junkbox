#include <Arduino.h>
#include <math.h>

#define N 32                      // Number of samples (keep it small for demo)
#define PI 3.14159265358979323846

// Cutoff frequency (normalized): frequencies with |xi| < cutoff go to the low–band.
const float cutoff = 0.2;         // Adjust as needed

// Input signal array (length N)
float inputSignal[N];
// Output arrays for low–band and high–band signals
float lowBandSignal[N];
float highBandSignal[N];

/*
  applyWeylFilter()

  This function implements a discrete version of the Weyl–quantized operator.
  For a given sample index x_index, it computes

     output(x_index) = (1/(N^2)) * sum_y sum_k [ input[y] * exp(i 2pi (x_index - y)*xi) ]
     
  but only includes the frequency components xi that lie in the desired band:
    - if lowBand is true, include frequencies with |xi| < cutoff.
    - if lowBand is false, include frequencies with |xi| >= cutoff.

  (For simplicity we use only the real part (cosine) and assume the signal is real.)
*/
float applyWeylFilter(const float input[], int N, int x_index, float cutoff, bool lowBand) {
  float sum = 0.0;
  
  // Loop over all time samples y
  for (int y = 0; y < N; y++) {
    // Loop over all discrete frequency indices k
    for (int k = 0; k < N; k++) {
      // To get symmetric (negative and positive) frequencies,
      // shift the index when k > N/2.
      int k_shifted = (k <= N/2) ? k : k - N;
      float xi = (float)k_shifted / N;  // normalized frequency
      
      // Decide whether to include this frequency in the filter
      bool include = false;
      if (lowBand) {
        if (fabs(xi) < cutoff)
          include = true;
      } else {
        if (fabs(xi) >= cutoff)
          include = true;
      }
      
      if (include) {
        // Compute the phase factor:
        // exp(i 2pi (x_index - y)*xi) --> here we take only the real (cosine) part.
        float phase = 2.0 * PI * (x_index - y) * xi;
        sum += input[y] * cos(phase);
      }
    }
  }
  // Normalize the sum (the normalization factor depends on the discretization)
  return sum / (N * N);
}

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // wait for serial port to connect.
  }
  
  Serial.println("ESP32 Arduino: Weyl Algebra Band Splitting Filter Demo");
  
  // --- Create a test signal ---
  // Here the input is a sum of two sine waves:
  //   - a low frequency sine (normalized frequency 0.1)
  //   - a high frequency sine (normalized frequency 0.35)
  for (int i = 0; i < N; i++) {
    // You could also think of 'i' as discrete time (with t = i/N)
    inputSignal[i] = sin(2.0 * PI * 0.1 * i) + 0.5 * sin(2.0 * PI * 0.35 * i);
  }
  
  // Print the input signal
  Serial.println("Input Signal:");
  for (int i = 0; i < N; i++) {
    Serial.print(inputSignal[i], 4);
    Serial.print(" ");
  }
  Serial.println();
  
  // --- Apply the Weyl–based band splitting filter ---
  // For each sample, compute the low–band and high–band outputs.
  for (int x = 0; x < N; x++) {
    lowBandSignal[x] = applyWeylFilter(inputSignal, N, x, cutoff, true);
    highBandSignal[x] = applyWeylFilter(inputSignal, N, x, cutoff, false);
  }
  
  // Print the low–band output
  Serial.println("Low Band Signal:");
  for (int i = 0; i < N; i++) {
    Serial.print(lowBandSignal[i], 4);
    Serial.print(" ");
  }
  Serial.println();
  
  // Print the high–band output
  Serial.println("High Band Signal:");
  for (int i = 0; i < N; i++) {
    Serial.print(highBandSignal[i], 4);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  // Nothing to do in loop.
}
