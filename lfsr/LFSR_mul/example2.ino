#include <Arduino.h>

// LFSR Class with feedback based on cyclotomic or primitive polynomials
class LFSR {
  private:
    uint32_t state;  // LFSR state
    uint32_t taps;   // Feedback taps
    uint8_t length;  // Length of LFSR (number of bits)

  public:
    LFSR(uint32_t init_state, uint32_t tap_polynomial, uint8_t len) : state(init_state), taps(tap_polynomial), length(len) {}

    uint8_t next() {
      uint8_t lsb = state & 1;  // Get LSB
      state >>= 1;
      if (lsb) {
        state ^= taps;  // Apply feedback taps if LSB is 1
      }
      return state & 1;
    }

    uint32_t getState() {
      return state;
    }
};

// Polynomial multiplication over GF(16) or higher
uint32_t polyMulGF(uint32_t p1, uint32_t p2, uint32_t modPoly, uint8_t fieldDegree) {
  uint32_t result = 0;
  while (p2) {
    if (p2 & 1) {
      result ^= p1;  // Add p1 if the LSB of p2 is 1
    }
    p1 <<= 1;
    if (p1 & (1 << fieldDegree)) {  // If degree exceeds field degree, reduce mod modPoly
      p1 ^= modPoly;
    }
    p2 >>= 1;
  }
  return result;
}

// Polynomial derivative in GF(2): differentiate polynomial in GF(2)
uint32_t polyDerivativeGF2(uint32_t poly) {
  uint32_t derivative = 0;
  uint32_t mask = 2; // Start at x^1
  while (poly >= mask) {
    if (poly & mask) {
      derivative ^= (mask >> 1); // Differentiate each term: d(x^n)/dx = n * x^(n-1)
    }
    mask <<= 1;
  }
  return derivative;
}

// Function to calculate the first and second derivatives of an LFSR sequence
void analyzeDerivatives(LFSR &lfsr, int numSteps) {
  uint32_t sequence = 0;
  for (int i = 0; i < numSteps; i++) {
    sequence = (sequence << 1) | lfsr.next();
  }
  
  uint32_t firstDerivative = polyDerivativeGF2(sequence);
  uint32_t secondDerivative = polyDerivativeGF2(firstDerivative);

  Serial.print("Original Sequence: 0x");
  Serial.println(sequence, HEX);

  Serial.print("First Derivative: 0x");
  Serial.println(firstDerivative, HEX);

  Serial.print("Second Derivative: 0x");
  Serial.println(secondDerivative, HEX);
}

// Cyclotomic polynomial for n = 5 in GF(2): x^4 + x^3 + x^2 + x + 1
uint32_t cyclotomicPolynomial5 = 0x1F;  // Feedback taps for x^4 + x^3 + x^2 + x + 1

// Primitive polynomial for maximal length LFSR: x^3 + x^2 + 1
uint32_t primitivePolynomial3 = 0x6;  // x^3 + x^2 + 1 (maximal-length LFSR for 3 bits)

// Primitive polynomial for GF(16): x^4 + x + 1
uint32_t modPolyGF16 = 0x13;  // x^4 + x + 1

// Function to multiply two LFSR sequences (signal multiplier)
void multiplyLFSRs(LFSR &lfsr1, LFSR &lfsr2, int numSteps, uint8_t fieldDegree, uint32_t modPoly) {
  uint32_t sequence1 = 0, sequence2 = 0;
  for (int i = 0; i < numSteps; i++) {
    sequence1 = (sequence1 << 1) | lfsr1.next();
    sequence2 = (sequence2 << 1) | lfsr2.next();
  }

  uint32_t result = polyMulGF(sequence1, sequence2, modPoly, fieldDegree);

  Serial.print("LFSR1: 0x");
  Serial.println(sequence1, HEX);
  Serial.print("LFSR2: 0x");
  Serial.println(sequence2, HEX);
  Serial.print("Multiplied Result (GF(");
  Serial.print(fieldDegree);
  Serial.println(")): 0x");
  Serial.println(result, HEX);
}

// Analyze LFSRs with signal analysis (derivatives, periods, feature extraction)
void analyzeLFSRs() {
  // Cyclotomic LFSR (using cyclotomic polynomial for n=5)
  Serial.println("Analyzing Cyclotomic LFSR for n=5 (x^4 + x^3 + x^2 + x + 1):");
  LFSR cyclotomicLFSR(0x1, cyclotomicPolynomial5, 5);
  analyzeDerivatives(cyclotomicLFSR, 31);

  // Primitive LFSR for maximal length
  Serial.println("Analyzing Primitive LFSR for n=3 (x^3 + x^2 + 1):");
  LFSR primitiveLFSR(0x1, primitivePolynomial3, 3);
  analyzeDerivatives(primitiveLFSR, 7);

  // Multiply two LFSR sequences in GF(16)
  Serial.println("Multiplying LFSR sequences in GF(16):");
  LFSR lfsr1(0x9, 0x6, 4);  // First LFSR in GF(16)
  LFSR lfsr2(0x3, 0x7, 4);  // Second LFSR in GF(16)
  multiplyLFSRs(lfsr1, lfsr2, 15, 16, modPolyGF16);  // Use GF(16) mod polynomial
}

void setup() {
  Serial.begin(115200);
  analyzeLFSRs();
}

void loop() {
  // Nothing in loop
}
