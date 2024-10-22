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

// Function to print an LFSR sequence
void printLFSRSequence(LFSR &lfsr, int numSteps) {
  Serial.print("LFSR Sequence: ");
  for (int i = 0; i < numSteps; i++) {
    Serial.print(lfsr.next());
    Serial.print(" ");
  }
  Serial.println();
}

// Cyclotomic polynomial for n = 5 in GF(2): x^4 + x^3 + x^2 + x + 1
uint32_t cyclotomicPolynomial5 = 0x1F;  // Feedback taps for x^4 + x^3 + x^2 + x + 1

// Primitive polynomial for maximal length LFSR: x^3 + x^2 + 1
uint32_t primitivePolynomial3 = 0x6;  // x^3 + x^2 + 1 (maximal-length LFSR for 3 bits)

// Analyze Cyclotomic LFSRs
void analyzeCyclotomicLFSRs() {
  // Cyclotomic LFSR (using cyclotomic polynomial for n=5)
  Serial.println("Analyzing Cyclotomic LFSR for n=5 (x^4 + x^3 + x^2 + x + 1):");
  LFSR cyclotomicLFSR(0x1, cyclotomicPolynomial5, 5);
  printLFSRSequence(cyclotomicLFSR, 31);  // Expect period of 31 (2^5 - 1)

  // Primitive LFSR for maximal length
  Serial.println("Analyzing Primitive LFSR for n=3 (x^3 + x^2 + 1):");
  LFSR primitiveLFSR(0x1, primitivePolynomial3, 3);
  printLFSRSequence(primitiveLFSR, 7);  // Expect period of 7 (2^3 - 1)
}

void setup() {
  Serial.begin(115200);
  analyzeCyclotomicLFSRs();
}

void loop() {
  // Loop code can go here if necessary
}
