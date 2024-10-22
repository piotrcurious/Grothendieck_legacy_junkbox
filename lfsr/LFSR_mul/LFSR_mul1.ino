#include <Arduino.h>

// LFSR Class
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

// Function to multiply two polynomials over GF(2)
uint32_t polyMulGF2(uint32_t p1, uint32_t p2) {
  uint32_t result = 0;
  while (p2) {
    if (p2 & 1) {
      result ^= p1;  // Add p1 if the LSB of p2 is 1 (since in GF(2), addition is XOR)
    }
    p1 <<= 1;
    p2 >>= 1;
  }
  return result;
}

// Function to multiply two polynomials over GF(16)
uint32_t polyMulGF16(uint32_t p1, uint32_t p2, uint32_t modPoly) {
  uint32_t result = 0;
  while (p2) {
    if (p2 & 1) {
      result ^= p1;  // Add p1 if the LSB of p2 is 1
    }
    p1 <<= 1;
    if (p1 & (1 << 16)) {  // If degree of p1 exceeds 16, reduce mod modPoly
      p1 ^= modPoly;
    }
    p2 >>= 1;
  }
  return result;
}

// Polynomial modulo for GF(16)
uint32_t modPolyGF16 = 0x13;  // x^4 + x + 1

void setup() {
  Serial.begin(115200);

  // Initialize two LFSRs in GF(2)
  LFSR lfsr1(0x5, 0x3, 3);  // Polynomial: x^3 + x + 1
  LFSR lfsr2(0x3, 0x7, 3);  // Polynomial: x^3 + x^2 + 1

  // Test multiplication in GF(2)
  uint32_t poly1_GF2 = 0x5;  // Polynomial x^2 + 1
  uint32_t poly2_GF2 = 0x7;  // Polynomial x^2 + x + 1
  uint32_t result_GF2 = polyMulGF2(poly1_GF2, poly2_GF2);
  
  Serial.print("Result of GF(2) Polynomial Multiplication: 0x");
  Serial.println(result_GF2, HEX);

  // Test multiplication in GF(16)
  uint32_t poly1_GF16 = 0x9;  // Polynomial x^3 + 1
  uint32_t poly2_GF16 = 0x6;  // Polynomial x^2 + x
  uint32_t result_GF16 = polyMulGF16(poly1_GF16, poly2_GF16, modPolyGF16);

  Serial.print("Result of GF(16) Polynomial Multiplication: 0x");
  Serial.println(result_GF16, HEX);
}

void loop() {
  // Output LFSR sequences for monitoring
  static LFSR lfsr1(0x5, 0x3, 3);  // LFSR1
  static LFSR lfsr2(0x3, 0x7, 3);  // LFSR2

  uint8_t bit1 = lfsr1.next();
  uint8_t bit2 = lfsr2.next();

  Serial.print("LFSR1 output: ");
  Serial.println(bit1);
  Serial.print("LFSR2 output: ");
  Serial.println(bit2);

  delay(1000);
}
