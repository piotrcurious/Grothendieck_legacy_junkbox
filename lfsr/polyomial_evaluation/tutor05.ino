#include <Arduino.h>

// Class to represent elements of GF(8)
class GF8 {
  public:
    byte value;

    GF8(byte v = 0) : value(v % 8) {}

    GF8 operator+(const GF8& other) const {
      return GF8(value ^ other.value);
    }

    GF8 operator*(const GF8& other) const {
      byte a = value, b = other.value, result = 0;
      for (int i = 0; i < 3; i++) {
        if (b & 1) result ^= a;
        byte high_bit = a & 4;
        a <<= 1;
        if (high_bit) a ^= 0b1011;  // x^3 + x + 1
        b >>= 1;
      }
      return GF8(result);
    }

    bool operator==(const GF8& other) const {
      return value == other.value;
    }
};

// Class to represent polynomials over GF(8)
class Polynomial {
  public:
    static const int MAX_DEGREE = 5;
    GF8 coeffs[MAX_DEGREE + 1];

    Polynomial() {
      for (int i = 0; i <= MAX_DEGREE; i++) {
        coeffs[i] = GF8(0);
      }
    }

    GF8 evaluate(GF8 x) const {
      GF8 result(0);
      GF8 x_power(1);
      for (int i = 0; i <= MAX_DEGREE; i++) {
        result = result + coeffs[i] * x_power;
        x_power = x_power * x;
      }
      return result;
    }
};

// Function to convert polynomial to LFSR
void polynomial_to_lfsr(const Polynomial& poly, Polynomial& feedback, GF8 initial_state[]) {
  feedback = Polynomial();
  feedback.coeffs[0] = GF8(1);
  feedback.coeffs[MAX_DEGREE] = poly.coeffs[MAX_DEGREE];
  
  for (int i = 0; i < MAX_DEGREE; i++) {
    initial_state[i] = poly.coeffs[i];
  }
}

// Function to run LFSR
GF8 run_lfsr(const Polynomial& feedback, GF8 state[]) {
  GF8 output = state[0];
  GF8 new_element(0);
  
  for (int i = 0; i <= MAX_DEGREE; i++) {
    new_element = new_element + feedback.coeffs[i] * state[i];
  }
  
  for (int i = 0; i < MAX_DEGREE - 1; i++) {
    state[i] = state[i + 1];
  }
  state[MAX_DEGREE - 1] = new_element;
  
  return output;
}

// Global variables
Polynomial poly, feedback;
GF8 state[Polynomial::MAX_DEGREE];

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect
  }

  // Initialize random polynomial
  randomSeed(analogRead(0));
  for (int i = 0; i <= Polynomial::MAX_DEGREE; i++) {
    poly.coeffs[i] = GF8(random(8));
  }

  Serial.println("Original polynomial coefficients:");
  for (int i = 0; i <= Polynomial::MAX_DEGREE; i++) {
    Serial.print(poly.coeffs[i].value);
    Serial.print(" ");
  }
  Serial.println();

  // Convert to LFSR
  polynomial_to_lfsr(poly, feedback, state);

  Serial.println("LFSR feedback polynomial coefficients:");
  for (int i = 0; i <= Polynomial::MAX_DEGREE; i++) {
    Serial.print(feedback.coeffs[i].value);
    Serial.print(" ");
  }
  Serial.println();

  Serial.println("LFSR initial state:");
  for (int i = 0; i < Polynomial::MAX_DEGREE; i++) {
    Serial.print(state[i].value);
    Serial.print(" ");
  }
  Serial.println();
}

void loop() {
  Serial.println("LFSR output and verification:");
  GF8 x(2);  // Generator of GF(8)*
  GF8 x_power(1);

  for (int i = 0; i < 20; i++) {
    GF8 lfsr_output = run_lfsr(feedback, state);
    GF8 poly_output = poly.evaluate(x_power);

    Serial.print("Step ");
    Serial.print(i);
    Serial.print(": LFSR = ");
    Serial.print(lfsr_output.value);
    Serial.print(", P(");
    Serial.print(x_power.value);
    Serial.print(") = ");
    Serial.print(poly_output.value);
    Serial.println(lfsr_output == poly_output ? " (Match)" : " (Mismatch)");

    x_power = x_power * x;
  }

  Serial.println("Press any key to run again...");
  while (!Serial.available()) {
    ; // wait for input
  }
  while (Serial.available()) {
    Serial.read();  // clear the input buffer
  }
}
