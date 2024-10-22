// GF(256) LFSR implementation for ESP32
#include <Arduino.h>

// Irreducible polynomial for GF(256): x⁸ + x⁴ + x³ + x + 1
#define GF256_POLY 0x11B

class GF256LFSR {
private:
    uint8_t state;
    uint8_t feedback_poly;
    
    // Helper function for GF(256) multiplication
    uint8_t gf256_mult(uint8_t a, uint8_t b) {
        uint8_t result = 0;
        uint8_t hi_bit_set;
        
        for (int i = 0; i < 8; i++) {
            if (b & 1)
                result ^= a;
            hi_bit_set = a & 0x80;
            a <<= 1;
            if (hi_bit_set)
                a ^= (GF256_POLY & 0xFF);
            b >>= 1;
        }
        return result;
    }

public:
    GF256LFSR(uint8_t initial_state, uint8_t polynomial) {
        state = initial_state;
        feedback_poly = polynomial;
    }
    
    uint8_t next() {
        uint8_t feedback = 0;
        uint8_t out = state;
        
        // Calculate feedback using polynomial
        for (int i = 0; i < 8; i++) {
            if (feedback_poly & (1 << i)) {
                feedback ^= gf256_mult(state, 1 << i);
            }
        }
        
        state = feedback;
        return out;
    }
    
    uint8_t getState() {
        return state;
    }
};

// Class to handle LFSR product operations
class LFSRProduct {
private:
    GF256LFSR* lfsr1;
    GF256LFSR* lfsr2;
    uint8_t product_poly;

public:
    LFSRProduct(uint8_t init1, uint8_t poly1, uint8_t init2, uint8_t poly2) {
        lfsr1 = new GF256LFSR(init1, poly1);
        lfsr2 = new GF256LFSR(init2, poly2);
        
        // Calculate product polynomial
        product_poly = gf256_multiply_poly(poly1, poly2);
    }
    
    ~LFSRProduct() {
        delete lfsr1;
        delete lfsr2;
    }
    
    uint8_t gf256_multiply_poly(uint8_t p1, uint8_t p2) {
        uint8_t result = 0;
        uint8_t temp;
        
        for (int i = 0; i < 8; i++) {
            if (p1 & (1 << i)) {
                temp = p2;
                for (int j = 0; j < i; j++) {
                    temp = gf256_mult(temp, 2); // Multiply by x in GF(256)
                }
                result ^= temp;
            }
        }
        return result;
    }
    
    uint8_t next() {
        uint8_t a = lfsr1->next();
        uint8_t b = lfsr2->next();
        return gf256_mult(a, b);
    }
    
    uint8_t getProductPoly() {
        return product_poly;
    }
};

void setup() {
    Serial.begin(115200);
    
    // Example polynomials in GF(256)
    uint8_t poly1 = 0x1B; // x⁴ + x³ + x + 1
    uint8_t poly2 = 0x0D; // x³ + x² + 1
    
    // Initial states
    uint8_t init1 = 0x01;
    uint8_t init2 = 0x01;
    
    LFSRProduct product_lfsr(init1, poly1, init2, poly2);
    
    Serial.println("LFSR Product Demo in GF(256)");
    Serial.print("Product Polynomial: 0x");
    Serial.println(product_lfsr.getProductPoly(), HEX);
    
    // Generate and print first 10 values
    Serial.println("First 10 output values:");
    for (int i = 0; i < 10; i++) {
        uint8_t output = product_lfsr.next();
        Serial.print("0x");
        Serial.println(output, HEX);
        delay(100);
    }
}

void loop() {
    // Empty loop
}
