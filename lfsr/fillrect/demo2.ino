/*
  ESP32 / Arduino:
  Pseudo-random rectangle traversal — visits every pixel exactly once.

  Method:
    Build a bijection on [0, 2^bits) via iterated xor-shift / multiply / add,
    then cycle-walk each sequential index into the actual pixel domain [0, w*h).
    Because the underlying map is a bijection, cycle-walking produces a bijection
    on [0, w*h), guaranteeing every pixel is visited exactly once.
*/

#include <Arduino.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Murmur3 finaliser — good avalanche, cheap on 32-bit MCU.
static uint32_t mix32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352dU;
    x ^= x >> 15; x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Ceiling log2 of v, minimum 1.  Returns b such that 2^b >= v, b >= 1.
static uint8_t bitsNeeded(uint32_t v) {
    if (v <= 2u) return 1;
    uint8_t b = 0;
    // Use 64-bit p to avoid UB on the final left shift.
    uint64_t p = 1;
    while (p < (uint64_t)v) { p <<= 1; ++b; }
    return b;
}

// One step of a 32-bit Galois LFSR.
// Primitive polynomial x^32 + x^22 + x^2 + x + 1 → period 2^32-1.
static uint32_t lfsr32Step(uint32_t s) {
    return (s >> 1) ^ ((s & 1u) ? 0x80200003u : 0u);
}

// ---------------------------------------------------------------------------
// Pixel walker
// ---------------------------------------------------------------------------

class RectPixelWalker {
public:
    // Returns false if w or h is zero.
    bool begin(int16_t x0, int16_t y0, uint16_t w, uint16_t h, uint32_t seed) {
        if (!w || !h) return false;

        x0_  = x0; y0_ = y0; w_ = w; h_ = h;
        total_ = (uint32_t)w * h;
        bits_  = bitsNeeded(total_);
        mask_  = (bits_ == 32) ? 0xFFFFFFFFu : ((1u << bits_) - 1u);
        idx_   = 0;

        // Derive a stable base from geometry + seed so that different rectangles
        // sharing the same seed still produce independent orderings.
        uint32_t gA   = mix32((uint32_t)x0 ^ ((uint32_t)y0  * 2654435761u));
        uint32_t gB   = mix32((uint32_t)w  ^ ((uint32_t)h   * 2246822519u));
        uint32_t base = mix32(seed ^ gA ^ mix32(gB));

        // Derive four sub-keys.  Step an LFSR between derivations so
        // similar bases cannot accidentally produce equal keys.
        uint32_t s = (base == 0u) ? 0xdeadbeef : base;
        for (int i = 0; i < 4; ++i) {
            s = lfsr32Step(s);
            keys_[i] = mix32(s ^ kSalts[i]);
        }

        // Xor-shift amounts, adapted to bit width.
        // Each must be in (0, bits_) for the step to be a bijection on 2^bits.
        const uint8_t b = bits_;
        sh_[0] = clampShift(b, b / 2);
        sh_[1] = clampShift(b, b / 4);
        sh_[2] = clampShift(b, (uint8_t)(b * 3 / 4));
        sh_[3] = clampShift(b, b / 3);

        return true;
    }

    uint32_t remaining() const { return idx_ < total_ ? total_ - idx_ : 0; }

    // Returns the next pseudo-random pixel, or false when all are exhausted.
    bool next(int16_t &outX, int16_t &outY) {
        if (idx_ >= total_) return false;
        uint32_t v = idx_++;
        do { v = permute(v); } while (v >= total_);
        outX = x0_ + (int16_t)(v % w_);
        outY = y0_ + (int16_t)(v / w_);
        return true;
    }

private:
    // Salt constants for key derivation — arbitrary non-zero 32-bit values.
    static constexpr uint32_t kSalts[4] = {
        0xA341316Cu, 0xC8013EA4u, 0xAD90777Du, 0x7F4A7C15u
    };

    int16_t  x0_ = 0, y0_ = 0;
    uint16_t w_  = 0, h_  = 0;
    uint32_t total_ = 0;
    uint32_t idx_   = 0;
    uint8_t  bits_  = 0;
    uint32_t mask_  = 0;
    uint32_t keys_[4]{};
    uint8_t  sh_[4]{};  // adaptive xor-shift amounts

    // Clamp shift s into [1, bits-1].  Falls back to 1 for 1-bit domains.
    static uint8_t clampShift(uint8_t bits, uint8_t s) {
        if (s < 1) s = 1;
        if (s >= bits && bits > 1) s = bits - 1;
        return s;
    }

    // Bijection on [0, 2^bits_).  Every individual step is invertible:
    //   xor-shift  x ^ (x >> s)  : invertible for 0 < s < bits
    //   multiply   odd factor     : invertible mod 2^bits
    //   add / xor  constant       : trivially invertible
    uint32_t permute(uint32_t x) const {
        x &= mask_;
        x ^= x >> sh_[0];                                      x &= mask_;
        x  = (uint32_t)(((uint64_t)x * (keys_[0] | 1u))     & mask_);
        x ^= x >> sh_[1];                                      x &= mask_;
        x  = (uint32_t)(((uint64_t)x * (keys_[1] | 1u))     & mask_);
        x ^= x >> sh_[2];                                      x &= mask_;
        x  = (x + keys_[2])                                   & mask_;
        x ^= keys_[3];                                         x &= mask_;
        x ^= x >> sh_[3];                                      x &= mask_;
        return x;
    }
};

// out-of-line storage for the constexpr static member (C++14 / older GCC)
constexpr uint32_t RectPixelWalker::kSalts[4];

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

RectPixelWalker walker;

void plotPixel(int16_t x, int16_t y, uint32_t pass) {
    Serial.printf("pass=%lu  x=%d  y=%d\n", (unsigned long)pass, x, y);
}

void setup() {
    Serial.begin(115200);
    delay(500);
    if (!walker.begin(20, 30, 16, 12, 0x12345678u)) {
        Serial.println("Invalid rectangle dimensions");
    }
}

void loop() {
    static uint32_t pass = 0;
    int16_t x, y;

    if (walker.next(x, y)) {
        plotPixel(x, y, pass);
        // Remove or reduce this delay for real display use.
        delay(5);
    } else {
        ++pass;
        Serial.println("---- complete; restarting with new seed ----");
        walker.begin(20, 30, 16, 12, mix32(0x12345678u ^ pass));
        delay(250);
    }
}
