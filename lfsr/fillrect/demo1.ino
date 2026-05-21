/*
  ESP32 / Arduino demo:
  - Coho-inspired seed derivation from rectangle geometry
  - LFSR-driven key schedule
  - Bijective permutation over a power-of-two domain
  - Cycle-walking to cover exactly w*h pixels once
  - Output in pseudo-random pixel order

  Replace plotPixel(x, y, pass) with your display's pixel-drawing call.
*/

#include <Arduino.h>

struct Rect {
  int16_t x0;
  int16_t y0;
  uint16_t w;
  uint16_t h;
};

static uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

static uint8_t ceilLog2_u32(uint32_t v) {
  uint8_t bits = 0;
  uint32_t p = 1;
  while (p < v && bits < 32) {
    p <<= 1;
    bits++;
  }
  return bits;
}

// Simple 32-bit Galois LFSR.
// For a maximal-period LFSR, use a primitive feedback polynomial for your width.
class LFSR32 {
public:
  explicit LFSR32(uint32_t seed = 1) { reseed(seed); }

  void reseed(uint32_t seed) {
    state = (seed == 0) ? 1u : seed;
  }

  uint32_t next() {
    // Feedback mask chosen for a compact demo.
    // This is not the only possible polynomial/mask.
    uint32_t lsb = state & 1u;
    state >>= 1;
    if (lsb) {
      state ^= 0x80200003u;
    }
    return state;
  }

  uint32_t peek() const { return state; }

private:
  uint32_t state = 1;
};

class RectPixelWalker {
public:
  void begin(int16_t x0, int16_t y0, uint16_t w, uint16_t h, uint32_t userSeed) {
    rect.x0 = x0;
    rect.y0 = y0;
    rect.w  = w;
    rect.h  = h;

    total = (uint32_t)w * (uint32_t)h;
    if (total == 0) {
      total = 1;
    }

    bits = ceilLog2_u32(total);
    if (bits == 0) bits = 1;

    mask = (bits == 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);

    // "Cohomology-inspired" boundary seed:
    // combine vertices / edge lengths / user seed into a stable base.
    uint32_t boundary0 = mix32((uint32_t)x0 ^ ((uint32_t)y0 << 16));
    uint32_t boundary1 = mix32((uint32_t)w  ^ ((uint32_t)h  << 16));
    uint32_t base = mix32(userSeed ^ boundary0 ^ (boundary1 << 1) ^ (uint32_t)(w * 131u + h * 197u));

    // Three LFSRs standing in for three independent geometric "cochains":
    // horizontal, vertical, and boundary coupling.
    lfsrA.reseed(mix32(base ^ 0xA341316Cu));
    lfsrB.reseed(mix32(base ^ 0xC8013EA4u));
    lfsrC.reseed(mix32(base ^ 0xAD90777Du));

    // Key schedule for the permutation.
    keys[0] = lfsrA.next() ^ mix32(base + 0x9E3779B9u);
    keys[1] = lfsrB.next() ^ mix32(base + 0x7F4A7C15u);
    keys[2] = lfsrC.next() ^ mix32(base + 0x94D049BBu);
    keys[3] = lfsrA.next() ^ lfsrB.next() ^ lfsrC.next();

    // Make sure at least one multiplier is odd and nonzero in the masked domain.
    if (bits < 32) {
      uint32_t domainMask = mask;
      keys[0] &= domainMask;
      keys[1] &= domainMask;
      keys[2] &= domainMask;
      keys[3] &= domainMask;
    }

    idx = 0;
  }

  bool next(int16_t &outX, int16_t &outY) {
    if (idx >= total) {
      return false;
    }

    uint32_t v = idx++;
    // Cycle-walk until the permutation lands inside [0, total).
    do {
      v = permute(v);
    } while (v >= total);

    outX = rect.x0 + (int16_t)(v % rect.w);
    outY = rect.y0 + (int16_t)(v / rect.w);
    return true;
  }

private:
  Rect rect{};
  uint32_t total = 0;
  uint8_t bits = 0;
  uint32_t mask = 0;
  uint32_t idx = 0;
  uint32_t keys[4]{};

  LFSR32 lfsrA{1}, lfsrB{2}, lfsrC{3};

  uint32_t permute(uint32_t x) const {
    x &= mask;

    // A bijection on 2^bits states:
    // xor-shifts + odd multipliers + adds/xors.
    // Every step is invertible in the masked domain.
    x ^= (x >> 16);
    x &= mask;

    x = (uint32_t)(((uint64_t)x * ((keys[0] | 1u))) & mask);

    x ^= (x >> 11);
    x &= mask;

    x = (uint32_t)(((uint64_t)x * ((keys[1] | 1u))) & mask);

    x ^= (x >> 7);
    x &= mask;

    x = (x + keys[2]) & mask;
    x ^= keys[3];
    x &= mask;

    // One more reversible diffusion step.
    x ^= (x >> 13);
    x &= mask;

    return x;
  }
};

RectPixelWalker walker;

// Replace this with your TFT / framebuffer write.
void plotPixel(int16_t x, int16_t y, uint32_t pass) {
  Serial.printf("pass=%lu  x=%d  y=%d\n", (unsigned long)pass, x, y);
}

void setup() {
  Serial.begin(115200);
  delay(500);

  // Example rectangle: arbitrary position and size.
  walker.begin(
    20,    // x0
    30,    // y0
    16,    // width
    12,    // height
    0x12345678u  // user seed
  );

  Serial.println("Starting pseudo-random rectangle traversal...");
}

void loop() {
  static uint32_t pass = 0;
  static int16_t x, y;

  if (walker.next(x, y)) {
    plotPixel(x, y, pass);
    delay(5); // slow enough to watch in Serial Monitor
  } else {
    pass++;
    Serial.println("---- rectangle complete; restarting with new seed ----");

    // Change seed each pass so the next traversal order changes.
    walker.begin(
      20,
      30,
      16,
      12,
      0x12345678u ^ mix32(pass * 0x9E3779B9u)
    );

    delay(250);
  }
}
