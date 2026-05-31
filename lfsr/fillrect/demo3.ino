#include <Arduino.h>

/**
 * RectPixelWalker: Visits every pixel in a rectangle exactly once in a 
 * pseudo-random order using a Galois LFSR and Cycle-Walking.
 */
class RectPixelWalker {
public:
    /**
     * @param x0, y0 : Top-left corner
     * @param w, h   : Dimensions
     * @param seed   : Random seed (0 is handled internally)
     */
    bool begin(int16_t x0, int16_t y0, uint16_t w, uint16_t h, uint32_t seed) {
        if (w == 0 || h == 0) return false;

        x0_ = x0; y0_ = y0; w_ = w; h_ = h;
        total_ = (uint32_t)w * h;
        
        // Find smallest n such that 2^n >= total_
        // Using __builtin_clz is a high-performance way to find the bit-width on ESP32/ARM
        uint8_t leadingZeros = __builtin_clz(total_ - 1);
        bits_ = 32 - leadingZeros;
        if (bits_ < 2) bits_ = 2; 

        tap_ = getPrimitiveTap(bits_);
        
        // LFSR cannot be 0. We XOR the seed with a constant to ensure 
        // 0-seeds work and stay within the current bit-mask.
        uint32_t mask = (1UL << bits_) - 1;
        state_ = (seed ^ 0xACE1u) & mask;
        if (state_ == 0) state_ = 1; 

        count_ = 0;
        return true;
    }

    /**
     * Retrieves the next pixel. 
     * @return true if a pixel was found, false if the rectangle is fully traversed.
     */
    bool next(int16_t &outX, int16_t &outY) {
        if (count_ >= total_) return false;

        uint32_t v;
        while (true) {
            // LFSR range is [1, 2^n - 1]. We subtract 1 to map to [0, 2^n - 2].
            v = state_ - 1;
            
            // Advance the LFSR
            bool lsb = state_ & 1;
            state_ >>= 1;
            if (lsb) state_ ^= tap_;

            // Cycle-walking: if the value is in our range, we use it.
            if (v < total_) break;
        }

        outX = x0_ + (int16_t)(v % w_);
        outY = y0_ + (int16_t)(v / w_);
        count_++;
        return true;
    }

    uint32_t remaining() const { return total_ - count_; }

private:
    int16_t x0_, y0_;
    uint16_t w_, h_;
    uint32_t total_, count_, state_, tap_;
    uint8_t bits_;

    // Table of primitive polynomials for Galois LFSRs (taps)
    // These provide the maximum period for each bit-width.
    static uint32_t getPrimitiveTap(uint8_t bits) {
        switch (bits) {
            case 2:  return 0x3;          case 3:  return 0x6;
            case 4:  return 0xC;          case 5:  return 0x14;
            case 6:  return 0x30;         case 7:  return 0x60;
            case 8:  return 0xB8;         case 9:  return 0x110;
            case 10: return 0x240;        case 11: return 0x500;
            case 12: return 0xE08;        case 13: return 0x1C80;
            case 14: return 0x3501;       case 15: return 0x6000;
            case 16: return 0xB400;       case 17: return 0x12000;
            case 18: return 0x20400;      case 19: return 0x72000;
            case 20: return 0x90000;      case 21: return 0x140000;
            case 22: return 0x300000;     case 23: return 0x420000;
            case 24: return 0xE10000;     case 25: return 0x1200000;
            case 26: return 0x2000023;    case 27: return 0x4000013;
            case 28: return 0x9000000;    case 29: return 0x14000000;
            case 30: return 0x20000029;   case 31: return 0x48000000;
            case 32: return 0x80200003;
            default: return 0x80200003;
        }
    }
};

// --- Demo Code ---

RectPixelWalker walker;

void setup() {
    Serial.begin(115200);
    delay(1000);

    // Initialize with a rectangle at (0,0) size 128x64
    uint32_t startSeed = 0x1337;
    if (!walker.begin(0, 0, 128, 64, startSeed)) {
        Serial.println("Initialization Failed");
    }
}

void loop() {
    int16_t x, y;
    static uint32_t lastMillis = 0;

    if (walker.next(x, y)) {
        // Output coordinates. In a real app, use display.drawPixel(x, y, color);
        Serial.printf("Pixel: %d, %d | Remaining: %u\n", x, y, walker.remaining());
        
        // Optional delay to see the "filling" effect
        delay(1); 
    } else {
        Serial.println("Traversal Complete. Restarting with new seed...");
        delay(2000);
        walker.begin(0, 0, 128, 64, millis()); 
    }
}
