#include <Arduino.h>

/**
 * SpriteWalker: Randomly blits a source bitmap to a destination.
 */
class SpriteWalker {
public:
    /**
     * @param x0, y0 : Destination coordinates on screen
     * @param w, h   : Dimensions of the sprite
     * @param bitmap : Pointer to the pixel data (e.g., const uint16_t*)
     * @param seed   : Randomization seed
     */
    bool begin(int16_t x0, int16_t y0, uint16_t w, uint16_t h, const uint16_t* bitmap, uint32_t seed) {
        if (w == 0 || h == 0 || bitmap == nullptr) return false;

        x0_ = x0; y0_ = y0; w_ = w; h_ = h;
        bitmap_ = bitmap;
        total_ = (uint32_t)w * h;
        
        // Find bit-width for the LFSR
        uint8_t leadingZeros = __builtin_clz(total_ - 1);
        bits_ = 32 - leadingZeros;
        if (bits_ < 2) bits_ = 2; 

        tap_ = getPrimitiveTap(bits_);
        
        uint32_t mask = (1UL << bits_) - 1;
        state_ = (seed ^ 0xACE1u) & mask;
        if (state_ == 0) state_ = 1; 

        count_ = 0;
        return true;
    }

    /**
     * Calculates the next pixel to draw.
     * @param outX, outY : Resulting screen coordinates
     * @param outColor   : The color from the source bitmap
     * @return true if pixel is valid, false if blit is finished
     */
    bool next(int16_t &outX, int16_t &outY, uint16_t &outColor) {
        if (count_ >= total_) return false;

        uint32_t v;
        while (true) {
            v = state_ - 1; // LFSR state to 0-based index
            
            // Advance LFSR
            bool lsb = state_ & 1;
            state_ >>= 1;
            if (lsb) state_ ^= tap_;

            if (v < total_) break; // Cycle-walk check
        }

        // Optimization: v is exactly the 1D index of the pixel in the bitmap
        outColor = bitmap_[v];
        
        // Map 1D index to 2D screen coordinates
        outX = x0_ + (int16_t)(v % w_);
        outY = y0_ + (int16_t)(v / w_);
        
        count_++;
        return true;
    }

private:
    int16_t x0_, y0_;
    uint16_t w_, h_;
    const uint16_t* bitmap_;
    uint32_t total_, count_, state_, tap_;
    uint8_t bits_;

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

// --- Usage with a hypothetical display library ---

// Example 16x16 sprite data (RGB565)
const uint16_t myIcon[256] = { /* ... pixel data ... */ };

SpriteWalker blitter;

void startBlit() {
    // Blit myIcon to screen position (50, 50)
    blitter.begin(50, 50, 16, 16, myIcon, micros());
}

void updateBlit() {
    int16_t x, y;
    uint16_t color;

    // Draw 10 pixels per frame to create a "dissolve-in" effect
    for (int i = 0; i < 10; i++) {
        if (blitter.next(x, y, color)) {
            // Replace with your actual display draw function:
            // tft.drawPixel(x, y, color); 
            Serial.printf("Drawing color 0x%04X at %d,%d\n", color, x, y);
        }
    }
}
