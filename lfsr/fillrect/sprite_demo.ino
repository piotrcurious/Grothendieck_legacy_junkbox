#define LGFX_USE_V1
#include <LovyanGFX.hpp>

// --- 1. Display Configuration ---
// This uses LGFX's autodetect feature which works with most ESP32 dev boards.
class LGFX : public lgfx::LGFX_Device {
    lgfx::Panel_Device* _panel_instance = nullptr;
    lgfx::Bus_SPI* _bus_instance = nullptr;

public:
    LGFX(void) {
        // Auto-detect covers many common boards (TTGO, M5Stack, Wio Terminal, etc.)
        // If autodetect fails, you would insert your specific SPI/Pin config here.
    }
};

static LGFX tft;

// --- 2. Optimized LFSR Sprite Walker ---
class SpriteWalker {
public:
    bool begin(int16_t x0, int16_t y0, uint16_t w, uint16_t h, const uint16_t* bitmap, uint32_t seed) {
        if (w == 0 || h == 0 || bitmap == nullptr) return false;

        x0_ = x0; y0_ = y0; w_ = w; h_ = h;
        bitmap_ = bitmap;
        total_ = (uint32_t)w * h;
        
        // Find minimal bit-width (2^n >= total)
        uint8_t leadingZeros = __builtin_clz(total_ - 1);
        bits_ = 32 - leadingZeros;
        if (bits_ < 2) bits_ = 2; 

        tap_ = getPrimitiveTap(bits_);
        
        // Seed LFSR (Must be non-zero and masked)
        uint32_t mask = (1UL << bits_) - 1;
        state_ = (seed ^ 0xACE1u) & mask;
        if (state_ == 0) state_ = 1; 

        count_ = 0;
        return true;
    }

    bool next(int16_t &outX, int16_t &outY, uint16_t &outColor) {
        if (count_ >= total_) return false;

        uint32_t v;
        while (true) {
            v = state_ - 1;
            bool lsb = state_ & 1;
            state_ >>= 1;
            if (lsb) state_ ^= tap_;
            if (v < total_) break; // Cycle-walk
        }

        outColor = bitmap_[v];
        outX = x0_ + (int16_t)(v % w_);
        outY = y0_ + (int16_t)(v / w_);
        
        count_++;
        return true;
    }

    uint32_t remaining() const { return total_ - count_; }

private:
    int16_t x0_, y0_;
    uint16_t w_, h_;
    const uint16_t* bitmap_;
    uint32_t total_, count_, state_, tap_;
    uint8_t bits_;

    static uint32_t getPrimitiveTap(uint8_t bits) {
        static const uint32_t taps[] = {
            0, 0, 0x3, 0x6, 0xC, 0x14, 0x30, 0x60, 0xB8, 0x110, 0x240, 
            0x500, 0xE08, 0x1C80, 0x3501, 0x6000, 0xB400, 0x12000, 
            0x20400, 0x72000, 0x90000, 0x140000, 0x300000, 0x420000, 
            0xE10000, 0x1200000, 0x2000023, 0x4000013, 0x9000000, 
            0x14000000, 0x20000029, 0x48000000, 0x80200003
        };
        return (bits <= 32) ? taps[bits] : taps[32];
    }
};

// --- 3. Global Variables ---
const uint16_t SPRITE_W = 100;
const uint16_t SPRITE_H = 100;
uint16_t* testSprite = nullptr;
SpriteWalker walker;

// Helper to create a colorful test pattern
void createTestSprite() {
    testSprite = (uint16_t*)malloc(SPRITE_W * SPRITE_H * sizeof(uint16_t));
    for (int y = 0; y < SPRITE_H; y++) {
        for (int x = 0; x < SPRITE_W; x++) {
            // Generate a gradient pattern (RGB565)
            uint8_t r = (x * 255) / SPRITE_W;
            uint8_t g = (y * 255) / SPRITE_H;
            uint8_t b = 128;
            testSprite[y * SPRITE_W + x] = tft.color565(r, g, b);
        }
    }
}

// --- 4. Main Program ---

void setup() {
    tft.init();
    tft.setRotation(1);
    tft.fillScreen(TFT_BLACK);
    
    createTestSprite();
    
    // Start first walker at center of 320x240 screen
    int16_t startX = (320 - SPRITE_W) / 2;
    int16_t startY = (240 - SPRITE_H) / 2;
    walker.begin(startX, startY, SPRITE_W, SPRITE_H, testSprite, 42);
}

void loop() {
    int16_t x, y;
    uint16_t color;

    // Draw 250 pixels per frame for a smooth dissolve effect
    // Increase this number for faster blitting
    for (int i = 0; i < 250; i++) {
        if (walker.next(x, y, color)) {
            tft.drawPixel(x, y, color);
        } else {
            // Sequence finished! 
            delay(1500); 
            tft.fillScreen(TFT_BLACK);
            
            // Re-randomize position and seed
            int16_t nextX = random(0, 320 - SPRITE_W);
            int16_t nextY = random(0, 240 - SPRITE_H);
            walker.begin(nextX, nextY, SPRITE_W, SPRITE_H, testSprite, micros());
            break;
        }
    }
}
