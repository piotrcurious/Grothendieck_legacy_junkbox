/*
  ESP32 + LovyanGFX + ST7789 320x240
  SGA3 Exposé IX visualization demo

  What this sketch shows, directly in data and rendering:
  1) A diagonalizable / character-lattice panel: explicit lattice, weights, and character marks.
  2) A polar LUT sprite engine: arbitrary rotation and zoom via circular lookup tables.
  3) A torsion-kernel scene: nG shown as kernels of [n] on a circle.
  4) A lifting/rigidity scene: successive lifts across nilpotent levels and near-identity conjugation.

  Wiring is the usual ST7789 SPI setup; adjust pins below to your board.
  LovyanGFX library required.
*/

#include <Arduino.h>
#include <LovyanGFX.hpp>
#include <math.h>

// -----------------------------------------------------------------------------
// Display configuration
// -----------------------------------------------------------------------------

class LGFX : public lgfx::LGFX_Device {
  lgfx::Panel_ST7789 _panel_instance;
  lgfx::Bus_SPI _bus_instance;
  lgfx::Light_PWM _light_instance;

public:
  LGFX(void) {
    {
      auto cfg = _bus_instance.config();

      cfg.spi_host   = SPI2_HOST;   // VSPI on many ESP32 boards
      cfg.spi_mode   = 0;
      cfg.freq_write = 40000000;
      cfg.freq_read  = 16000000;
      cfg.spi_3wire  = false;
      cfg.use_lock   = true;

      cfg.dma_channel = SPI_DMA_CH_AUTO;

      cfg.pin_sclk = 18;
      cfg.pin_mosi = 23;
      cfg.pin_miso = -1;
      cfg.pin_dc   = 2;

      _bus_instance.config(cfg);
      _panel_instance.setBus(&_bus_instance);
    }

    {
      auto cfg = _panel_instance.config();

      cfg.pin_cs           = 5;
      cfg.pin_rst          = 4;
      cfg.pin_busy         = -1;

      cfg.panel_width      = 320;
      cfg.panel_height     = 240;
      cfg.offset_x         = 0;
      cfg.offset_y         = 0;
      cfg.offset_rotation  = 0;
      cfg.dummy_read_pixel = 8;
      cfg.dummy_read_bits  = 1;
      cfg.readable         = false;
      cfg.invert           = true;
      cfg.rgb_order        = false;
      cfg.dlen_16bit       = false;
      cfg.bus_shared       = false;

      _panel_instance.config(cfg);
    }

    {
      auto cfg = _light_instance.config();
      cfg.pin_bl = 15;
      cfg.invert = false;
      cfg.freq   = 5000;
      cfg.pwm_channel = 7;
      _light_instance.config(cfg);
      _panel_instance.light(&_light_instance);
    }

    setPanel(&_panel_instance);
  }
};

static LGFX lcd;

// -----------------------------------------------------------------------------
// Geometry / palette helpers
// -----------------------------------------------------------------------------

static constexpr int TFT_W = 320;
static constexpr int TFT_H = 240;

static constexpr int ANGLE_STEPS = 3600; // 0.1 degree resolution
static constexpr float ANGLE_TO_RAD = 0.0017453292519943296f; // PI / 1800
static constexpr float RAD_TO_ANGLE_INDEX = 572.9577951308232f; // 1800 / PI

static constexpr int SRC_W = 64;
static constexpr int SRC_H = 64;

static constexpr int WARP_W = 132;
static constexpr int WARP_H = 132;

static constexpr int SRC_PANEL_X = 8;
static constexpr int SRC_PANEL_Y = 16;
static constexpr int WARP_PANEL_X = 110;
static constexpr int WARP_PANEL_Y = 28;
static constexpr int INFO_PANEL_Y = 176;

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
  return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

static inline void unpack565(uint16_t c, uint8_t &r, uint8_t &g, uint8_t &b) {
  r = (uint8_t)(((c >> 11) & 0x1F) * 255 / 31);
  g = (uint8_t)(((c >>  5) & 0x3F) * 255 / 63);
  b = (uint8_t)(( c        & 0x1F) * 255 / 31);
}

static inline uint16_t lerp565(uint16_t a, uint16_t b, float t) {
  uint8_t ar, ag, ab, br, bg, bb;
  unpack565(a, ar, ag, ab);
  unpack565(b, br, bg, bb);
  uint8_t rr = (uint8_t)(ar + (br - ar) * t);
  uint8_t gg = (uint8_t)(ag + (bg - ag) * t);
  uint8_t bb2 = (uint8_t)(ab + (bb - ab) * t);
  return rgb565(rr, gg, bb2);
}

static inline uint16_t blend4(
  uint16_t c00, uint16_t c10, uint16_t c01, uint16_t c11,
  float fx, float fy
) {
  uint16_t a = lerp565(c00, c10, fx);
  uint16_t b = lerp565(c01, c11, fx);
  return lerp565(a, b, fy);
}

static constexpr uint16_t COL_BG      = 0x0000;
static constexpr uint16_t COL_PANEL   = 0x18E3;
static constexpr uint16_t COL_FRAME   = 0x7BEF;
static constexpr uint16_t COL_TEXT    = 0xFFFF;
static constexpr uint16_t COL_SUBTLE   = 0x4208;
static constexpr uint16_t COL_ACCENT1  = 0xF800; // red
static constexpr uint16_t COL_ACCENT2  = 0x07E0; // green
static constexpr uint16_t COL_ACCENT3  = 0x001F; // blue
static constexpr uint16_t COL_ACCENT4  = 0xFFE0; // yellow
static constexpr uint16_t COL_ACCENT5  = 0xF81F; // magenta
static constexpr uint16_t COL_ACCENT6  = 0x07FF; // cyan

// -----------------------------------------------------------------------------
// Trig LUT
// -----------------------------------------------------------------------------

struct TrigLUT {
  int16_t sinv[ANGLE_STEPS];
  int16_t cosv[ANGLE_STEPS];

  void build() {
    for (int i = 0; i < ANGLE_STEPS; ++i) {
      const float a = (float)i * ANGLE_TO_RAD;
      sinv[i] = (int16_t)lrintf(sinf(a) * 32767.0f);
      cosv[i] = (int16_t)lrintf(cosf(a) * 32767.0f);
    }
  }

  inline float sinfIdx(int idx) const {
    idx %= ANGLE_STEPS;
    if (idx < 0) idx += ANGLE_STEPS;
    return (float)sinv[idx] / 32767.0f;
  }

  inline float cosfIdx(int idx) const {
    idx %= ANGLE_STEPS;
    if (idx < 0) idx += ANGLE_STEPS;
    return (float)cosv[idx] / 32767.0f;
  }

  inline int idxFromDeg(float deg) const {
    int idx = (int)lroundf(deg * 10.0f);
    idx %= ANGLE_STEPS;
    if (idx < 0) idx += ANGLE_STEPS;
    return idx;
  }
};

static TrigLUT gTrig;

// -----------------------------------------------------------------------------
// Circular mapping LUT for a square warp window
// -----------------------------------------------------------------------------

struct PolarEntry {
  uint16_t angleIdx;     // base angle in LUT coordinates
  uint16_t radiusQ15;    // normalized radius [0..32767]
  uint8_t  inside;       // 1 if inside the unit disk
};

static PolarEntry gPolarLUT[WARP_W * WARP_H];

static void buildPolarLUT() {
  const float cx = (WARP_W - 1) * 0.5f;
  const float cy = (WARP_H - 1) * 0.5f;
  const float maxR = (float)((WARP_W < WARP_H ? WARP_W : WARP_H) - 1) * 0.5f;

  for (int y = 0; y < WARP_H; ++y) {
    for (int x = 0; x < WARP_W; ++x) {
      const float dx = (x - cx) / maxR;
      const float dy = (y - cy) / maxR;
      const float r = sqrtf(dx * dx + dy * dy);
      PolarEntry &e = gPolarLUT[y * WARP_W + x];
      e.inside = (r <= 1.0f) ? 1 : 0;
      const float ang = atan2f(dy, dx); // [-pi, pi]
      int idx = (int)lroundf((ang + PI) * (float)ANGLE_STEPS / (2.0f * PI));
      idx %= ANGLE_STEPS;
      if (idx < 0) idx += ANGLE_STEPS;
      e.angleIdx = (uint16_t)idx;
      e.radiusQ15 = (uint16_t)lroundf((r > 1.0f ? 1.0f : r) * 32767.0f);
    }
  }
}

// -----------------------------------------------------------------------------
// Source object: explicit lattice + asymmetric motif
// -----------------------------------------------------------------------------

static uint16_t gSourceBuf[SRC_W * SRC_H];
static LGFX_Sprite gSourceSprite(&lcd);
static LGFX_Sprite gWarpSprite(&lcd);

static void setSrcPixel(int x, int y, uint16_t c) {
  if ((unsigned)x < SRC_W && (unsigned)y < SRC_H) {
    gSourceBuf[y * SRC_W + x] = c;
  }
}

static uint16_t sampleSource(float fx, float fy, uint16_t bg) {
  if (fx < 0.0f || fy < 0.0f || fx >= (float)(SRC_W - 1) || fy >= (float)(SRC_H - 1)) return bg;

  const int x0 = (int)fx;
  const int y0 = (int)fy;
  const float tx = fx - x0;
  const float ty = fy - y0;

  const uint16_t c00 = gSourceBuf[y0 * SRC_W + x0];
  const uint16_t c10 = gSourceBuf[y0 * SRC_W + (x0 + 1)];
  const uint16_t c01 = gSourceBuf[(y0 + 1) * SRC_W + x0];
  const uint16_t c11 = gSourceBuf[(y0 + 1) * SRC_W + (x0 + 1)];

  return blend4(c00, c10, c01, c11, tx, ty);
}

static void buildSourceObject() {
  const uint16_t bg = rgb565(8, 10, 18);
  for (int i = 0; i < SRC_W * SRC_H; ++i) gSourceBuf[i] = bg;

  const int cx = SRC_W / 2;
  const int cy = SRC_H / 2;

  // Grid, axes, and basis-like lines.
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      if ((x % 8) == 0 || (y % 8) == 0) {
        gSourceBuf[y * SRC_W + x] = rgb565(18, 26, 40);
      }
    }
  }

  // Outer circle.
  for (int a = 0; a < ANGLE_STEPS; a += 4) {
    const float rad = (float)a * ANGLE_TO_RAD;
    const int x = cx + (int)lroundf(cosf(rad) * 24.0f);
    const int y = cy + (int)lroundf(sinf(rad) * 24.0f);
    setSrcPixel(x, y, COL_FRAME);
  }

  // Filled disk with faint gradient.
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      const float dx = (x - cx) / 24.0f;
      const float dy = (y - cy) / 24.0f;
      const float r = sqrtf(dx * dx + dy * dy);
      if (r <= 1.0f) {
        const uint8_t v = (uint8_t)(60 + (1.0f - r) * 120.0f);
        setSrcPixel(x, y, rgb565(v, v / 2, 255 - v / 2));
      }
    }
  }

  // Central axis cross.
  for (int i = -20; i <= 20; ++i) {
    setSrcPixel(cx + i, cy, rgb565(220, 220, 220));
    setSrcPixel(cx, cy + i, rgb565(220, 220, 220));
  }

  // A diagonal "weight vector" line.
  for (int i = -18; i <= 18; ++i) {
    setSrcPixel(cx + i, cy - i / 2, COL_ACCENT4);
    if (i % 2 == 0) setSrcPixel(cx + i, cy - i / 2 + 1, COL_ACCENT4);
  }

  // Asymmetric bright marker so rotation is visible.
  for (int yy = -3; yy <= 3; ++yy) {
    for (int xx = -3; xx <= 3; ++xx) {
      const int px = cx + 14 + xx;
      const int py = cy - 9 + yy;
      if ((xx * xx + yy * yy) <= 9) {
        setSrcPixel(px, py, rgb565(255, 80, 20));
      }
    }
  }

  // A second marker on the opposite side.
  for (int yy = -2; yy <= 2; ++yy) {
    for (int xx = -2; xx <= 2; ++xx) {
      const int px = cx - 11 + xx;
      const int py = cy + 12 + yy;
      if ((xx * xx + yy * yy) <= 4) {
        setSrcPixel(px, py, COL_ACCENT2);
      }
    }
  }

  // Render buffer into a static preview sprite once.
  gSourceSprite.setColorDepth(16);
  gSourceSprite.createSprite(SRC_W, SRC_H);
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      gSourceSprite.drawPixel(x, y, gSourceBuf[y * SRC_W + x]);
    }
  }
}

// -----------------------------------------------------------------------------
// Warp renderer
// -----------------------------------------------------------------------------

static void renderPolarWarp(float angleDeg, float zoom, int srcCx, int srcCy, float srcRadius) {
  const uint16_t bg = rgb565(4, 7, 12);
  const int rotIdx = gTrig.idxFromDeg(angleDeg);

  for (int y = 0; y < WARP_H; ++y) {
    for (int x = 0; x < WARP_W; ++x) {
      const PolarEntry &e = gPolarLUT[y * WARP_W + x];
      if (!e.inside) {
        gWarpSprite.drawPixel(x, y, bg);
        continue;
      }

      const int idx = (e.angleIdx + rotIdx) % ANGLE_STEPS;
      const float c = gTrig.cosfIdx(idx);
      const float s = gTrig.sinfIdx(idx);
      const float radius = ((float)e.radiusQ15 / 32767.0f) * srcRadius / zoom;

      const float sx = (float)srcCx + c * radius;
      const float sy = (float)srcCy + s * radius;

      const uint16_t cpx = sampleSource(sx, sy, bg);
      gWarpSprite.drawPixel(x, y, cpx);
    }
  }
}

// -----------------------------------------------------------------------------
// Simple shapes and text helpers
// -----------------------------------------------------------------------------

static void drawPanelFrame(int x, int y, int w, int h, const char *title) {
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
  lcd.fillRect(x + 1, y + 1, w - 2, 14, COL_SUBTLE);
  lcd.drawString(title, x + 6, y + 2);
}

static void drawFooter(float angleDeg, float zoom, int scene, int kernelN) {
  lcd.fillRect(0, 208, 320, 32, COL_SUBTLE);
  lcd.setTextColor(COL_TEXT, COL_SUBTLE);
  lcd.setCursor(6, 212);
  lcd.printf("scene=%d  angle=%.1f deg  zoom=%.3f  n=%d", scene, angleDeg, zoom, kernelN);
  lcd.setCursor(6, 224);
  lcd.print("SGA3 IX: local diagonalization, torsion kernels, infinitesimal rigidity");
}

static void drawCharacterLatticeMini(int x, int y, int w, int h, float angleDeg) {
  lcd.drawRect(x, y, w, h, COL_FRAME);
  const int cx = x + w / 2;
  const int cy = y + h / 2;
  const int rotIdx = gTrig.idxFromDeg(angleDeg);

  for (int i = -3; i <= 3; ++i) {
    const int gx = cx + i * 12;
    lcd.drawLine(gx, y + 4, gx, y + h - 5, COL_SUBTLE);
    const int gy = cy + i * 12;
    lcd.drawLine(x + 4, gy, x + w - 5, gy, COL_SUBTLE);
  }

  // Basis vectors
  const float c = gTrig.cosfIdx(rotIdx);
  const float s = gTrig.sinfIdx(rotIdx);
  const int vx = (int)lroundf(c * 22.0f);
  const int vy = (int)lroundf(s * 22.0f);
  lcd.drawLine(cx, cy, cx + vx, cy + vy, COL_ACCENT1);
  lcd.drawLine(cx, cy, cx - vy, cy + vx, COL_ACCENT2);

  // A few lattice points.
  for (int m = -2; m <= 2; ++m) {
    for (int n = -2; n <= 2; ++n) {
      const int px = cx + m * 12 + n * vx / 4;
      const int py = cy + n * 12 + m * vy / 4;
      if (px > x + 2 && px < x + w - 2 && py > y + 2 && py < y + h - 2) {
        lcd.fillCircle(px, py, 2, ((m + n) & 1) ? COL_ACCENT4 : COL_ACCENT6);
      }
    }
  }

  lcd.setCursor(x + 4, y + h - 12);
  lcd.printf("character chart");
}

static void drawTorsionCircle(int cx, int cy, int radius, int n, float phaseDeg) {
  lcd.drawCircle(cx, cy, radius, COL_FRAME);
  lcd.drawCircle(cx, cy, radius - 1, COL_SUBTLE);
  lcd.drawLine(cx - radius, cy, cx + radius, cy, COL_SUBTLE);
  lcd.drawLine(cx, cy - radius, cx, cy + radius, COL_SUBTLE);

  const int phaseIdx = gTrig.idxFromDeg(phaseDeg);
  for (int k = 0; k < n; ++k) {
    const int idx = (phaseIdx + (k * ANGLE_STEPS) / n) % ANGLE_STEPS;
    const float c = gTrig.cosfIdx(idx);
    const float s = gTrig.sinfIdx(idx);
    const int x = cx + (int)lroundf(c * radius);
    const int y = cy + (int)lroundf(s * radius);
    const uint16_t col = (k == 0) ? COL_ACCENT1 : (k % 3 == 0 ? COL_ACCENT4 : COL_ACCENT6);
    lcd.fillCircle(x, y, 3, col);
  }

  // highlight the kernel generator
  const int idx1 = phaseIdx;
  const float c1 = gTrig.cosfIdx(idx1);
  const float s1 = gTrig.sinfIdx(idx1);
  const int hx = cx + (int)lroundf(c1 * radius);
  const int hy = cy + (int)lroundf(s1 * radius);
  lcd.drawLine(cx, cy, hx, hy, COL_ACCENT5);
}

static void drawRigidityTower(int x, int y, int w, int h, int levels, float bend) {
  lcd.drawRect(x, y, w, h, COL_FRAME);
  const int top = y + 10;
  const int left = x + 8;
  const int step = (h - 24) / levels;
  const int mid = x + w / 2;

  for (int i = 0; i < levels; ++i) {
    const int yy = top + i * step;
    const float t = (float)i / (float)(levels - 1);
    const int dx = (int)lroundf(sinf(t * PI) * bend);
    lcd.drawRoundRect(left + dx, yy, w - 16 - 2 * dx, step - 6, 4, (i == 0) ? COL_ACCENT1 : (i == levels - 1 ? COL_ACCENT2 : COL_SUBTLE));
    lcd.setCursor(left + dx + 5, yy + 3);
    lcd.printf("u_%d", i);
    if (i < levels - 1) {
      const int x1 = mid + dx;
      const int y1 = yy + step - 6;
      const int x2 = mid + (int)lroundf(sinf(((float)(i + 1) / (levels - 1)) * PI) * bend);
      const int y2 = yy + step + 4;
      lcd.drawLine(x1, y1, x2, y2, COL_ACCENT4);
      lcd.drawTriangle(x2 - 3, y2 - 4, x2 + 3, y2 - 4, x2, y2 + 2, COL_ACCENT4);
    }
  }

  lcd.setCursor(x + 5, y + h - 12);
  lcd.print("successive nilpotent lifts");
}

static void drawConjugationIcon(int x, int y, int w, int h, float deltaDeg) {
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
  const int cx = x + w / 2;
  const int cy = y + h / 2;
  const int r = (w < h ? w : h) / 3;

  // base object
  lcd.drawCircle(cx, cy, r, COL_ACCENT6);
  lcd.drawCircle(cx, cy, r - 6, COL_SUBTLE);

  // near-identity conjugator as tiny rotation arrow
  const float a = deltaDeg * 0.5f * DEG_TO_RAD;
  const int ax = cx + (int)lroundf(cosf(a) * r);
  const int ay = cy + (int)lroundf(sinf(a) * r);
  lcd.drawLine(cx, cy, ax, ay, COL_ACCENT5);
  lcd.fillCircle(ax, ay, 3, COL_ACCENT5);

  lcd.setCursor(x + 6, y + h - 12);
  lcd.printf("g ≡ e (mod J)");
}

// -----------------------------------------------------------------------------
// Parser for serial commands
// -----------------------------------------------------------------------------

static float gAngleDeg = 24.0f;
static float gZoom = 1.0f;
static int gScene = 0;
static int gKernelN = 12;
static bool gAutoAnimate = true;

static char gCmdBuf[64];
static size_t gCmdLen = 0;

static void printHelp() {
  Serial.println();
  Serial.println(F("Commands:"));
  Serial.println(F("  a <deg>      set rotation angle"));
  Serial.println(F("  z <zoom>     set zoom, e.g. 1.0"));
  Serial.println(F("  scene <0-2>  select scene"));
  Serial.println(F("  n <int>      set torsion kernel order"));
  Serial.println(F("  auto <0|1>   disable/enable auto animation"));
  Serial.println(F("Examples: a 45, z 1.4, scene 1, n 24"));
}

static void handleCommand(const char *line) {
  while (*line == ' ' || *line == '\t') ++line;
  if (*line == '\0') return;

  if (!strncmp(line, "a ", 2) || !strncmp(line, "angle ", 6)) {
    gAngleDeg = atof(strchr(line, ' ') + 1);
  } else if (!strncmp(line, "z ", 2) || !strncmp(line, "zoom ", 5)) {
    gZoom = max(0.2f, atof(strchr(line, ' ') + 1));
  } else if (!strncmp(line, "scene ", 6)) {
    gScene = atoi(strchr(line, ' ') + 1);
    gScene = (gScene % 3 + 3) % 3;
  } else if (!strncmp(line, "n ", 2)) {
    gKernelN = atoi(strchr(line, ' ') + 1);
    if (gKernelN < 1) gKernelN = 1;
  } else if (!strncmp(line, "auto ", 5)) {
    gAutoAnimate = atoi(strchr(line, ' ') + 1) != 0;
  } else if (!strcmp(line, "help")) {
    printHelp();
  } else {
    Serial.print(F("Unknown: "));
    Serial.println(line);
  }
}

// -----------------------------------------------------------------------------
// Scenes
// -----------------------------------------------------------------------------

static void sceneWarp() {
  drawPanelFrame(4, 20, 90, 150, "source chart");
  drawPanelFrame(104, 20, 148, 150, "LUT warp");
  drawPanelFrame(258, 20, 58, 150, "data");

  // source preview
  gSourceSprite.pushSprite(SRC_PANEL_X, SRC_PANEL_Y);

  // transformed object
  const int srcCx = SRC_W / 2;
  const int srcCy = SRC_H / 2;
  const float srcRadius = 24.0f;

  gWarpSprite.fillSprite(rgb565(4, 7, 12));
  renderPolarWarp(gAngleDeg, gZoom, srcCx, srcCy, srcRadius);

  // frame and overlay in the warp sprite
  gWarpSprite.drawCircle(WARP_W / 2, WARP_H / 2, 48, COL_FRAME);
  gWarpSprite.drawCircle(WARP_W / 2, WARP_H / 2, 49, COL_SUBTLE);
  gWarpSprite.drawLine(WARP_W / 2 - 50, WARP_H / 2, WARP_W / 2 + 50, WARP_H / 2, COL_SUBTLE);
  gWarpSprite.drawLine(WARP_W / 2, WARP_H / 2 - 50, WARP_W / 2, WARP_H / 2 + 50, COL_SUBTLE);
  gWarpSprite.drawString("rotation + zoom", 22, 4);

  gWarpSprite.pushSprite(WARP_PANEL_X, WARP_PANEL_Y);

  // right mini chart showing chart basis and character lattice
  drawCharacterLatticeMini(268, 34, 44, 64, gAngleDeg);
  lcd.setCursor(264, 104);
  lcd.printf("T = (Gm)^2");
  lcd.setCursor(264, 116);
  lcd.print("char basis");
  lcd.setCursor(264, 128);
  lcd.print("local chart");
  lcd.setCursor(264, 140);
  lcd.print("diagonal");
}

static void sceneTorsion() {
  drawPanelFrame(10, 16, 300, 192, "torsion kernels nG");
  const int cx = 110;
  const int cy = 110;
  const int radius = 58;

  lcd.setCursor(180, 34);
  lcd.print("multiplication-by-n");
  lcd.setCursor(180, 46);
  lcd.print("ker([n]) = nG");
  lcd.setCursor(180, 58);
  lcd.printf("n = %d", gKernelN);

  drawTorsionCircle(cx, cy, radius, gKernelN, gAngleDeg);

  // a very explicit dense-sampling strip
  for (int n = 2; n <= 10; ++n) {
    const int y = 170;
    const int x0 = 175 + (n - 2) * 12;
    lcd.drawCircle(x0, y, 4, COL_SUBTLE);
    const int idx = (gTrig.idxFromDeg(gAngleDeg) + (gTrig.idxFromDeg(gAngleDeg) * 0 + (ANGLE_STEPS / n))) % ANGLE_STEPS;
    const float c = gTrig.cosfIdx(idx);
    const float s = gTrig.sinfIdx(idx);
    lcd.fillCircle(x0 + (int)lroundf(c * 2.0f), y + (int)lroundf(s * 2.0f), 2, (n == gKernelN) ? COL_ACCENT1 : COL_ACCENT6);
  }

  lcd.setCursor(180, 150);
  lcd.print("family (nG) becomes finer");
  lcd.setCursor(180, 162);
  lcd.print("as n grows");
}

static void sceneRigidity() {
  drawPanelFrame(10, 16, 148, 192, "nilpotent lifts");
  drawPanelFrame(164, 16, 146, 192, "conjugacy");
  drawRigidityTower(18, 28, 132, 168, 5, 10.0f + 4.0f * sinf(millis() * 0.0013f));
  drawConjugationIcon(170, 28, 134, 72, 18.0f + 8.0f * sinf(millis() * 0.0011f));

  lcd.setCursor(174, 112);
  lcd.print("if two lifts agree");
  lcd.setCursor(174, 124);
  lcd.print("mod J, a conjugator");
  lcd.setCursor(174, 136);
  lcd.print("exists and is");
  lcd.setCursor(174, 148);
  lcd.print("near the identity");

  // tiny deformation field visualization
  const int x0 = 180;
  const int y0 = 170;
  for (int i = 0; i < 5; ++i) {
    const int x = x0 + i * 20;
    const int y = y0 + (int)lroundf(sinf((float)i * 0.8f + millis() * 0.001f) * 6.0f);
    lcd.fillCircle(x, y, 3, (i == 0) ? COL_ACCENT1 : COL_ACCENT4);
    if (i) lcd.drawLine(x - 20, y0 + (int)lroundf(sinf((float)(i - 1) * 0.8f + millis() * 0.001f) * 6.0f), x, y, COL_SUBTLE);
  }
}

// -----------------------------------------------------------------------------
// Setup / loop
// -----------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  delay(200);
  printHelp();

  lcd.init();
  lcd.setRotation(1);
  lcd.setBrightness(200);
  lcd.fillScreen(COL_BG);
  lcd.setTextColor(COL_TEXT, COL_BG);
  lcd.setTextDatum(TL_DATUM);
  lcd.setTextWrap(false);
  lcd.setTextSize(1);

  gTrig.build();
  buildPolarLUT();
  buildSourceObject();

  gWarpSprite.setColorDepth(16);
  gWarpSprite.createSprite(WARP_W, WARP_H);
  gWarpSprite.setTextColor(COL_TEXT, 0);
  gWarpSprite.setTextDatum(TL_DATUM);
  gWarpSprite.setTextWrap(false);
  gWarpSprite.setTextSize(1);

  lcd.drawString("SGA3 IX: multiplicative type homomorphisms", 8, 2);
}

static void pollSerial() {
  while (Serial.available() > 0) {
    const char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      gCmdBuf[gCmdLen] = '\0';
      handleCommand(gCmdBuf);
      gCmdLen = 0;
    } else if (gCmdLen + 1 < sizeof(gCmdBuf)) {
      gCmdBuf[gCmdLen++] = ch;
    }
  }
}

void loop() {
  pollSerial();

  if (gAutoAnimate) {
    gAngleDeg = fmodf((float)millis() * 0.020f, 360.0f);
    gZoom = 1.0f + 0.35f * sinf((float)millis() * 0.0012f);
    gKernelN = 2 + (int)((millis() / 1200UL) % 48UL);
    gScene = (int)((millis() / 9000UL) % 3UL);
  }

  lcd.startWrite();
  lcd.fillScreen(COL_BG);

  // Header
  lcd.setTextColor(COL_TEXT, COL_BG);
  lcd.setTextDatum(TL_DATUM);
  lcd.drawString("SGA3 Exposé IX / Homomorphismes", 8, 2);

  switch (gScene) {
    case 0: sceneWarp(); break;
    case 1: sceneTorsion(); break;
    case 2: sceneRigidity(); break;
    default: sceneWarp(); break;
  }

  drawFooter(gAngleDeg, gZoom, gScene, gKernelN);
  lcd.endWrite();

  delay(16);
}
