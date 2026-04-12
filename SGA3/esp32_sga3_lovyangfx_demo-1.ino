/*
  ESP32 + LovyanGFX + ST7789 320x240
  SGA3 Exposé IX visualization demo (rigorously labeled)

  Purpose:
  - Show the paper's actual notions in a UI that keeps the formal names visible.
  - Map each notion to a computational mechanism used by the sketch.
  - Avoid loose metaphors: every label is tied to a theorem/definition in the paper.

  Scenes:
  0) Diagonalizable / multiplicative type / local chart / base change.
  1) [n] : G -> G, ker([n]) = nG, and schematic density of the family (nG).
  2) Nilpotent thickening J, lifting, and conjugacy modulo J.
  3) Subgroups, quotients, and extensions: exact-sequence style factorization.

  Serial commands:
    help
    scene <0-3>
    a <deg>      rotation angle
    z <zoom>     zoom factor (>0)
    n <int>      torsion order / kernel size in scene 1
    auto <0|1>

  Notes:
  - The display code uses LovyanGFX's standard LGFX_Device + Panel_ST7789 pattern.
  - The visuals illustrate the paper's structure; they are not proofs.
*/

#include <Arduino.h>
#include <LovyanGFX.hpp>
#include <math.h>

// -----------------------------------------------------------------------------
// Display configuration
// -----------------------------------------------------------------------------

class LGFX : public lgfx::LGFX_Device {
  lgfx::Panel_ST7789 _panel;
  lgfx::Bus_SPI _bus;
  lgfx::Light_PWM _light;

public:
  LGFX(void) {
    {
      auto cfg = _bus.config();
      cfg.spi_host   = SPI2_HOST;
      cfg.spi_mode   = 0;
      cfg.freq_write = 40000000;
      cfg.freq_read  = 16000000;
      cfg.spi_3wire  = false;
      cfg.use_lock   = true;
      cfg.dma_channel = SPI_DMA_CH_AUTO;
      cfg.pin_sclk   = 18;
      cfg.pin_mosi   = 23;
      cfg.pin_miso   = -1;
      cfg.pin_dc     = 2;
      _bus.config(cfg);
      _panel.setBus(&_bus);
    }

    {
      auto cfg = _panel.config();
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
      _panel.config(cfg);
    }

    {
      auto cfg = _light.config();
      cfg.pin_bl = 15;
      cfg.invert = false;
      cfg.freq = 5000;
      cfg.pwm_channel = 7;
      _light.config(cfg);
      _panel.light(&_light);
    }

    setPanel(&_panel);
  }
};

static LGFX lcd;

// -----------------------------------------------------------------------------
// Basic helpers
// -----------------------------------------------------------------------------

static constexpr int TFT_W = 320;
static constexpr int TFT_H = 240;

static constexpr int ANGLE_STEPS = 3600; // 0.1 degree resolution
static constexpr float ANGLE_TO_RAD = 0.0017453292519943296f; // PI / 1800

static constexpr int SRC_W = 64;
static constexpr int SRC_H = 64;
static constexpr int WARP_W = 132;
static constexpr int WARP_H = 132;

static constexpr uint16_t COL_BG      = 0x0000;
static constexpr uint16_t COL_PANEL   = 0x1082;
static constexpr uint16_t COL_FRAME   = 0x7BEF;
static constexpr uint16_t COL_TEXT    = 0xFFFF;
static constexpr uint16_t COL_SUBTLE  = 0x39E7;
static constexpr uint16_t COL_ACCENT1 = 0xF800;
static constexpr uint16_t COL_ACCENT2 = 0x07E0;
static constexpr uint16_t COL_ACCENT3 = 0x001F;
static constexpr uint16_t COL_ACCENT4 = 0xFFE0;
static constexpr uint16_t COL_ACCENT5 = 0xF81F;
static constexpr uint16_t COL_ACCENT6 = 0x07FF;

static inline uint16_t rgb565(uint8_t r, uint8_t g, uint8_t b) {
  return (uint16_t)(((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3));
}

static inline void unpack565(uint16_t c, uint8_t &r, uint8_t &g, uint8_t &b) {
  r = (uint8_t)(((c >> 11) & 0x1F) * 255 / 31);
  g = (uint8_t)(((c >> 5) & 0x3F) * 255 / 63);
  b = (uint8_t)((c & 0x1F) * 255 / 31);
}

static inline uint16_t lerp565(uint16_t a, uint16_t b, float t) {
  uint8_t ar, ag, ab, br, bg, bb;
  unpack565(a, ar, ag, ab);
  unpack565(b, br, bg, bb);
  const uint8_t rr = (uint8_t)(ar + (br - ar) * t);
  const uint8_t gg = (uint8_t)(ag + (bg - ag) * t);
  const uint8_t bb2 = (uint8_t)(ab + (bb - ab) * t);
  return rgb565(rr, gg, bb2);
}

static inline uint16_t blend4(uint16_t c00, uint16_t c10, uint16_t c01, uint16_t c11, float fx, float fy) {
  const uint16_t a = lerp565(c00, c10, fx);
  const uint16_t b = lerp565(c01, c11, fx);
  return lerp565(a, b, fy);
}

static inline int clampi(int v, int lo, int hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// -----------------------------------------------------------------------------
// Trigonometric LUT
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

  inline int idxFromDeg(float deg) const {
    int idx = (int)lroundf(deg * 10.0f);
    idx %= ANGLE_STEPS;
    if (idx < 0) idx += ANGLE_STEPS;
    return idx;
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
};

static TrigLUT gTrig;

// -----------------------------------------------------------------------------
// Circular lookup table used by the warp renderer
// -----------------------------------------------------------------------------

struct PolarEntry {
  uint16_t angleIdx;   // angle coordinate in LUT space
  uint16_t radiusQ15;  // normalized radius in [0, 32767]
  uint8_t inside;      // 1 if inside unit disk
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
      const float ang = atan2f(dy, dx); // [-pi, pi]
      int idx = (int)lroundf((ang + PI) * (float)ANGLE_STEPS / (2.0f * PI));
      idx %= ANGLE_STEPS;
      if (idx < 0) idx += ANGLE_STEPS;

      PolarEntry &e = gPolarLUT[y * WARP_W + x];
      e.inside = (r <= 1.0f) ? 1 : 0;
      e.angleIdx = (uint16_t)idx;
      e.radiusQ15 = (uint16_t)lroundf(clampf(r, 0.0f, 1.0f) * 32767.0f);
    }
  }
}

// -----------------------------------------------------------------------------
// Source object rendered in the local chart
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

  // Coordinate grid.
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      if ((x % 8) == 0 || (y % 8) == 0) {
        gSourceBuf[y * SRC_W + x] = rgb565(18, 26, 40);
      }
    }
  }

  // Unit circle / chart boundary.
  for (int a = 0; a < ANGLE_STEPS; a += 4) {
    const float rad = (float)a * ANGLE_TO_RAD;
    const int x = cx + (int)lroundf(cosf(rad) * 24.0f);
    const int y = cy + (int)lroundf(sinf(rad) * 24.0f);
    setSrcPixel(x, y, COL_FRAME);
  }

  // Disk fill with a mild radial gradient.
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      const float dx = (x - cx) / 24.0f;
      const float dy = (y - cy) / 24.0f;
      const float r = sqrtf(dx * dx + dy * dy);
      if (r <= 1.0f) {
        const uint8_t v = (uint8_t)(56 + (1.0f - r) * 128.0f);
        setSrcPixel(x, y, rgb565(v, (uint8_t)(v * 0.45f), (uint8_t)(255 - v * 0.35f)));
      }
    }
  }

  // Cross axes.
  for (int i = -20; i <= 20; ++i) {
    setSrcPixel(cx + i, cy, rgb565(220, 220, 220));
    setSrcPixel(cx, cy + i, rgb565(220, 220, 220));
  }

  // A directional vector that makes rotation visible.
  for (int i = -18; i <= 18; ++i) {
    setSrcPixel(cx + i, cy - i / 2, COL_ACCENT4);
    if ((i & 1) == 0) setSrcPixel(cx + i, cy - i / 2 + 1, COL_ACCENT4);
  }

  // Marker A.
  for (int yy = -3; yy <= 3; ++yy) {
    for (int xx = -3; xx <= 3; ++xx) {
      const int px = cx + 14 + xx;
      const int py = cy - 9 + yy;
      if ((xx * xx + yy * yy) <= 9) setSrcPixel(px, py, rgb565(255, 80, 20));
    }
  }

  // Marker B.
  for (int yy = -2; yy <= 2; ++yy) {
    for (int xx = -2; xx <= 2; ++xx) {
      const int px = cx - 11 + xx;
      const int py = cy + 12 + yy;
      if ((xx * xx + yy * yy) <= 4) setSrcPixel(px, py, COL_ACCENT2);
    }
  }

  // Write into a preview sprite.
  gSourceSprite.setColorDepth(16);
  gSourceSprite.createSprite(SRC_W, SRC_H);
  for (int y = 0; y < SRC_H; ++y) {
    for (int x = 0; x < SRC_W; ++x) {
      gSourceSprite.drawPixel(x, y, gSourceBuf[y * SRC_W + x]);
    }
  }
}

// -----------------------------------------------------------------------------
// Warp renderer: circular LUT + rotation + zoom
// -----------------------------------------------------------------------------

static void renderPolarWarp(float angleDeg, float zoom, int srcCx, int srcCy, float srcRadius) {
  const uint16_t bg = rgb565(4, 7, 12);
  const int rotIdx = gTrig.idxFromDeg(angleDeg);
  const float invZoom = 1.0f / clampf(zoom, 0.15f, 8.0f);

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
      const float radius = ((float)e.radiusQ15 / 32767.0f) * srcRadius * invZoom;
      const float sx = (float)srcCx + c * radius;
      const float sy = (float)srcCy + s * radius;
      gWarpSprite.drawPixel(x, y, sampleSource(sx, sy, bg));
    }
  }
}

// -----------------------------------------------------------------------------
// Layout helpers
// -----------------------------------------------------------------------------

static void fillPanel(int x, int y, int w, int h) {
  lcd.fillRoundRect(x, y, w, h, 6, COL_PANEL);
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
}

static void drawTitleBar(const char *title, const char *subtitle) {
  lcd.fillRect(0, 0, TFT_W, 18, COL_SUBTLE);
  lcd.setTextColor(COL_TEXT, COL_SUBTLE);
  lcd.setCursor(6, 2);
  lcd.print(title);
  if (subtitle && subtitle[0]) {
    lcd.setCursor(170, 2);
    lcd.print(subtitle);
  }
}

static void drawFooter(float angleDeg, float zoom, int scene, int kernelN) {
  lcd.fillRect(0, 208, 320, 32, COL_SUBTLE);
  lcd.setTextColor(COL_TEXT, COL_SUBTLE);
  lcd.setCursor(6, 212);
  lcd.printf("scene=%d  angle=%.1f deg  zoom=%.3f  n=%d", scene, angleDeg, zoom, kernelN);
  lcd.setCursor(6, 224);
  lcd.print("terms: diag | type | nG | J | conj.");
}

static void drawTermBox(int x, int y, int w, int h, const char *term, const char *meaning, uint16_t accent) {
  lcd.drawRoundRect(x, y, w, h, 5, accent);
  lcd.fillRect(x + 1, y + 1, w - 2, 11, COL_SUBTLE);
  lcd.setTextColor(COL_TEXT, COL_SUBTLE);
  lcd.setCursor(x + 4, y + 2);
  lcd.print(term);
  lcd.setTextColor(COL_TEXT, COL_PANEL);
  lcd.setCursor(x + 4, y + 14);
  lcd.print(meaning);
}

static void drawArrow(int x1, int y1, int x2, int y2, uint16_t col) {
  lcd.drawLine(x1, y1, x2, y2, col);
  const float dx = (float)(x2 - x1);
  const float dy = (float)(y2 - y1);
  const float len = sqrtf(dx * dx + dy * dy);
  if (len < 1.0f) return;
  const float ux = dx / len;
  const float uy = dy / len;
  const float px = -uy;
  const float py = ux;
  const int ax = x2 - (int)lroundf(ux * 6.0f);
  const int ay = y2 - (int)lroundf(uy * 6.0f);
  lcd.drawLine(x2, y2, ax + (int)lroundf(px * 3.0f), ay + (int)lroundf(py * 3.0f), col);
  lcd.drawLine(x2, y2, ax - (int)lroundf(px * 3.0f), ay - (int)lroundf(py * 3.0f), col);
}

// -----------------------------------------------------------------------------
// Scene 0: diagonalizable / multiplicative type / local chart / base change
// -----------------------------------------------------------------------------

static void drawBasisChart(int x, int y, int w, int h, float angleDeg, const char *caption) {
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
  const int cx = x + w / 2;
  const int cy = y + h / 2;
  const int rotIdx = gTrig.idxFromDeg(angleDeg);
  const float c = gTrig.cosfIdx(rotIdx);
  const float s = gTrig.sinfIdx(rotIdx);

  for (int i = -3; i <= 3; ++i) {
    const int gx = cx + i * 12;
    lcd.drawLine(gx, y + 4, gx, y + h - 5, COL_SUBTLE);
    const int gy = cy + i * 12;
    lcd.drawLine(x + 4, gy, x + w - 5, gy, COL_SUBTLE);
  }

  // rotated basis vectors in the chart
  const int vx = (int)lroundf(c * 22.0f);
  const int vy = (int)lroundf(s * 22.0f);
  lcd.drawLine(cx, cy, cx + vx, cy + vy, COL_ACCENT1);
  lcd.drawLine(cx, cy, cx - vy, cy + vx, COL_ACCENT2);
  lcd.fillCircle(cx + vx, cy + vy, 3, COL_ACCENT1);
  lcd.fillCircle(cx - vy, cy + vx, 3, COL_ACCENT2);

  for (int m = -2; m <= 2; ++m) {
    for (int n = -2; n <= 2; ++n) {
      const int px = cx + m * 12 + (n * vx) / 5;
      const int py = cy + n * 12 + (m * vy) / 5;
      if (px > x + 2 && px < x + w - 2 && py > y + 2 && py < y + h - 2) {
        lcd.fillCircle(px, py, 2, ((m + n) & 1) ? COL_ACCENT4 : COL_ACCENT6);
      }
    }
  }

  lcd.setTextColor(COL_TEXT, COL_PANEL);
  lcd.setCursor(x + 4, y + h - 12);
  lcd.print(caption);
}

static void sceneDiagonalizable() {
  fillPanel(4, 20, 154, 150);
  fillPanel(164, 20, 152, 150);

  // Left: raw chart/source object.
  gSourceSprite.pushSprite(12, 30);
  lcd.setTextColor(COL_TEXT, COL_PANEL);
  lcd.setCursor(12, 104);
  lcd.print("chart on S");
  lcd.setCursor(12, 116);
  lcd.print("local object");
  lcd.setCursor(12, 128);
  lcd.print("before change");

  // Right: rotated / zoomed local chart.
  gWarpSprite.fillSprite(rgb565(4, 7, 12));
  renderPolarWarp(gAngleDeg, gZoom, SRC_W / 2, SRC_H / 2, 24.0f);
  gWarpSprite.drawCircle(WARP_W / 2, WARP_H / 2, 48, COL_FRAME);
  gWarpSprite.drawLine(WARP_W / 2 - 50, WARP_H / 2, WARP_W / 2 + 50, WARP_H / 2, COL_SUBTLE);
  gWarpSprite.drawLine(WARP_W / 2, WARP_H / 2 - 50, WARP_W / 2, WARP_H / 2 + 50, COL_SUBTLE);
  gWarpSprite.drawString("D_S(M) after cover", 6, 4);
  gWarpSprite.pushSprite(170, 30);

  lcd.setCursor(170, 104);
  lcd.print("chart change");
  lcd.setCursor(170, 116);
  lcd.print("rotation = basis change");
  lcd.setCursor(170, 128);
  lcd.print("zoom = scale");

  // Right column: nomenclature cards.
  drawTermBox(248, 30, 66, 32, "diag", "local D(M)", COL_ACCENT1);
  drawTermBox(248, 66, 66, 32, "type", "fiber label", COL_ACCENT2);
  drawTermBox(248, 102, 66, 32, "fpqc", "local cover", COL_ACCENT4);
  drawTermBox(248, 138, 66, 26, "base chg", "G x_S S'", COL_ACCENT6);
}

// -----------------------------------------------------------------------------
// Scene 1: [n], nG, and schematic density
// -----------------------------------------------------------------------------

static void drawKernelRow(int x, int y, int w, int n, float phaseDeg) {
  const int cx = x + 24;
  const int cy = y + 10;
  const int r = 9;

  lcd.drawCircle(cx, cy, r, COL_FRAME);
  lcd.drawLine(cx - r - 4, cy, cx + r + 4, cy, COL_SUBTLE);
  lcd.drawLine(cx, cy - r - 4, cx, cy + r + 4, COL_SUBTLE);

  const int phaseIdx = gTrig.idxFromDeg(phaseDeg);
  for (int k = 0; k < n; ++k) {
    const int idx = (phaseIdx + (k * ANGLE_STEPS) / n) % ANGLE_STEPS;
    const float c = gTrig.cosfIdx(idx);
    const float s = gTrig.sinfIdx(idx);
    const int px = cx + (int)lroundf(c * r);
    const int py = cy + (int)lroundf(s * r);
    lcd.fillCircle(px, py, 2, (k == 0) ? COL_ACCENT1 : (k % 2 ? COL_ACCENT4 : COL_ACCENT6));
  }

  lcd.setCursor(x + 48, y + 4);
  lcd.printf("n=%d", n);
  lcd.setCursor(x + 48, y + 14);
  lcd.print("ker([n]) = nG");
}

static void sceneTorsionDensity() {
  fillPanel(4, 20, 190, 150);
  fillPanel(198, 20, 118, 150);

  lcd.setCursor(12, 26);
  lcd.print("n-action on G");
  lcd.setCursor(12, 38);
  lcd.print("[n] : G -> G");
  lcd.setCursor(12, 50);
  lcd.print("ker([n]) = nG");

  // A vertical family of kernel rows.
  const int ns[] = {2, 3, 4, 5, 6, 8, 12};
  for (int i = 0; i < 7; ++i) {
    drawKernelRow(10, 64 + i * 18, 170, ns[i], gAngleDeg);
  }

  // The theorem statement in paper language.
  drawTermBox(202, 28, 106, 32, "schem. dense", "tests detect all", COL_ACCENT4);
  drawTermBox(202, 66, 106, 32, "reduced torus", "union of nG", COL_ACCENT6);
  drawTermBox(202, 104, 106, 32, "test family", "no vanish", COL_ACCENT2);
  drawTermBox(202, 142, 106, 20, "paper", "Theorem 4.7", COL_ACCENT1);
}

// -----------------------------------------------------------------------------
// Scene 2: nilpotent thickening, lifting, conjugacy
// -----------------------------------------------------------------------------

static void drawLiftTower(int x, int y, int w, int h, int levels, float bend) {
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
  const int top = y + 10;
  const int left = x + 8;
  const int step = (h - 24) / levels;
  const int mid = x + w / 2;

  for (int i = 0; i < levels; ++i) {
    const int yy = top + i * step;
    const float t = (levels <= 1) ? 0.0f : (float)i / (float)(levels - 1);
    const int dx = (int)lroundf(sinf(t * PI) * bend);
    const int bw = w - 16 - 2 * dx;
    lcd.drawRoundRect(left + dx, yy, bw, step - 6, 4, (i == 0) ? COL_ACCENT1 : (i == levels - 1 ? COL_ACCENT2 : COL_SUBTLE));
    lcd.setCursor(left + dx + 5, yy + 3);
    lcd.printf("J^%d", levels - 1 - i);
    if (i < levels - 1) {
      const int x1 = mid + dx;
      const int y1 = yy + step - 6;
      const int x2 = mid + (int)lroundf(sinf(((float)(i + 1) / (levels - 1)) * PI) * bend);
      const int y2 = yy + step + 4;
      drawArrow(x1, y1, x2, y2, COL_ACCENT4);
    }
  }
}

static void drawConjugacyIcon(int x, int y, int w, int h, float deltaDeg) {
  lcd.drawRoundRect(x, y, w, h, 6, COL_FRAME);
  const int cx = x + w / 2;
  const int cy = y + h / 2 - 2;
  const int r = (w < h ? w : h) / 4;

  // Object.
  lcd.drawCircle(cx, cy, r, COL_ACCENT6);
  lcd.drawCircle(cx, cy, r - 6, COL_SUBTLE);

  // Near-identity conjugator.
  const float a = deltaDeg * DEG_TO_RAD * 0.5f;
  const int ax = cx + (int)lroundf(cosf(a) * r);
  const int ay = cy + (int)lroundf(sinf(a) * r);
  drawArrow(cx, cy, ax, ay, COL_ACCENT5);
  lcd.fillCircle(ax, ay, 3, COL_ACCENT5);

  lcd.setCursor(x + 6, h + y - 12);
  lcd.print("g = e mod J");
}

static void sceneLiftingRigidity() {
  fillPanel(4, 20, 144, 150);
  fillPanel(154, 20, 162, 150);

  lcd.setCursor(10, 26);
  lcd.print("J nilpotent");
  lcd.setCursor(10, 38);
  lcd.print("H^2 -> lift");
  lcd.setCursor(10, 50);
  lcd.print("H^1 -> conj.");

  drawLiftTower(10, 64, 128, 96, 5, 10.0f + 4.0f * sinf(millis() * 0.0013f));

  drawConjugacyIcon(160, 26, 150, 68, 18.0f + 8.0f * sinf(millis() * 0.0011f));

  drawTermBox(160, 102, 150, 22, "H^2 = 0", "lift exists", COL_ACCENT1);
  drawTermBox(160, 128, 150, 22, "H^1 = 0", "up to conj.", COL_ACCENT2);
  drawTermBox(160, 154, 150, 16, "nilpotent J", "finite layers", COL_ACCENT4);
}

// -----------------------------------------------------------------------------
// Scene 3: kernel / quotient / extension
// -----------------------------------------------------------------------------

static void sceneExtensions() {
  fillPanel(4, 20, 176, 150);
  fillPanel(186, 20, 130, 150);

  // exact sequence visualization
  const int y = 66;
  const int boxW = 36;
  const int boxH = 26;
  const int xH = 14;
  const int xG = 80;
  const int xK = 148;

  lcd.drawRoundRect(xH, y, boxW, boxH, 4, COL_ACCENT1);
  lcd.drawRoundRect(xG, y, boxW, boxH, 4, COL_ACCENT4);
  lcd.drawRoundRect(xK, y, boxW, boxH, 4, COL_ACCENT2);
  lcd.setCursor(xH + 12, y + 8); lcd.print("H");
  lcd.setCursor(xG + 12, y + 8); lcd.print("G");
  lcd.setCursor(xK + 12, y + 8); lcd.print("K");

  drawArrow(54, y + 13, 80, y + 13, COL_TEXT);
  drawArrow(120, y + 13, 148, y + 13, COL_TEXT);

  lcd.setCursor(18, 32);
  lcd.print("1 -> H -> G -> K -> 1");
  lcd.setCursor(18, 44);
  lcd.print("kernel / quotient / ext.");
  lcd.setCursor(18, 56);
  lcd.print("stay in the class");

  // Factor through the quotient.
  lcd.drawRoundRect(30, 108, 128, 40, 6, COL_SUBTLE);
  lcd.setCursor(38, 116);
  lcd.print("factorization:");
  lcd.setCursor(38, 128);
  lcd.print("H -> G -> G/H");
  drawArrow(128, 128, 148, 128, COL_ACCENT5);
  lcd.fillCircle(150, 128, 3, COL_ACCENT5);

  drawTermBox(190, 28, 114, 30, "kernel", "Ker(u) type M", COL_ACCENT1);
  drawTermBox(190, 62, 114, 30, "quotient", "G/H stays type M", COL_ACCENT2);
  drawTermBox(190, 96, 114, 30, "extension", "glue H and K", COL_ACCENT4);
  drawTermBox(190, 130, 114, 24, "field case", "section 8", COL_ACCENT6);
}

// -----------------------------------------------------------------------------
// Serial command handling
// -----------------------------------------------------------------------------

static float gAngleDeg = 24.0f;
static float gZoom = 1.0f;
static int gScene = 0;
static int gKernelN = 12;
static bool gAutoAnimate = true;

static char gCmdBuf[80];
static size_t gCmdLen = 0;

static void printHelp() {
  Serial.println();
  Serial.println(F("Commands:"));
  Serial.println(F("  help        show this message"));
  Serial.println(F("  scene N     0..3"));
  Serial.println(F("  a DEG       rotation angle"));
  Serial.println(F("  z ZOOM      zoom factor"));
  Serial.println(F("  n N         torsion/kernel order in scene 1"));
  Serial.println(F("  auto 0|1    disable/enable animation"));
  Serial.println(F("Examples: scene 2, a 45, z 1.35, n 24, auto 0"));
}

static void handleCommand(const char *line) {
  while (*line == ' ' || *line == '\t') ++line;
  if (*line == '\0') return;

  const char *sp = strchr(line, ' ');
  if (!strncmp(line, "a ", 2) || !strncmp(line, "angle ", 6)) {
    gAngleDeg = atof(sp ? sp + 1 : line + 1);
  } else if (!strncmp(line, "z ", 2) || !strncmp(line, "zoom ", 5)) {
    gZoom = clampf(atof(sp ? sp + 1 : line + 1), 0.15f, 8.0f);
  } else if (!strncmp(line, "scene ", 6)) {
    gScene = atoi(sp ? sp + 1 : line + 5);
    gScene = (gScene % 4 + 4) % 4;
  } else if (!strncmp(line, "n ", 2)) {
    gKernelN = clampi(atoi(sp ? sp + 1 : line + 1), 1, 60);
  } else if (!strncmp(line, "auto ", 5)) {
    gAutoAnimate = atoi(sp ? sp + 1 : line + 4) != 0;
  } else if (!strcmp(line, "help")) {
    printHelp();
  } else {
    Serial.print(F("Unknown command: "));
    Serial.println(line);
  }
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
  lcd.setTextDatum(TL_DATUM);
  lcd.setTextWrap(false);
  lcd.setTextSize(1);
  lcd.setTextColor(COL_TEXT, COL_BG);

  gTrig.build();
  buildPolarLUT();
  buildSourceObject();

  gWarpSprite.setColorDepth(16);
  gWarpSprite.createSprite(WARP_W, WARP_H);
  gWarpSprite.setTextDatum(TL_DATUM);
  gWarpSprite.setTextWrap(false);
  gWarpSprite.setTextSize(1);
  gWarpSprite.setTextColor(COL_TEXT, 0);

  drawTitleBar("SGA3 Expose IX / homomorphismes", "formal terms -> code");
}

void loop() {
  pollSerial();

  if (gAutoAnimate) {
    gAngleDeg = fmodf((float)millis() * 0.020f, 360.0f);
    gZoom = 1.0f + 0.35f * sinf((float)millis() * 0.0012f);
    gKernelN = 2 + (int)((millis() / 1200UL) % 48UL);
    gScene = (int)((millis() / 9000UL) % 4UL);
  }

  lcd.startWrite();
  lcd.fillScreen(COL_BG);
  drawTitleBar("SGA3 Expose IX / homomorphismes", "formal terms -> code");

  switch (gScene) {
    case 0: sceneDiagonalizable(); break;
    case 1: sceneTorsionDensity(); break;
    case 2: sceneLiftingRigidity(); break;
    case 3: sceneExtensions(); break;
    default: sceneDiagonalizable(); break;
  }

  drawFooter(gAngleDeg, gZoom, gScene, gKernelN);
  lcd.endWrite();

  delay(16);
}

