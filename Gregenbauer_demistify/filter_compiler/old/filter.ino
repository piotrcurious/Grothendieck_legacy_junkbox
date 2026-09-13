#include "gegenbauer_coeffs.h"
#include <string.h>

// Sampling rate matches mock environment (2000 Hz)
#define SAMPLING_RATE 2000

const uint8_t LP_PIN = 9, HP_PIN = 10;
#define B 128
#define MASK (B - 1)
int16_t x[B];
uint8_t i = 0;
uint32_t last_micros = 0;

void setup() {
  pinMode(LP_PIN, OUTPUT);
  pinMode(HP_PIN, OUTPUT);
  analogReadResolution(8);
  memset(x, 0, sizeof(x));
}

void loop() {
  uint32_t now = micros();
  if (now - last_micros < 1000000 / SAMPLING_RATE) return;
  last_micros = now;

  x[i] = analogRead(A1);

  int32_t y1 = 0, y2 = 0;
  for (uint8_t j = 0; j < GEG_N; j++) {
    int32_t in = (int32_t)x[(i - j) & MASK] - 128;
    y1 += (int32_t)(int16_t)pgm_read_word(&h_geg_fixed[j]) * in;
    y2 += (int32_t)(int16_t)pgm_read_word(&g_geg_fixed[j]) * in;
  }

  // Scale back to 8-bit PWM (0..255) from 16-bit fixed point coefficients (scaled by 2^15 = 32768)
  int16_t out_lp = ((y1 + 16384L) >> 15) + 128;
  int16_t out_hp = ((y2 + 16384L) >> 15) + 128;

  analogWrite(LP_PIN, constrain(out_lp, 0, 255));
  analogWrite(HP_PIN, constrain(out_hp, 0, 255));

  i = (i + 1) & MASK;
}
