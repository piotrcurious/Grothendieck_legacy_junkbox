#include <TFT_eSPI.h>  // Include the TFT_eSPI library

TFT_eSPI tft = TFT_eSPI();  // Create TFT display object

// Function to map mathematical values to screen coordinates
int mapToScreenX(float x) {
  return int(x * 50 + 160);  // Scale and shift for screen size
}

int mapToScreenY(float y) {
  return int(-y * 50 + 120);  // Invert y-axis and scale
}

void setup() {
  tft.init();          // Initialize the TFT screen
  tft.setRotation(3);  // Set screen rotation
  tft.fillScreen(TFT_BLACK);  // Clear screen with black background

  tft.setTextColor(TFT_WHITE, TFT_BLACK); // Set text color for display

  // Draw title
  tft.setCursor(10, 10);
  tft.setTextSize(1);
  tft.print("Algebraic Geometry & Cofibration Demo");

  // Draw the axes
  tft.drawLine(0, 120, 320, 120, TFT_WHITE);  // X axis
  tft.drawLine(160, 0, 160, 240, TFT_WHITE);  // Y axis

  // Draw the polynomial curve y = x^2
  tft.setColor(TFT_GREEN);
  for (float x = -3.0; x < 3.0; x += 0.1) {
    float y = x * x;  // Polynomial equation y = x^2
    int screenX = mapToScreenX(x);
    int screenY = mapToScreenY(y);
    tft.drawPixel(screenX, screenY);
  }

  // Demonstrate the concept of a cofibration by drawing a subregion
  tft.setColor(TFT_RED);
  for (float x = -1.0; x < 1.0; x += 0.05) {
    float y = x * x;  // Subspace y = x^2 within [-1,1]
    int screenX = mapToScreenX(x);
    int screenY = mapToScreenY(y);
    tft.drawPixel(screenX, screenY);
  }

  // Add annotations for cofibration and algebraic geometry concepts
  tft.setTextColor(TFT_YELLOW, TFT_BLACK);
  tft.setCursor(10, 200);
  tft.setTextSize(1);
  tft.print("Cofibration region in red. Full curve in green.");
}

void loop() {
  // Nothing to do here for this demo
}
