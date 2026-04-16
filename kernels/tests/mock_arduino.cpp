#include "Arduino.h"

MockSerial Serial;

static int mock_analog_value = 512;
static unsigned long mock_millis = 0;

void delay(unsigned long ms) {
    mock_millis += ms;
}

unsigned long millis() {
    return mock_millis;
}

int analogRead(uint8_t pin) {
    return mock_analog_value;
}

void setAnalogReadValue(int value) {
    mock_analog_value = value;
}

void setMillis(unsigned long ms) {
    mock_millis = ms;
}
