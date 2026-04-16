#include "Arduino.h"
#include <iostream>
#include <vector>

#include "../a/1.ino"

int main() {
    setup();
    for(int i = 0; i < 20; i++) {
        loop();
    }
    return 0;
}
