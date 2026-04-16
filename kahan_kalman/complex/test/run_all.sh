#!/bin/bash
set -e
mkdir -p build
for f in ../*.ino; do
    name=$(basename "$f" .ino)
    echo "Building test for $name..."
    cat > build/test_$name.cpp <<INNER_EOF
#include "Arduino.h"
#include "KahanMatrix.h"
#include "$f"
int main() {
    setup();
    for(int i=0; i<10; ++i) loop();
    return 0;
}
INNER_EOF
    g++ -o build/test_$name_bin build/test_$name.cpp mock_arduino.cpp -I. -I..
    echo "Running test for $name..."
    ./build/test_$name_bin | tail -n 5
done
