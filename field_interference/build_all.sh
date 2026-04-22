#!/bin/bash
# build_all.sh: Build all C++ visualizers in field_interference

set -e

echo "--- Building field_interference C++ suite ---"

LIBS="-lfltk -lfltk_gl -lGL -lGLU -lm"
CXXFLAGS="-std=c++17 -O3"

# 1. Educational Demos
echo "[1/3] Building educational demos in demo1/..."
cd field_interference/demo1
g++ $CXXFLAGS demo01.cpp -o demo01 $LIBS
g++ $CXXFLAGS demo02.cpp -o demo02 $LIBS
g++ $CXXFLAGS demo03.cpp -o demo03 $LIBS
cd ../..

# 2. Advanced Explorers
echo "[2/3] Building advanced explorers in interference/..."
DIRS=(demo05 demo06 demo07 demo08 demo09 demo0a demo0b demo0c)
for d in "${DIRS[@]}"; do
    echo "  -> Building $d..."
    cd field_interference/interference/$d
    g++ $CXXFLAGS interference.cpp -o interference $LIBS
    cd ../../..
done

# 3. Verification
echo "[3/3] Verification..."
find field_interference -name "demo01" -o -name "interference" | xargs ls -lh

echo "--- Done. All visualizers built successfully. ---"
