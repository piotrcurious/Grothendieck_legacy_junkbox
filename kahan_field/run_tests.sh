#!/bin/bash
set -e

# Change to the directory where the script is located
cd "$(dirname "$0")"

echo "Compiling tests..."
g++ -O3 tests/test_gfp.cpp -o tests/test_gfp
g++ -O3 tests/test_tensor.cpp -o tests/test_tensor
g++ -O3 tests/test_kahan.cpp -o tests/test_kahan
g++ -O3 tests/test_advanced.cpp -o tests/test_advanced

echo "Running test_gfp..."
./tests/test_gfp
echo "Running test_tensor..."
./tests/test_tensor
echo "Running test_kahan..."
./tests/test_kahan
echo "Running test_advanced..."
./tests/test_advanced

echo "All tests passed!"
