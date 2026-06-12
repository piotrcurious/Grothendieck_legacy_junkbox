#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#ifndef NO_MAIN
#define NO_MAIN
#endif
#include "kan_field_geometry.cpp"

void test_poly_math() {
    std::cout << "Testing poly math..." << std::endl;
    assert(degree(0b1011) == 3);
    assert(poly_mod(0b1011, 0b11) == 1); // (x^3+x+1) mod (x+1) -> x^3+x+1 = (x^2+x)(x+1) + 1
    assert(poly_gcd(0b1011, 0b11) == 1);
    std::cout << "Poly math passed." << std::endl;
}

void test_irreducibility() {
    std::cout << "Testing irreducibility..." << std::endl;
    assert(is_irreducible(0b11) == true);   // x + 1
    assert(is_irreducible(0b111) == true);  // x^2 + x + 1
    assert(is_irreducible(0b1011) == true); // x^3 + x + 1
    assert(is_irreducible(0b1101) == true); // x^3 + x^2 + 1
    assert(is_irreducible(0b10011) == true); // x^4 + x + 1
    assert(is_irreducible(0b1111) == false); // x^3 + x^2 + x + 1
    std::cout << "Irreducibility passed." << std::endl;
}

void test_primitivity() {
    std::cout << "Testing primitivity..." << std::endl;
    assert(is_primitive(0b1011) == true);
    assert(is_primitive(0b10011) == true);
    std::cout << "Primitivity passed." << std::endl;
}

void test_kan_inference() {
    std::cout << "Testing Kan inference..." << std::endl;
    unsigned n = 4;
    u64 p = 0b10011; // x^4 + x + 1
    u64 s = 1;
    std::string bits;
    for(int i=0; i<10; ++i) {
        bits.push_back((s & 1) ? '1' : '0');
        s = poly_mul_mod(s, 2, p);
    }

    Observation obs{n, bits};
    auto candidates = direct_recognizer(obs);
    bool found = false;
    for(u64 c : candidates) if(c == p) found = true;
    assert(found);
    std::cout << "Kan inference passed." << std::endl;
}

void test_matrix() {
    std::cout << "Testing Matrix transitions..." << std::endl;
    u64 p = 0b1000011; // x^6 + x + 1
    unsigned n = 6;
    GaloisMatrix M = GaloisMatrix::companion(p);
    u64 s = 1;
    for(int i=0; i<100; ++i) {
        u64 s_next_poly = poly_mul_mod(s, 2, p);
        u64 s_next_mat = M.apply(s);
        assert(s_next_poly == s_next_mat);
        s = s_next_poly;
    }
    std::cout << "Matrix transitions passed." << std::endl;
}

void test_decimation() {
    std::cout << "Testing Decimation..." << std::endl;
    u64 p = 0b1000011;
    u64 xk = poly_pow_mod(2, 3, p); // Decimate by 3
    u64 s = 1;
    for(int i=0; i<10; ++i) {
        u64 s3 = s;
        for(int j=0; j<3; ++j) s3 = poly_mul_mod(s3, 2, p);
        u64 s_dec = poly_mul_mod(s, xk, p);
        assert(s3 == s_dec);
        s = s_dec;
    }
    std::cout << "Decimation passed." << std::endl;
}

int main() {
    test_poly_math();
    test_irreducibility();
    test_primitivity();
    test_kan_inference();
    test_matrix();
    test_decimation();
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}
