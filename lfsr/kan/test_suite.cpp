#include <iostream>
#include <cassert>
#include <vector>
#include <string>
#define NO_MAIN
#include "kan_field_geometry.cpp"

void test_poly_math() {
    std::cout << "Testing poly math..." << std::endl;
    assert(degree(0b1011) == 3);
    assert(poly_mod(0b1011, 0b11) == 1); // (x^3+x+1) mod (x+1) -> x^3+x+1 = (x^2+x)(x+1) + 1
    // x^3+x+1 / x+1:
    // x^3+x^2
    // -----
    // x^2+x+1
    // x^2+x
    // -----
    // 1
    assert(poly_mod(0b1011, 0b11) == 1);

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
    assert(is_irreducible(0b1111) == false); // x^3 + x^2 + x + 1 = (x+1)^3 ? No, (x+1)(x^2+1) = (x+1)^3
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
    AlgebraicLFSR m(n, p, 1);
    std::string bits;
    for(int i=0; i<10; ++i) {
        bits.push_back((m.state & 1) ? '1' : '0');
        m.step();
    }

    Observation obs{n, bits};
    auto candidates = direct_recognizer(obs);
    bool found = false;
    for(u64 c : candidates) if(c == p) found = true;
    assert(found);
    std::cout << "Kan inference passed." << std::endl;
}

int main() {
    test_poly_math();
    test_irreducibility();
    test_primitivity();
    test_kan_inference();
    std::cout << "ALL TESTS PASSED" << std::endl;
    return 0;
}
