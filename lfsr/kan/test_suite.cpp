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
    assert(poly_mod(0b1011, 0b11) == 1);
    assert(poly_gcd(0b1011, 0b11) == 1);

    // Check degree up to 63
    assert(degree(1ULL << 63) == 63);
    assert(mask_width(64) == ~0ULL);
    std::cout << "Poly math passed." << std::endl;
}

void test_irreducibility_and_primitivity() {
    std::cout << "Testing irreducibility and primitivity for various n..." << std::endl;
    for (unsigned n = 2; n <= 12; ++n) {
        auto locus = primitive_locus(n);
        assert(!locus.empty());
        for (u64 p : locus) {
            assert(degree(p) == n);
            assert(is_irreducible(p));
            assert(is_primitive(p));
        }
        std::cout << "  n=" << n << " passed (" << locus.size() << " primitive polynomials found)" << std::endl;
    }

    // Explicit large case (n=31 is often used for LFSRs)
    u64 p31 = (1ULL << 31) | (1ULL << 3) | 1ULL; // x^31 + x^3 + 1
    assert(is_primitive(p31));
    std::cout << "  n=31 passed (Mersenne exponent)" << std::endl;

    // Explicit n=32 case
    u64 p32 = (1ULL << 32) | (1ULL << 22) | (1ULL << 2) | (1ULL << 1) | 1ULL; // x^32 + x^22 + x^2 + x + 1
    assert(is_primitive(p32));
    std::cout << "  n=32 passed" << std::endl;

    std::cout << "Irreducibility and primitivity passed." << std::endl;
}

void test_matrix_and_decimation() {
    std::cout << "Testing Matrix and Decimation for various n..." << std::endl;
    unsigned test_ns[] = {4, 5, 6, 7, 13, 31};
    for (unsigned n : test_ns) {
        u64 p = 0;
        if (n == 4) p = 0b10011;
        else if (n == 5) p = 0b100101;
        else if (n == 6) p = 0b1000011;
        else if (n == 7) p = 0b10000011;
        else if (n == 13) p = 0b10000000011011;
        else if (n == 31) p = (1ULL << 31) | (1ULL << 3) | 1ULL;

        GaloisMatrix M = GaloisMatrix::companion(p);
        u64 x3 = poly_pow_mod(2, 3, p); // Decimate by 3
        u64 s = 1;
        for (int i = 0; i < 50; ++i) {
            u64 s_next_poly = poly_mul_mod(s, 2, p);
            u64 s_next_mat = M.apply(s);
            assert(s_next_poly == s_next_mat);

            // Decimation check
            u64 s_steps_3 = s;
            for(int j=0; j<3; ++j) s_steps_3 = poly_mul_mod(s_steps_3, 2, p);
            u64 s_dec = poly_mul_mod(s, x3, p);
            assert(s_steps_3 == s_dec);

            s = s_next_poly;
        }
        std::cout << "  n=" << n << " passed" << std::endl;
    }
    std::cout << "Matrix and Decimation passed." << std::endl;
}

void test_trace_oracle() {
    std::cout << "Testing Trace against oracle for n=5, 6..." << std::endl;
    // n=5, x^5+x^2+1: 10010110011111000110
    u64 p5 = 0b100101;
    u64 s = 1;
    std::string got5 = "";
    for(int i=0; i<20; ++i) {
        got5 += (calculate_trace(s, p5) ? '1' : '0');
        s = poly_mul_mod(s, 2, p5);
    }
    assert(got5 == "10010110011111000110");

    // n=6, x^6+x+1:   00000100001100010100
    u64 p6 = 0b1000011;
    s = 1;
    std::string got6 = "";
    for(int i=0; i<20; ++i) {
        got6 += (calculate_trace(s, p6) ? '1' : '0');
        s = poly_mul_mod(s, 2, p6);
    }
    assert(got6 == "00000100001100010100");
    std::cout << "Trace oracle passed." << std::endl;
}

int main() {
    test_poly_math();
    test_irreducibility_and_primitivity();
    test_matrix_and_decimation();
    test_trace_oracle();
    std::cout << "ALL TESTS PASSED FOR VARIOUS N" << std::endl;
    return 0;
}
