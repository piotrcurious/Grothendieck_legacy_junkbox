#include "Arduino.h"
#include "../FieldExtension.h"
#include <cassert>
#include <iostream>
#include <iomanip>

void demoBasicOperations();
void demoPrecisionComparison();
void demoNumericalStability();
void demoTranscendentalFunctions();
void demoAdvancedMath();

int main() {
    std::cout << "Starting ESP32 Field Extension Library Tests..." << std::endl;

    demoBasicOperations();
    demoPrecisionComparison();
    demoNumericalStability();
    demoTranscendentalFunctions();
    demoAdvancedMath();

    std::cout << "All tests completed successfully!" << std::endl;
    return 0;
}

void demoBasicOperations() {
    std::cout << "\n=== Basic Operations ===" << std::endl;

    FieldElement4 a(3.5f);
    FieldElement4 b = FieldElement4::pi();
    FieldElement4 c = FieldElement4::e();

    FieldElement4 res = a;
    res += b;
    std::cout << "3.5 + π = " << res.toFloat() << std::endl;
    assert(std::abs(res.toFloat() - (3.5f + M_PI)) < 1e-5);

    FieldElement4 twoPi = 2.0f * b;
    std::cout << "2 × π = " << twoPi.toFloat() << std::endl;
    assert(std::abs(twoPi.toFloat() - 2.0f * M_PI) < 1e-5);

    FieldElement4 result = a * b + c / a;
    std::cout << "3.5 × π + e ÷ 3.5 = " << result.toFloat() << std::endl;
}

void demoPrecisionComparison() {
    std::cout << "\n=== Precision Comparison ===" << std::endl;

    float pi_f = 3.14159265359f;
    float e_f = 2.71828182846f;
    float std_result = pow(pi_f + e_f, 2) - pow(pi_f, 2) - pow(e_f, 2) - 2 * pi_f * e_f;

    std::cout << "Standard float result (should be 0): " << std_result << std::endl;

    FieldElement4 pi = FieldElement4::pi();
    FieldElement4 e = FieldElement4::e();
    FieldElement4 sum = pi + e;
    FieldElement4 squared = sum * sum;
    FieldElement4 pi_squared = pi * pi;
    FieldElement4 e_squared = e * e;
    FieldElement4 two_pi_e = pi * e * 2.0f;
    FieldElement4 field_result = squared - pi_squared - e_squared - two_pi_e;

    std::cout << "Field extension result (should be 0): " << field_result.toFloat() << std::endl;
}

void demoNumericalStability() {
    std::cout << "\n=== Numerical Stability ===" << std::endl;

    float large_num = 1000000.0f;
    float pi_f = 3.14159265359f;
    float std_result = (large_num + pi_f) - large_num;

    std::cout << "Standard float: (1000000 + π) - 1000000 = " << std_result << std::endl;

    FieldElement4 large_fe(large_num);
    FieldElement4 pi = FieldElement4::pi();
    FieldElement4 sum = large_fe + pi;
    FieldElement4 field_result = sum - large_fe;

    std::cout << "Field extension: (1000000 + π) - 1000000 = " << field_result.toFloat() << std::endl;
    assert(std::abs(field_result.toFloat() - M_PI) < 1e-5);
}

void demoTranscendentalFunctions() {
    std::cout << "\n=== Transcendental Functions ===" << std::endl;

    FieldElement4 x = FieldElement4::pi() * 0.5f;
    FieldElement4 sin_x = sin(x);
    FieldElement4 cos_x = cos(x);

    std::cout << "sin(π/2) = " << sin_x.toFloat() << std::endl;
    std::cout << "cos(π/2) = " << cos_x.toFloat() << std::endl;

    FieldElement4 identity = sin_x * sin_x + cos_x * cos_x;
    std::cout << "sin²(π/2) + cos²(π/2) = " << identity.toFloat() << std::endl;
    assert(std::abs(identity.toFloat() - 1.0f) < 1e-6);

    FieldElement4 pi = FieldElement4::pi();
    std::cout << "Symbolic exactness: sin(pi) = " << sin(pi).getCoefficient(0) << std::endl;
    assert(sin(pi).getCoefficient(0) == 0.0f);
}

void demoAdvancedMath() {
  std::cout << "\n=== Advanced Math Features ===" << std::endl;
  FieldElement4 pi = FieldElement4::pi();
  FieldElement4 p2 = pow(pi, 2.0f);
  std::cout << "π² = " << p2.toFloat() << std::endl;
  FieldElement4 sq = sqrt(p2);
  std::cout << "sqrt(π²) = " << sq.toFloat() << std::endl;
  assert(std::abs(sq.toFloat() - M_PI) < 1e-5);

  FieldElement4 hx(0.5f);
  FieldElement4 s = sinh(hx);
  FieldElement4 c = cosh(hx);
  FieldElement4 res = c * c - s * s;
  std::cout << "cosh²(0.5) - sinh²(0.5) = " << res.toFloat() << std::endl;
  assert(std::abs(res.toFloat() - 1.0f) < 1e-6);

  FieldElement4 one(1.0f);
  FieldElement4 a = atan(one);
  std::cout << "atan(1) = " << a.toFloat() << " (pi/4=" << M_PI*0.25 << ")" << std::endl;
  assert(std::abs(a.toFloat() - M_PI * 0.25f) < 1e-6);
}
