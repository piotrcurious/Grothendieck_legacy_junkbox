# ESP32 Field Extension Library

A compact library for ESP32 Arduino to improve numerical precision and stability using field extensions over transcendental numbers.

## Overview

This library implements a mathematical approach to enhance floating-point arithmetic precision on resource-constrained devices like the ESP32. By representing numbers as elements in a field extension over transcendental numbers (π, e, √2), the library can:

1. Improve numerical precision beyond standard 32-bit floats
2. Enhance stability in calculations involving large magnitude differences
3. Maintain exact representations of important transcendental numbers
4. Reduce propagation of rounding errors in complex calculations

## How It Works

Instead of representing numbers as simple floating-point values, this library represents each number as a linear combination of basis elements:

```
a + b·π + c·e + d·√2 + e·π² + f·e·π + ...
```

Where a, b, c, d, etc. are floating-point coefficients, and the basis elements are combinations of transcendental numbers like π, e, and √2.

This approach allows for:
- Exact representation of important transcendental numbers
- Preservation of numerical relationships through algebraic structure
- Reduced cancellation errors in subtraction of nearly equal values
- Higher effective precision for many mathematical operations

## Features

- Template-based implementation allowing different field extension sizes
- Support for basic arithmetic operations (+, -, *, /)
- Implementation of transcendental functions (sin, cos, exp, log)
- Configurable basis elements
- Low memory footprint suitable for ESP32 and other microcontrollers
- Easy conversion between standard floats and field elements

## Usage

### Basic Example

```cpp
#include "FieldExtension.h"

void setup() {
  Serial.begin(115200);
  
  // Create field elements
  FieldElement4 a(3.5);            // Regular number
  FieldElement4 b = FieldElement4::pi();  // Exact π
  FieldElement4 c = FieldElement4::e();   // Exact e
  
  // Perform calculations
  FieldElement4 result = a * b + c / a;
  
  // Convert back to float
  float floatResult = result.toFloat();
  
  Serial.print("Result: ");
  Serial.println(floatResult);
}
```

### Improving Precision

The library shines when calculating expressions that would normally suffer from floating-point errors:

```cpp
// Standard float calculation (shows error due to float limitations)
float std_result = pow(pi_f + e_f, 2) - pow(pi_f, 2) - pow(e_f, 2) - 2 * pi_f * e_f;

// Field extension calculation (maintains exact representation)
FieldElement4 pi = FieldElement4::pi();
FieldElement4 e = FieldElement4::e();
FieldElement4 field_result = (pi + e) * (pi + e) - pi * pi - e * e - pi * e * 2.0;
```

### Enhancing Numerical Stability

The library provides better stability for calculations involving large magnitude differences:

```cpp
// Standard float calculation (loses precision)
float large_num = 1000000.0;
float pi_f = 3.14159265359;
float std_result = (large_num + pi_f) - large_num;  // May not equal π exactly

// Field extension calculation (maintains exact value)
FieldElement4 large_fe(large_num);
FieldElement4 pi = FieldElement4::pi();
FieldElement4 field_result = (large_fe + pi) - large_fe;  // Exactly π
```

## Field Extension Sizes

The library provides three predefined field extension sizes:

- **FieldElement4**: Basic field with [1, π, e, √2]
- **FieldElement8**: Extended field with more combinations
- **FieldElement16**: Advanced field with many combinations

Larger field extensions provide more precision but require more memory and computation.

## Performance Considerations

- Each operation with field elements requires multiple floating-point operations
- Memory usage increases linearly with the field extension size
- The ESP32's floating-point unit allows for efficient implementation

## Implementation Notes

The library approximates products that would result in basis elements not included in the chosen field extension. This approximation is a necessary trade-off for maintaining a compact representation.

## License

This library is released under the MIT License.
