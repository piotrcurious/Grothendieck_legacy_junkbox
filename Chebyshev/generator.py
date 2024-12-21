def generate_chebyshev_coefficients(max_order=8, precision=6):
    """
    Generates Chebyshev polynomials and formats them as Arduino code.
    Returns coefficients for polynomials from T0 to Tn.
    """
    import numpy as np
    
    def chebyshev_recurrence(n):
        """Generate coefficients for nth Chebyshev polynomial using recurrence relation"""
        if n == 0:
            return [1.0]
        elif n == 1:
            return [0.0, 1.0]
        else:
            Tn_minus_1 = chebyshev_recurrence(n - 1)
            Tn_minus_2 = chebyshev_recurrence(n - 2)
            
            # Pad arrays to same length
            max_len = max(len(Tn_minus_1), len(Tn_minus_2))
            Tn_minus_1 = np.pad(Tn_minus_1, (0, max_len - len(Tn_minus_1)))
            Tn_minus_2 = np.pad(Tn_minus_2, (0, max_len - len(Tn_minus_2)))
            
            # Calculate next polynomial using recurrence relation: Tn = 2xTn-1 - Tn-2
            result = np.zeros(max_len + 1)
            result[1:] += 2 * Tn_minus_1
            result -= Tn_minus_2
            
            # Remove trailing zeros
            while abs(result[-1]) < 1e-10 and len(result) > 1:
                result = result[:-1]
                
            return result.tolist()

    # Generate Arduino code
    arduino_code = "// Chebyshev polynomials coefficients up to order " + str(max_order) + "\n"
    arduino_code += "// Each row represents coefficients for polynomial Tn\n"
    arduino_code += "const float chebyshev_coeffs[][" + str(max_order + 2) + "] = {\n"
    
    # Generate coefficients for each order
    for n in range(max_order + 1):
        coeffs = chebyshev_recurrence(n)
        # Pad with zeros to maintain consistent array size
        coeffs.extend([0.0] * (max_order + 2 - len(coeffs)))
        
        # Format coefficients with specified precision
        formatted_coeffs = [f"{x:." + str(precision) + "f" for x in coeffs]
        arduino_code += "    {" + ", ".join(formatted_coeffs) + "}, // T" + str(n) + "\n"
    
    arduino_code += "};"
    
    return arduino_code

# Generate code with 8th order polynomials and 6 decimal precision
print(generate_chebyshev_coefficients(8, 6))
