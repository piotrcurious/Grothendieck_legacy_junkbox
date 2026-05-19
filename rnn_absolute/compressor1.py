import itertools
import struct
import math
from concurrent.futures import ThreadPoolExecutor
import os

def generate_example_data(file_path, num_elements=64):
    """Generates synthetic data if missing."""
    data = [100 * math.sin(i * 0.1) + 120 for i in range(num_elements)]
    with open(file_path, 'wb') as f:
        f.write(struct.pack(f'{num_elements}d', *data))
    return data

def read_example_data(file_path):
    if not os.path.exists(file_path):
        return generate_example_data(file_path)
    with open(file_path, 'rb') as file:
        data = file.read()
    num_elements = len(data) // 8
    return struct.unpack(f'{num_elements}d', data)

def polynomial_transform(x, coeffs):
    if isinstance(x, (list, tuple)):
        return [sum(c * (val ** i) for i, c in enumerate(coeffs)) for val in x]
    return sum(c * (x ** i) for i, c in enumerate(coeffs))

def generate_polynomial_coefficients(max_degree, coeff_range):
    return list(itertools.product(coeff_range, repeat=max_degree + 1))

def is_commutative(coeffs, data_chunk):
    for i in range(len(data_chunk)):
        for j in range(i + 1, len(data_chunk)):
            x, y = data_chunk[i], data_chunk[j]
            fx = polynomial_transform(x, coeffs)
            fy = polynomial_transform(y, coeffs)
            if not math.isclose(polynomial_transform(fx, coeffs),
                               polynomial_transform(fy, coeffs), rel_tol=1e-7):
                return False
    return True

def transform_and_check(coeffs, data_chunk):
    transformed_data = polynomial_transform(data_chunk, coeffs)
    if is_commutative(coeffs, data_chunk):
        return coeffs, transformed_data
    return None, None

def find_best_transformation(data, max_degree, coeff_range, min_window_size, max_window_size):
    best_coeffs = None
    best_compressed_size = float('inf')
    best_window_size = min_window_size
    transformations = generate_polynomial_coefficients(max_degree, coeff_range)

    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(0, len(data), window_size):
            data_chunk = data[i:i + window_size]
            if len(data_chunk) < window_size: continue

            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda c: transform_and_check(c, data_chunk), transformations)
                for result in results:
                    coeffs, transformed_data = result
                    if coeffs is None: continue
                    unique_elements = len(set([round(val, 4) for val in transformed_data]))
                    compressed_size = math.log2(unique_elements) * len(transformed_data) if unique_elements > 0 else 0
                    if compressed_size < best_compressed_size:
                        best_compressed_size, best_coeffs, best_window_size = compressed_size, coeffs, window_size
    return best_coeffs, best_compressed_size, best_window_size

def store_transformation_parameters(coeffs, file_path):
    if coeffs:
        with open(file_path, 'wb') as file:
            file.write(struct.pack(f'{len(coeffs)}d', *coeffs))

if __name__ == "__main__":
    example_data_file = 'example_data.bin'
    transformation_params_file = 'transformation_params.bin'
    data = read_example_data(example_data_file)
    # Fast test parameters
    best_coeffs, best_size, best_win = find_best_transformation(data[:16], 1, range(-2, 3), 4, 8)
    store_transformation_parameters(best_coeffs, transformation_params_file)
    print(f"Best Coefficients: {best_coeffs}, Size: {best_size}, Window: {best_win}")
