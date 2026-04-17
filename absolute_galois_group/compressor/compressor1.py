import numpy as np
import itertools
import struct
from concurrent.futures import ThreadPoolExecutor

# Step 1: Read Example Data in Binary Format
def read_example_data(file_path):
    """
    Reads example data from a binary file and returns it as a numpy array of doubles.
    
    Args:
        file_path (str): Path to the binary data file.
    
    Returns:
        np.array: Array of doubles read from the file.
    """
    with open(file_path, 'rb') as file:
        data = file.read()
    data = struct.unpack(f'{len(data)//8}d', data)
    return np.array(data)

# Step 2: Define Transformation Functions
def polynomial_transform(x, coeffs):
    """
    Applies a polynomial transformation to a given value x using provided coefficients.
    
    Args:
        x (float): Input value.
        coeffs (list): List of coefficients for the polynomial transformation.
    
    Returns:
        float: Transformed value.
    """
    return sum(c * (x ** i) for i, c in enumerate(coeffs))

# Generate a set of polynomial coefficients (transformation functions)
def generate_polynomial_coefficients(max_degree, coeff_range):
    """
    Generates all possible sets of polynomial coefficients up to a given degree within a specified range.
    
    Args:
        max_degree (int): Maximum degree of the polynomial.
        coeff_range (range): Range of coefficient values.
    
    Returns:
        list: List of tuples, each containing a set of coefficients.
    """
    return list(itertools.product(coeff_range, repeat=max_degree + 1))

# Verify commutativity of the transformation function
def is_commutative(coeffs, data_chunk):
    """
    Checks if the polynomial transformation defined by coeffs is commutative for a given data chunk.
    
    Args:
        coeffs (tuple): Coefficients of the polynomial transformation.
        data_chunk (np.array): Chunk of data to check commutativity on.
    
    Returns:
        bool: True if commutative, False otherwise.
    """
    for x, y in itertools.combinations(data_chunk, 2):
        if not np.isclose(polynomial_transform(polynomial_transform(x, coeffs), coeffs),
                          polynomial_transform(polynomial_transform(y, coeffs), coeffs)):
            return False
    return True

# Apply transformation and check commutativity
def transform_and_check(coeffs, data_chunk):
    """
    Applies a polynomial transformation to a data chunk and checks if the transformation is commutative.
    
    Args:
        coeffs (tuple): Coefficients of the polynomial transformation.
        data_chunk (np.array): Chunk of data to transform.
    
    Returns:
        tuple: (coeffs, transformed_data) if commutative, (None, None) otherwise.
    """
    transformed_data = polynomial_transform(data_chunk, coeffs)
    if is_commutative(coeffs, data_chunk):
        return coeffs, transformed_data
    return None, None

# Step 3: Brute Force Search for Optimal Transformation
def find_best_transformation(data, max_degree, coeff_range, min_window_size, max_window_size):
    """
    Performs a brute-force search to find the best commutative polynomial transformation that compresses the data.
    
    Args:
        data (np.array): Input data array.
        max_degree (int): Maximum degree of the polynomial transformations.
        coeff_range (range): Range of coefficient values.
        min_window_size (int): Minimum size of the data chunk window.
        max_window_size (int): Maximum size of the data chunk window.
    
    Returns:
        tuple: (best_coeffs, best_compressed_size, best_window_size) representing the best transformation parameters.
    """
    best_coeffs = None
    best_compressed_size = float('inf')
    best_window_size = min_window_size

    # Generate all possible polynomial transformations
    transformations = generate_polynomial_coefficients(max_degree, coeff_range)

    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(0, len(data), window_size):
            data_chunk = data[i:i + window_size]
            if len(data_chunk) < window_size:
                continue  # Skip incomplete chunks

            with ThreadPoolExecutor() as executor:
                results = executor.map(lambda coeffs: transform_and_check(coeffs, data_chunk), transformations)
                
                for result in results:
                    coeffs, transformed_data = result
                    if coeffs is None:
                        continue  # Skip non-commutative transformations

                    # Calculate compressed size as a proxy for actual compression efficiency
                    compressed_size = np.log2(len(np.unique(transformed_data))) * len(transformed_data)
                    
                    if compressed_size < best_compressed_size:
                        best_compressed_size = compressed_size
                        best_coeffs = coeffs
                        best_window_size = window_size

    return best_coeffs, best_compressed_size, best_window_size

# Step 4: Store Transformation Parameters in Binary Format
def store_transformation_parameters(coeffs, file_path):
    """
    Stores the transformation parameters (coefficients) in a binary file.
    
    Args:
        coeffs (tuple): Coefficients of the polynomial transformation.
        file_path (str): Path to the binary file to store the parameters.
    """
    with open(file_path, 'wb') as file:
        packed_data = struct.pack(f'{len(coeffs)}d', *coeffs)
        file.write(packed_data)

# Example Usage
example_data_file = 'example_data.bin'
transformation_params_file = 'transformation_params.bin'

# Read the example data
data = read_example_data(example_data_file)
max_degree = 3
coeff_range = range(-10, 11)
min_window_size = 16
max_window_size = 128

# Find the best transformation
best_coeffs, best_compressed_size, best_window_size = find_best_transformation(data, max_degree, coeff_range, min_window_size, max_window_size)

# Store the best transformation parameters
store_transformation_parameters(best_coeffs, transformation_params_file)

print(f"Best Coefficients: {best_coeffs}")
print(f"Best Compressed Size: {best_compressed_size}")
print(f"Best Window Size: {best_window_size}")
