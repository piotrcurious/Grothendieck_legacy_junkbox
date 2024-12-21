// Chebyshev polynomials coefficients up to order 8
// Each row represents coefficients for polynomial Tn
const float chebyshev_coeffs[][9] = {
    {1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000}, // T0
    {0.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000}, // T1
    {-1.000000, 0.000000, 2.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000}, // T2
    {0.000000, -3.000000, 0.000000, 4.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000}, // T3
    {1.000000, 0.000000, -8.000000, 0.000000, 8.000000, 0.000000, 0.000000, 0.000000, 0.000000}, // T4
    {0.000000, 5.000000, 0.000000, -20.000000, 0.000000, 16.000000, 0.000000, 0.000000, 0.000000}, // T5
    {-1.000000, 0.000000, 18.000000, 0.000000, -48.000000, 0.000000, 32.000000, 0.000000, 0.000000}, // T6
    {0.000000, -7.000000, 0.000000, 56.000000, 0.000000, -112.000000, 0.000000, 64.000000, 0.000000}, // T7
    {1.000000, 0.000000, -32.000000, 0.000000, 160.000000, 0.000000, -256.000000, 0.000000, 128.000000}  // T8
};

// Structure to hold decomposition coefficients
struct ChebyshevDecomposition {
    float coefficients[9];  // Coefficients for each basis polynomial
    uint8_t order;         // Highest order used
};

class ChebyshevProcessor {
private:
    static const int MAX_ORDER = 8;
    static const int SAMPLE_POINTS = 50;  // Number of points to use in decomposition
    
    // Evaluate a single Chebyshev polynomial at x
    float evaluateSinglePolynomial(int n, float x) {
        float result = 0;
        for (int i = 0; i <= n; i++) {
            result += chebyshev_coeffs[n][i] * pow(x, i);
        }
        return result;
    }

public:
    // Decompose a dataset into Chebyshev coefficients
    ChebyshevDecomposition decompose(float* data, int dataSize, int order) {
        ChebyshevDecomposition result;
        order = min(order, MAX_ORDER);
        result.order = order;

        // Initialize coefficients to zero
        for (int i = 0; i <= MAX_ORDER; i++) {
            result.coefficients[i] = 0;
        }

        // Scale x values to [-1, 1] interval
        float* scaledX = new float[dataSize];
        for (int i = 0; i < dataSize; i++) {
            scaledX[i] = 2.0 * i / (dataSize - 1) - 1.0;
        }

        // Calculate coefficients using discrete approximation
        for (int k = 0; k <= order; k++) {
            float sum = 0;
            for (int i = 0; i < dataSize; i++) {
                sum += data[i] * evaluateSinglePolynomial(k, scaledX[i]);
            }
            result.coefficients[k] = (2.0 / dataSize) * sum;
        }

        // Adjust first coefficient
        result.coefficients[0] *= 0.5;

        delete[] scaledX;
        return result;
    }

    // Reconstruct data from Chebyshev coefficients
    void reconstruct(const ChebyshevDecomposition& decomp, float* output, int outputSize) {
        for (int i = 0; i < outputSize; i++) {
            float x = 2.0 * i / (outputSize - 1) - 1.0;
            output[i] = 0;
            
            for (int k = 0; k <= decomp.order; k++) {
                output[i] += decomp.coefficients[k] * evaluateSinglePolynomial(k, x);
            }
        }
    }

    // Calculate compression ratio
    float getCompressionRatio(int originalSize, const ChebyshevDecomposition& decomp) {
        int compressedSize = (decomp.order + 1) * sizeof(float) + sizeof(uint8_t);
        return (float)originalSize * sizeof(float) / compressedSize;
    }

    // Calculate RMS error between original and reconstructed data
    float calculateRMSError(float* original, float* reconstructed, int size) {
        float sumSquaredError = 0;
        for (int i = 0; i < size; i++) {
            float error = original[i] - reconstructed[i];
            sumSquaredError += error * error;
        }
        return sqrt(sumSquaredError / size);
    }
};

// Test function demonstrating usage
void testChebyshevCompression() {
    const int TEST_SIZE = 100;
    float testData[TEST_SIZE];
    
    // Generate test data (example: damped sine wave)
    for (int i = 0; i < TEST_SIZE; i++) {
        float x = (float)i / TEST_SIZE * 2 * PI;
        testData[i] = exp(-x/3) * sin(5*x);
    }

    ChebyshevProcessor processor;
    
    // Test different orders of approximation
    for (int order = 2; order <= 8; order += 2) {
        // Decompose
        ChebyshevDecomposition decomp = processor.decompose(testData, TEST_SIZE, order);
        
        // Reconstruct
        float reconstructed[TEST_SIZE];
        processor.reconstruct(decomp, reconstructed, TEST_SIZE);
        
        // Calculate metrics
        float compressionRatio = processor.getCompressionRatio(TEST_SIZE, decomp);
        float rmsError = processor.calculateRMSError(testData, reconstructed, TEST_SIZE);
        
        // Print results
        Serial.printf("Order %d:\n", order);
        Serial.printf("Compression ratio: %.2f:1\n", compressionRatio);
        Serial.printf("RMS Error: %.6f\n", rmsError);
        Serial.printf("Coefficients: ");
        for (int i = 0; i <= order; i++) {
            Serial.printf("%.4f ", decomp.coefficients[i]);
        }
        Serial.println("\n");
        delay(100);  // Give serial time to flush
    }
}

void setup() {
    Serial.begin(115200);
    while (!Serial) delay(100);
    
    Serial.println("Starting Chebyshev compression test...");
    testChebyshevCompression();
}

void loop() {
    // Empty loop
}
