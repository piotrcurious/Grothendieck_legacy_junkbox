#include <avr/pgmspace.h>
#include <limits.h>

#define DATA_POINTS 32  // Expanded data collection to enhance sheaf construction
int sensorPin = A0;
int timestamps[DATA_POINTS];
int data[DATA_POINTS];
int dataIndex = 0;

typedef struct {
    int timestamp;
    int value;
} PresheafData;

PresheafData presheaf[DATA_POINTS];

// Expanded set of candidate polynomials to consider more complex patterns
const int candidatePolynomials[] = {
    0b1, 0b11, 0b101, 0b110, 0b111,
    0b1001, 0b1011, 0b1101, 0b1111, 0b10011,
    0b10101, 0b11011, 0b11101, 0b11111,
    0x1D, 0x8D, 0xAF, 0xB4 // Common 8-bit LFSR polynomials
};
const int numCandidates = sizeof(candidatePolynomials) / sizeof(candidatePolynomials[0]);

void collectData() {
    if (dataIndex < DATA_POINTS) {
        int reading = analogRead(sensorPin) % 2;  // Reading binary data (0 or 1)
        int time = millis();

        // Store data into the presheaf structure
        presheaf[dataIndex].timestamp = time;
        presheaf[dataIndex].value = reading;
        timestamps[dataIndex] = time;
        data[dataIndex] = reading;

        Serial.print("Collected Data Point: ");
        Serial.print("Time: ");
        Serial.print(time);
        Serial.print(" ms, Value: ");
        Serial.println(reading);

        dataIndex++;
        delay(random(50, 500));  // Random interval to simulate irregular sampling
    }
}

int computeHammingDistance(int poly, int derivedPoly) {
    // Compute error using Hamming distance to compare polynomials
    int errorCount = 0;
    int xorResult = poly ^ derivedPoly;
    while (xorResult) {
        errorCount += xorResult & 1;
        xorResult >>= 1;
    }
    return errorCount;
}

int windowSizePenalty(int windowSize) {
    // Adjust penalty based on the window size used for constructing the sheaf
    // Favor windows between 4 and 8
    if (windowSize < 4) return (4 - windowSize) * 2;
    if (windowSize > 8) return (windowSize - 8) * 2;
    return 0;
}

int evaluatePolynomial(int poly, int derivedPoly, int windowSize) {
    // Compute error based on Hamming distance and window size penalty
    int error = computeHammingDistance(poly, derivedPoly) + windowSizePenalty(windowSize);
    return error;
}

// Improved candidate generator based on local bit patterns
int generateCandidateFromPattern(int* data_seg, int len) {
    if (len < 2) return 0b1;

    int pattern = 0;
    for (int i = 0; i < len && i < 8; i++) {
        if (data_seg[i]) pattern |= (1 << i);
    }

    // Mix in some transition information
    int transitions = 0;
    for (int i = 1; i < len && i < 8; i++) {
        if (data_seg[i] != data_seg[i-1]) transitions |= (1 << (i-1));
    }

    return pattern ^ (transitions << 1) ^ 0b1; // Ensure at least the constant term is considered
}

void constructSheaf() {
    Serial.println("Constructing Sheaf with Improved Polynomial Matching:");
    int bestPolynomial = 0;
    int bestError = INT_MAX;
    int bestStart = 0;

    // Iterate over different window sizes to expand data used in sheaf construction
    for (int window = 4; window <= 12 && window <= dataIndex; window++) {
        for (int start = 0; start <= dataIndex - window; start++) {

            // Extract a local candidate from this window's bit pattern
            int localCandidate = generateCandidateFromPattern(&data[start], window);

            // Evaluate candidate polynomials against the constructed sheaf data
            for (int i = 0; i < numCandidates; i++) {
                int candidate = candidatePolynomials[i];
                int error = evaluatePolynomial(candidate, localCandidate, window);

                // Update the best polynomial based on the least error found
                if (error < bestError) {
                    bestError = error;
                    bestPolynomial = candidate;
                    bestStart = start;
                }
            }
        }
    }

    Serial.print("Selected Best Polynomial: ");
    Serial.print(bestPolynomial, BIN);
    Serial.print(" with error: ");
    Serial.print(bestError);
    Serial.print(", starting at index: ");
    Serial.println(bestStart);
}

int computeTotalError(int poly, int* data_seq, int size) {
    int totalError = 0;
    int state = 0;
    // Initialize state with first 8 bits
    for (int i = 0; i < 8 && i < size; i++) {
        if (data_seq[i]) state |= (1 << i);
    }

    for (int i = 8; i < size; i++) {
        int next = 0;
        for (int j = 0; j < 8; j++) {
            if ((poly >> j) & 1) {
                next ^= (state >> j) & 1;
            }
        }
        totalError += (next != data_seq[i]);
        state = (state >> 1) | (next << 7);
    }
    return totalError;
}

int acceptableErrorThreshold() {
    return 4; // Allow some noise
}

int searchPolynomialsBruteForce(int* data_seq, int size) {
    int bestMatch = -1;
    int minError = INT_MAX;

    for (int i = 0; i < 256; i++) {
        int error = computeTotalError(i, data_seq, size);
        if (error < minError) {
            minError = error;
            bestMatch = i;
        }
        if (minError <= 1) break; // Good enough
    }
    return bestMatch;
}

int monteCarloSearch(int* data_seq, int* timestamps_seq, int size) {
    int bestCandidate = 0;
    int minError = INT_MAX;

    // Use some random starting points based on local patterns
    for (int i = 0; i < 10; i++) {
        int start = random(0, size - 8);
        int len = random(4, 9);
        int derivedPoly = generateCandidateFromPattern(&data_seq[start], len);
        int error = computeTotalError(derivedPoly, data_seq, size);

        if (error < minError) {
            minError = error;
            bestCandidate = derivedPoly;
        }
        if (minError <= acceptableErrorThreshold()) return bestCandidate;
    }

    return searchPolynomialsBruteForce(data_seq, size);
}

void setup() {
    Serial.begin(9600);
    randomSeed(analogRead(0));
}

void loop() {
    if (dataIndex < DATA_POINTS) {
        collectData();
    } else {
        Serial.println("Data Collection Complete.");
        constructSheaf();

        int feedbackPoly = monteCarloSearch(data, timestamps, DATA_POINTS);
        Serial.print("Best Feedback Polynomial (Monte Carlo Search): ");
        Serial.println(feedbackPoly, BIN);

        // Reset data for the next round of collection and processing
        dataIndex = 0;
        Serial.println("Resetting for next data collection cycle...");
        delay(2000);
    }
}
