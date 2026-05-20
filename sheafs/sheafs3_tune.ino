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
    0b10101, 0b11011, 0b11101, 0b11111
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
    return windowSize > 4 ? 2 * (windowSize - 4) : 0;
}

int evaluatePolynomial(int poly, int derivedPoly, int windowSize) {
    // Compute error based on least squares approximation or fitting criteria
    int error = computeHammingDistance(poly, derivedPoly) + windowSizePenalty(windowSize);
    return error;
}

int approximateDerivative(int prevValue, int currValue, int diffTime) {
    // Approximate the derivative using finite differences
    int derivative = (currValue - prevValue) / max(diffTime, 1);  // Protect against division by zero
    // Selecting a simple polynomial based on the observed rate of change
    if (derivative == 0) return 0b1;
    if (derivative == 1) return 0b11;
    return 0b101;
}

void constructSheaf() {
    Serial.println("Constructing Sheaf with Improved Polynomial Matching:");
    int bestPolynomial = 0;
    int bestError = INT_MAX;
    int bestStart = 0;

    // Iterate over different window sizes to expand data used in sheaf construction
    for (int window = 2; window <= dataIndex; window++) {
        for (int start = 0; start <= dataIndex - window; start++) {
            int derivedPoly = 0;

            // Approximate derivative using a broader set of points in the current window
            for (int i = start + 1; i < start + window; i++) {
                int diffTime = presheaf[i].timestamp - presheaf[i - 1].timestamp;

                derivedPoly ^= approximateDerivative(presheaf[i - 1].value, presheaf[i].value, diffTime);
            }

            // Evaluate candidate polynomials against the constructed sheaf data
            for (int i = 0; i < numCandidates; i++) {
                int candidate = candidatePolynomials[i];
                int error = evaluatePolynomial(candidate, derivedPoly, window);

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
    int state = data_seq[0];
    for (int i = 1; i < size; i++) {
        int next = 0;
        for (int j = 0; j < 8; j++) { // Assuming 8-bit state for LFSR simulation
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
    return 2;
}

int searchPolynomialsBruteForce(int* data_seq, int size) {
    int bestMatch = -1;
    int minError = INT_MAX;

    // Brute force search in 8-bit space instead of broken binary search
    for (int i = 0; i < 256; i++) {
        int error = computeTotalError(i, data_seq, size);
        if (error < minError) {
            minError = error;
            bestMatch = i;
        }
        if (minError <= acceptableErrorThreshold()) break;
    }
    return bestMatch;
}

int monteCarloSearch(int* data_seq, int size) {
    int bestCandidate = 0;
    int minError = INT_MAX;

    for (int i = 0; i < size - 1; i++) {
        int diffTime = timestamps[i + 1] - timestamps[i];
        int derivedPoly = approximateDerivative(data_seq[i], data_seq[i + 1], diffTime);
        int error = computeTotalError(derivedPoly, data_seq, size);

        if (error < minError) {
            minError = error;
            bestCandidate = derivedPoly;
        }
    }

    return (minError <= acceptableErrorThreshold()) ? bestCandidate : searchPolynomialsBruteForce(data_seq, size);
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

        int feedbackPoly = monteCarloSearch(data, DATA_POINTS);
        Serial.print("Best Feedback Polynomial (Monte Carlo Search): ");
        Serial.println(feedbackPoly, BIN);

        // Reset data for the next round of collection and processing
        dataIndex = 0;
        Serial.println("Resetting for next data collection cycle...");
        delay(2000);
    }
}
