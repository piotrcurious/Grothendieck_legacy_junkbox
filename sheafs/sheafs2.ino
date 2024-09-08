#include <avr/pgmspace.h>

#define DATA_POINTS 16
int sensorPin = A0;
int timestamps[DATA_POINTS];
int data[DATA_POINTS];
int dataIndex = 0;

typedef struct {
    int timestamp;
    int value;
} PresheafData;

PresheafData presheaf[DATA_POINTS];

const int candidatePolynomials[] = {0b1, 0b11, 0b101, 0b110, 0b111};
const int numCandidates = sizeof(candidatePolynomials) / sizeof(candidatePolynomials[0]);

void collectData() {
    if (dataIndex < DATA_POINTS) {
        int reading = analogRead(sensorPin) % 2;
        int time = millis();

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
        delay(random(100, 1000));
    }
}

void constructSheaf() {
    Serial.println("Constructing Sheaf via Iterative Polynomial Matching:");
    int bestPolynomial = 0;
    int bestError = INT_MAX;

    for (int i = 1; i < dataIndex; i++) {
        int diffValue = (presheaf[i].value - presheaf[i - 1].value) % 2;
        int diffTime = presheaf[i].timestamp - presheaf[i - 1].timestamp;

        int derivedPoly = approximateDerivative(presheaf[i - 1].value, presheaf[i].value, diffTime);

        if (isValidPolynomial(derivedPoly, diffTime)) {
            int error = evaluatePolynomial(derivedPoly, diffValue, diffTime);
            if (error < bestError) {
                bestPolynomial = derivedPoly;
                bestError = error;
            }
        }
    }

    Serial.print("Selected Best Polynomial: ");
    Serial.println(bestPolynomial, BIN);
}

int approximateDerivative(int prevValue, int currValue, int diffTime) {
    int derivative = (currValue - prevValue) / max(diffTime, 1); // Prevent division by zero
    int candidatePoly = 0;

    if (derivative == 0) candidatePoly = 0b1;
    else if (derivative == 1) candidatePoly = 0b11;
    else candidatePoly = 0b101;

    return candidatePoly;
}

int evaluatePolynomial(int poly, int diffValue, int diffTime) {
    int feedback = poly & diffValue;
    int timePenalty = abs(diffTime - feedback);
    return timePenalty + computeHammingDistance(poly, diffValue);
}

int computeHammingDistance(int poly, int diffValue) {
    int errorCount = 0;
    int xorResult = poly ^ diffValue;
    while (xorResult) {
        errorCount += xorResult & 1;
        xorResult >>= 1;
    }
    return errorCount;
}

bool isValidPolynomial(int poly, int diffTime) {
    const int maxBandwidth = 3;
    int termCount = __builtin_popcount(poly);
    int rateOfChange = diffTime;
    return termCount <= maxBandwidth && rateOfChange >= 100;
}

int monteCarloSearch(int* data, int size) {
    int bestCandidate = 0;
    int minError = INT_MAX;

    for (int i = 0; i < size - 1; i++) {
        int derivedPoly = approximateDerivative(data[i], data[i + 1], timestamps[i + 1] - timestamps[i]);
        int error = computeTotalError(derivedPoly, data, size);

        if (error < minError) {
            minError = error;
            bestCandidate = derivedPoly;
        }
    }

    return (minError < acceptableErrorThreshold()) ? bestCandidate : binarySearchPolynomials(data, size);
}

int computeTotalError(int poly, int* data, int size) {
    int totalError = 0;
    int state = data[0];
    for (int i = 1; i < size; i++) {
        int next = 0;
        for (int j = 0; j < size; j++) {
            if ((poly >> j) & 1) {
                next ^= (state >> j) & 1;
            }
        }
        totalError += (next != data[i]);
        state = (state >> 1) | (next << (size - 1));
    }
    return totalError;
}

int acceptableErrorThreshold() {
    return 2;
}

int binarySearchPolynomials(int* data, int size) {
    int low = 0, high = (1 << size) - 1;
    int bestMatch = -1;
    int minError = INT_MAX;

    while (low <= high) {
        int mid = (low + high) / 2;
        int error = computeTotalError(mid, data, size);

        if (error < minError) {
            minError = error;
            bestMatch = mid;
        }

        if (minError <= acceptableErrorThreshold()) break;

        if (mid < data[0]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return bestMatch;
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
        delay(2000);  // Optional delay before starting the next cycle
    }
}
