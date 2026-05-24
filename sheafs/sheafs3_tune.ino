#include <avr/pgmspace.h>
#include <limits.h>

#define DATA_POINTS 32
int sensorPin = A0;
int timestamps[DATA_POINTS];
int data[DATA_POINTS];
int dataIndex = 0;

typedef struct {
    int timestamp;
    int value;
} PresheafData;

PresheafData presheaf[DATA_POINTS];

// Standard feedback polynomials for common LFSR lengths
const int candidatePolynomials[] = {
    0b1, 0b11, 0b101, 0b110, 0b111,
    0x1D, 0x8D, 0xAF, 0xB4, 0x11D // x^8+x^4+x^3+x^2+1, etc.
};
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
        delay(random(50, 500));
    }
}

// Check how many bits of a sequence match the given polynomial's LFSR output
int countLfsrMatches(int poly, int* seq, int len) {
    if (len <= 8) return 0;
    int matches = 0;
    int state = 0;
    for (int i = 0; i < 8; i++) if (seq[i]) state |= (1 << i);

    for (int i = 8; i < len; i++) {
        int next = 0;
        for (int j = 0; j < 8; j++) {
            if ((poly >> j) & 1) next ^= (state >> j) & 1;
        }
        if (next == seq[i]) matches++;
        state = (state >> 1) | (next << 7);
    }
    return matches;
}

void constructSheaf() {
    Serial.println("Constructing Sheaf (Local Sections Check):");
    int bestPolynomial = 0;
    int bestScore = -1;

    // A "Sheaf" approach: evaluate how polynomials perform on local windows
    // and aggregate their performance.
    for (int i = 0; i < numCandidates; i++) {
        int poly = candidatePolynomials[i];
        int totalMatches = 0;
        int sectionsSupporting = 0;

        // Slide window (local sections)
        for (int start = 0; start <= dataIndex - 16; start += 4) {
            int matches = countLfsrMatches(poly, &data[start], 16);
            if (matches >= 6) { // Majority match in an 8-bit prediction window
                sectionsSupporting++;
                totalMatches += matches;
            }
        }

        // Score based on total matches weighted by consistency across sections
        int score = totalMatches * sectionsSupporting;
        if (score > bestScore) {
            bestScore = score;
            bestPolynomial = poly;
        }
    }

    Serial.print("Sheaf Consensus Best Polynomial: ");
    Serial.print(bestPolynomial, BIN);
    Serial.print(" with score: ");
    Serial.println(bestScore);
}

int computeTotalError(int poly, int* data_seq, int size) {
    if (size <= 8) return size;
    int totalError = 0;
    int state = 0;
    for (int i = 0; i < 8; i++) if (data_seq[i]) state |= (1 << i);

    for (int i = 8; i < size; i++) {
        int next = 0;
        for (int j = 0; j < 8; j++) {
            if ((poly >> j) & 1) next ^= (state >> j) & 1;
        }
        if (next != data_seq[i]) totalError++;
        state = (state >> 1) | (next << 7);
    }
    return totalError;
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

        // Brute force final check
        int bestMatch = -1;
        int minError = INT_MAX;
        for (int i = 0; i < 256; i++) {
            int err = computeTotalError(i, data, DATA_POINTS);
            if (err < minError) {
                minError = err;
                bestMatch = i;
            }
            if (minError == 0) break;
        }

        Serial.print("Final Minimal Error Polynomial: ");
        Serial.print(bestMatch, BIN);
        Serial.print(" (Error: ");
        Serial.print(minError);
        Serial.println(")");

        dataIndex = 0;
        Serial.println("Resetting cycle...");
        delay(3000);
    }
}
