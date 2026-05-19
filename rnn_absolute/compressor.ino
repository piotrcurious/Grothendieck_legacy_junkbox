#include <Arduino.h>
#include "transformation_params.h"

// Define the chunk size to be used for this run
const int CHUNK_SIZE = 4; // Adjust this value based on your needs

// Polynomial transformation function
double polynomialTransform(double x, const double* coeffs, int num_coeffs) {
    double result = 0;
    for (int i = 0; i < num_coeffs; i++) {
        result += coeffs[i] * pow(x, i);
    }
    return result;
}

// Compression function
void compressData(const double* data, int dataSize, double* compressedData) {
    int numChunks = dataSize / CHUNK_SIZE;
    for (int chunk = 0; chunk < numChunks; chunk++) {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            compressedData[chunk * CHUNK_SIZE + i] = polynomialTransform(data[chunk * CHUNK_SIZE + i], transformation_coeffs, num_coeffs);
        }
    }
}

// Decompression function (assuming inverse transformation is known)
double inversePolynomialTransform(double y, const double* coeffs, int num_coeffs) {
    // Inverse transformation logic needs to be defined. This is a placeholder.
    return y; // Replace this with actual inverse transformation logic.
}

void decompressData(const double* compressedData, int dataSize, double* decompressedData) {
    int numChunks = dataSize / CHUNK_SIZE;
    for (int chunk = 0; chunk < numChunks; chunk++) {
        for (int i = 0; i < CHUNK_SIZE; i++) {
            decompressedData[chunk * CHUNK_SIZE + i] = inversePolynomialTransform(compressedData[chunk * CHUNK_SIZE + i], transformation_coeffs, num_coeffs);
        }
    }
}

void setup() {
    Serial.begin(9600);
    
    // Example data to compress and decompress
    double exampleData[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; // Adjust based on your needs
    int dataSize = sizeof(exampleData) / sizeof(exampleData[0]);
    
    // Ensure dataSize is a multiple of CHUNK_SIZE
    if (dataSize % CHUNK_SIZE != 0) {
        Serial.println("Error: Data size is not a multiple of CHUNK_SIZE.");
        while (true); // Halt execution
    }
    
    double compressedData[dataSize];
    double decompressedData[dataSize];

    // Compress the data
    compressData(exampleData, dataSize, compressedData);
    
    Serial.println("Compressed Data:");
    for (int i = 0; i < dataSize; i++) {
        Serial.println(compressedData[i], 6); // Print with 6 decimal places
    }

    // Decompress the data
    decompressData(compressedData, dataSize, decompressedData);

    Serial.println("Decompressed Data:");
    for (int i = 0; i < dataSize; i++) {
        Serial.println(decompressedData[i], 6); // Print with 6 decimal places
    }
}

void loop() {
    // Nothing to do in loop
}
