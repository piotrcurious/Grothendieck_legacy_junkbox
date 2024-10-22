#include <Arduino.h>
#include <SD.h>
#include <SPI.h>

// LFSR class for pseudo-random number generation
class LFSR {
  private:
    uint32_t state;
    uint32_t taps;
    uint8_t length;

  public:
    LFSR(uint32_t init_state, uint32_t tap_polynomial, uint8_t len) : state(init_state), taps(tap_polynomial), length(len) {}

    uint8_t next() {
      uint8_t lsb = state & 1;   // Get the least significant bit
      state >>= 1;               // Shift state to the right by 1
      if (lsb) {
        state ^= taps;           // Apply taps if the LSB was 1
      }
      return state & 1;          // Return the new LSB
    }

    uint32_t getState() {
      return state;
    }
};

// Polynomial derivative in GF(2)
uint32_t polyDerivativeGF2(uint32_t poly) {
  uint32_t derivative = 0;
  uint32_t mask = 2;   // Start with x^1
  while (poly >= mask) {
    if (poly & mask) {
      derivative ^= (mask >> 1);   // Add the lower degree term to the derivative
    }
    mask <<= 1;
  }
  return derivative;
}

// Feature hashing
uint32_t featureHash(uint32_t sequence) {
  return sequence ^ (sequence >> 16);  // Simple hash function
}

// SD Card setup
const int chipSelect = 5;
File featureFile;
bool learningMode = true;  // Enable/Disable learning mode

// Initialize the SD card
void initSDCard() {
  if (!SD.begin(chipSelect)) {
    Serial.println("SD Card initialization failed!");
    return;
  }
  Serial.println("SD Card initialized.");
}

// Save a feature to the SD card
void saveFeatureToSD(uint32_t featureHash, const char* featureName) {
  featureFile = SD.open("features.txt", FILE_WRITE);
  if (featureFile) {
    featureFile.print(featureHash, HEX);
    featureFile.print(",");
    featureFile.println(featureName);
    featureFile.close();
    Serial.print("Saved feature hash: 0x");
    Serial.print(featureHash, HEX);
    Serial.print(" as ");
    Serial.println(featureName);
  } else {
    Serial.println("Error opening features.txt for writing.");
  }
}

// Check if a feature already exists on the SD card
bool matchFeatureFromSD(uint32_t featureHash) {
  featureFile = SD.open("features.txt", FILE_READ);
  if (featureFile) {
    while (featureFile.available()) {
      String line = featureFile.readStringUntil('\n');
      int separatorIndex = line.indexOf(',');
      String storedHashStr = line.substring(0, separatorIndex);
      uint32_t storedHash = (uint32_t)strtol(storedHashStr.c_str(), NULL, 16);

      if (storedHash == featureHash) {
        String storedName = line.substring(separatorIndex + 1);
        Serial.print("Matched feature: ");
        Serial.println(storedName);
        featureFile.close();
        return true;
      }
    }
    featureFile.close();
  } else {
    Serial.println("Error opening features.txt for reading.");
  }
  return false;
}

// List all stored features from the SD card
void listFeatures() {
  featureFile = SD.open("features.txt", FILE_READ);
  if (featureFile) {
    Serial.println("Listing stored features:");
    while (featureFile.available()) {
      String line = featureFile.readStringUntil('\n');
      Serial.println(line);
    }
    featureFile.close();
  } else {
    Serial.println("Error opening features.txt for reading.");
  }
}

// Delete all stored features on the SD card
void deleteAllFeatures() {
  if (SD.exists("features.txt")) {
    SD.remove("features.txt");
    Serial.println("All features deleted.");
  } else {
    Serial.println("No features to delete.");
  }
}

// Analyze the signal and detect or store a feature
void analyzeSignal(LFSR &lfsr, int numSteps, const char* featureName) {
  uint32_t sequence = 0;
  for (int i = 0; i < numSteps; i++) {
    sequence = (sequence << 1) | lfsr.next();
  }

  // Hash the sequence
  uint32_t hash = featureHash(sequence);
  Serial.print("Feature Hash: 0x");
  Serial.println(hash, HEX);

  // Check if feature already exists
  if (matchFeatureFromSD(hash)) {
    Serial.println("Feature already known.");
  } else {
    if (learningMode) {
      Serial.println("New feature detected. Saving to SD...");
      saveFeatureToSD(hash, featureName);
    } else {
      Serial.println("Unknown feature detected (Non-learning mode).");
    }
  }
}

// Command handling from the user via Serial input
void handleCommand(String command) {
  if (command.startsWith("sample")) {
    // Extract the feature name and trigger feature detection
    String featureName = command.substring(command.indexOf(' ') + 1);
    LFSR lfsr(0x1, 0x6, 3);  // Example LFSR initialization
    analyzeSignal(lfsr, 7, featureName.c_str());  // Sample for 7 steps
  } else if (command == "list") {
    listFeatures();
  } else if (command == "delete") {
    deleteAllFeatures();
  } else if (command == "learn on") {
    learningMode = true;
    Serial.println("Learning mode ON.");
  } else if (command == "learn off") {
    learningMode = false;
    Serial.println("Learning mode OFF.");
  } else {
    Serial.println("Unknown command.");
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);  // Wait for Serial to be ready

  // Initialize the SD card
  initSDCard();

  // Print available commands
  Serial.println("Command options:");
  Serial.println("1. sample <name>: Trigger signal sampling and feature detection.");
  Serial.println("2. list: List all stored features.");
  Serial.println("3. delete: Delete all stored features.");
  Serial.println("4. learn on: Enable learning mode.");
  Serial.println("5. learn off: Disable learning mode.");
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove extra whitespace
    handleCommand(command);  // Process the command
  }
}
