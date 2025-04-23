#ifndef FIELD_EXTENSION_H
#define FIELD_EXTENSION_H

#include <Arduino.h>
#include <vector>
#include <cmath>
#include <initializer_list>

// Transcendental basis enum
enum class Basis {
    One, Pi, E, Ln2
};

// Monomial of form coeff * basis
struct Term {
    float coeff;
    Basis basis;

    float evaluate() const {
        switch (basis) {
            case Basis::One: return coeff;
            case Basis::Pi:  return coeff * M_PI;
            case Basis::E:   return coeff * M_E;
            case Basis::Ln2: return coeff * log(2);
        }
        return NAN;
    }
};

// Field extension as linear combination of transcendental basis
class FieldExtension {
public:
    std::vector<Term> terms;

    FieldExtension() = default;
    FieldExtension(std::initializer_list<Term> list) : terms(list) {}

    // Evaluate to float
    float evaluate() const {
        float sum = 0.0f;
        for (const auto& t : terms) sum += t.evaluate();
        return sum;
    }

    // Add another extension
    FieldExtension operator+(const FieldExtension& other) const {
        FieldExtension result = *this;
        for (const auto& t : other.terms) result.addTerm(t);
        return result;
    }

    // Scalar multiplication
    FieldExtension operator*(float scalar) const {
        FieldExtension result;
        for (const auto& t : terms) {
            result.terms.push_back({t.coeff * scalar, t.basis});
        }
        return result;
    }

    void addTerm(const Term& t) {
        for (auto& existing : terms) {
            if (existing.basis == t.basis) {
                existing.coeff += t.coeff;
                return;
            }
        }
        terms.push_back(t);
    }

    void print(const String& label = "") const {
        Serial.print(label);
        Serial.print("FieldExtension: ");
        for (const auto& t : terms) {
            Serial.print(String(t.coeff) + "*");
            switch (t.basis) {
                case Basis::One: Serial.print("1"); break;
                case Basis::Pi: Serial.print("π"); break;
                case Basis::E: Serial.print("e"); break;
                case Basis::Ln2: Serial.print("ln(2)"); break;
            }
            Serial.print(" + ");
        }
        Serial.println();
    }
};

#endif
