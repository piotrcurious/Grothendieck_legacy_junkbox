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
    FieldExtension(float val) { terms.push_back({val, Basis::One}); }
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

    FieldExtension operator-(const FieldExtension& other) const {
        return (*this) + (other * -1.0f);
    }

    // Scalar multiplication
    FieldExtension operator*(float scalar) const {
        FieldExtension result;
        for (const auto& t : terms) {
            result.terms.push_back({t.coeff * scalar, t.basis});
        }
        return result;
    }

    // Multiplication (very basic distributive law for basis products)
    FieldExtension operator*(const FieldExtension& other) const {
        FieldExtension result;
        for (const auto& t1 : terms) {
            for (const auto& t2 : other.terms) {
                if (t1.basis == Basis::One) result.addTerm({t1.coeff * t2.coeff, t2.basis});
                else if (t2.basis == Basis::One) result.addTerm({t1.coeff * t2.coeff, t1.basis});
                else if (t1.basis == t2.basis) {
                    // Approximate Pi*Pi as a scalar to keep it compact
                    result.addTerm({t1.evaluate() * t2.evaluate(), Basis::One});
                }
                else {
                    // Out of basis product - evaluate to One
                    float val = t1.evaluate() * t2.evaluate();
                    result.addTerm({val, Basis::One});
                }
            }
        }
        return result;
    }

    FieldExtension recip() const {
        float v = evaluate();
        if (std::abs(v) < 1e-9) return FieldExtension(NAN);
        return FieldExtension(1.0f / v);
    }

    FieldExtension operator/(const FieldExtension& other) const {
        return (*this) * other.recip();
    }

    friend FieldExtension sin(const FieldExtension& x) { return FieldExtension(std::sin(x.evaluate())); }
    friend FieldExtension cos(const FieldExtension& x) { return FieldExtension(std::cos(x.evaluate())); }
    friend FieldExtension exp(const FieldExtension& x) { return FieldExtension(std::exp(x.evaluate())); }
    friend FieldExtension log(const FieldExtension& x) { return FieldExtension(std::log(x.evaluate())); }
    friend FieldExtension sqrt(const FieldExtension& x) { return FieldExtension(std::sqrt(x.evaluate())); }

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
