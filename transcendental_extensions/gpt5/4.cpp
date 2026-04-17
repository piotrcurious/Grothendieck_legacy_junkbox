#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <string>
#include <regex>
#include <algorithm>

class TranscendentalComplex {
public:
    // Constructor from numeric complex
    TranscendentalComplex(const std::complex<double>& value, const std::string& expr = "")
        : value_(value), expression_(expr.empty() ? toString(value) : expr) {}

    // Named constant constructor
    static TranscendentalComplex constant(const std::string& name, double real, double imag = 0.0) {
        return TranscendentalComplex({real, imag}, name);
    }

    // Arithmetic operators (symbolic + numeric composition)
    TranscendentalComplex operator+(const TranscendentalComplex& other) const {
        return TranscendentalComplex(value_ + other.value_, "(" + expression_ + "+" + other.expression_ + ")");
    }

    TranscendentalComplex operator-(const TranscendentalComplex& other) const {
        return TranscendentalComplex(value_ - other.value_, "(" + expression_ + "-" + other.expression_ + ")");
    }

    TranscendentalComplex operator*(const TranscendentalComplex& other) const {
        return TranscendentalComplex(value_ * other.value_, "(" + expression_ + "*" + other.expression_ + ")");
    }

    TranscendentalComplex operator/(const TranscendentalComplex& other) const {
        return TranscendentalComplex(value_ / other.value_, "(" + expression_ + "/" + other.expression_ + ")");
    }

    TranscendentalComplex operator^(const TranscendentalComplex& other) const {
        return TranscendentalComplex(std::pow(value_, other.value_), expression_ + "^" + other.expression_);
    }

    // Apply morphism rules and pattern match
    TranscendentalComplex simplify(const std::vector<std::pair<std::string, std::string>>& morphisms) const {
        std::string simplifiedExpr = expression_;
        // Remove spaces for more robust matching
        simplifiedExpr.erase(std::remove(simplifiedExpr.begin(), simplifiedExpr.end(), ' '), simplifiedExpr.end());

        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& pair : morphisms) {
                std::regex re(pair.first);
                std::string next = std::regex_replace(simplifiedExpr, re, pair.second);
                if (next != simplifiedExpr) {
                    simplifiedExpr = next;
                    changed = true;
                }
            }
        }
        // After pattern replacement, re-evaluate numeric value
        std::complex<double> eval = evaluateExpression(simplifiedExpr);
        return TranscendentalComplex(eval, simplifiedExpr);
    }

    // Accessors
    std::complex<double> getValue() const { return value_; }
    std::string getExpression() const { return expression_; }

private:
    std::complex<double> value_;
    std::string expression_;

    static std::string toString(const std::complex<double>& v) {
        return std::to_string(v.real()) + (v.imag() >= 0 ? "+" : "") + std::to_string(v.imag()) + "i";
    }

    // Evaluator that can handle numeric constants and some known labels
    static std::complex<double> evaluateExpression(const std::string& expr) {
        if (expr == "pi") return {M_PI, 0};
        if (expr == "e") return {std::exp(1.0), 0};
        if (expr == "-1") return {-1, 0};

        // Try to parse as double
        try {
            size_t pos;
            double val = std::stod(expr, &pos);
            if (pos == expr.length()) return {val, 0};
        } catch (...) {
            // Not a simple double
        }

        // Fallback for demo: if it contains '^', try to evaluate parts (extremely minimal)
        size_t caret = expr.find('^');
        if (caret != std::string::npos) {
            std::complex<double> base = evaluateExpression(expr.substr(0, caret));
            std::complex<double> exp = evaluateExpression(expr.substr(caret + 1));
            return std::pow(base, exp);
        }

        return {0, 0}; // fallback
    }
};

int main() {
    // Define constants
    auto pi = TranscendentalComplex::constant("pi", M_PI);
    auto e  = TranscendentalComplex::constant("e", std::exp(1.0));
    auto i  = TranscendentalComplex::constant("i", 0.0, 1.0);

    // Build expression: e^(i*pi)
    TranscendentalComplex eulerIdentity = e ^ (i * pi);

    // List of morphism rules (regex patterns → replacements)
    // Ordered from most specific to least specific
    std::vector<std::pair<std::string, std::string>> morphisms = {
        {R"(e\^\(i\*pi\))", "-1"},
        {R"(e\^i\*pi)", "-1"},
        {R"(\(i\*pi\))", "i*pi"},
        {R"(sin\(0\))", "0"},
        {R"(cos\(0\))", "1"},
        {R"(exp\(0\))", "1"},
        {R"(pi)", "3.14159265358979323846"},
        {R"(e)", "2.71828182845904523536"}
    };

    // Apply simplification (pattern match + replace)
    auto simplified = eulerIdentity.simplify(morphisms);

    std::cout << "Original: " << eulerIdentity.getExpression() << " = " << eulerIdentity.getValue() << "\n";
    std::cout << "Simplified: " << simplified.getExpression() << " = " << simplified.getValue() << "\n";
}
