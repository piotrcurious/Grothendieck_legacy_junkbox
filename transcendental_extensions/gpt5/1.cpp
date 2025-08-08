#include <iostream>
#include <complex>
#include <memory>
#include <cmath>
#include <string>
#include <sstream>

class TranscendentalComplex {
public:
    enum class OpType { CONST, ADD, SUB, MUL, DIV };

    // Constructor for constant value
    explicit TranscendentalComplex(const std::complex<double>& val, std::string label = "")
        : op_(OpType::CONST), value_(val), label_(std::move(label)) {}

    // Operator overloads build new expression nodes
    TranscendentalComplex operator+(const TranscendentalComplex& other) const {
        return TranscendentalComplex(OpType::ADD, *this, other);
    }

    TranscendentalComplex operator-(const TranscendentalComplex& other) const {
        return TranscendentalComplex(OpType::SUB, *this, other);
    }

    TranscendentalComplex operator*(const TranscendentalComplex& other) const {
        return TranscendentalComplex(OpType::MUL, *this, other);
    }

    TranscendentalComplex operator/(const TranscendentalComplex& other) const {
        return TranscendentalComplex(OpType::DIV, *this, other);
    }

    // Evaluate recursively
    std::complex<double> getValue() const {
        switch (op_) {
            case OpType::CONST: return value_;
            case OpType::ADD: return left_->getValue() + right_->getValue();
            case OpType::SUB: return left_->getValue() - right_->getValue();
            case OpType::MUL: return left_->getValue() * right_->getValue();
            case OpType::DIV: return left_->getValue() / right_->getValue();
        }
        return {};
    }

    // Pretty-print symbolic expression
    std::string toString() const {
        switch (op_) {
            case OpType::CONST: return label_.empty() ? toStringValue(value_) : label_;
            case OpType::ADD: return "(" + left_->toString() + " + " + right_->toString() + ")";
            case OpType::SUB: return "(" + left_->toString() + " - " + right_->toString() + ")";
            case OpType::MUL: return "(" + left_->toString() + " * " + right_->toString() + ")";
            case OpType::DIV: return "(" + left_->toString() + " / " + right_->toString() + ")";
        }
        return {};
    }

private:
    OpType op_;
    std::complex<double> value_{};
    std::string label_;

    std::shared_ptr<TranscendentalComplex> left_, right_;

    // Constructor for operations
    TranscendentalComplex(OpType op, const TranscendentalComplex& lhs, const TranscendentalComplex& rhs)
        : op_(op), left_(std::make_shared<TranscendentalComplex>(lhs)),
          right_(std::make_shared<TranscendentalComplex>(rhs)) {}

    static std::string toStringValue(const std::complex<double>& c) {
        std::ostringstream oss;
        oss << c;
        return oss.str();
    }
};

int main() {
    // Symbolic constants
    TranscendentalComplex pi({M_PI, 0}, "π");
    TranscendentalComplex e ({std::exp(1.0), 0}, "e");

    // Build symbolic expressions
    auto sum     = pi + e;
    auto product = pi * e;
    auto fancy   = (pi + e) * (pi - e) / e;

    // Show symbolic form
    std::cout << "Symbolic sum:     " << sum.toString() << "\n";
    std::cout << "Symbolic product: " << product.toString() << "\n";
    std::cout << "Fancy expression: " << fancy.toString() << "\n\n";

    // Evaluate numerically
    std::cout << "pi + e     = " << sum.getValue() << "\n";
    std::cout << "pi * e     = " << product.getValue() << "\n";
    std::cout << "fancy expr = " << fancy.getValue() << "\n";
}
