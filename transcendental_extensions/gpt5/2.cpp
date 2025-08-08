#include <iostream>
#include <complex>
#include <memory>
#include <string>
#include <map>
#include <cmath>
#include <sstream>

class TranscendentalComplex {
public:
    enum class NodeType { CONST, ADD, SUB, MUL, DIV, UNARY };

    using MorphismMap = std::map<std::string, std::function<std::complex<double>(const std::complex<double>&)>>;

    static MorphismMap& morphisms() {
        static MorphismMap morph = {
            {"sin", [](const std::complex<double>& z){ return std::sin(z); }},
            {"cos", [](const std::complex<double>& z){ return std::cos(z); }},
            {"exp", [](const std::complex<double>& z){ return std::exp(z); }},
            {"log", [](const std::complex<double>& z){ return std::log(z); }},
            {"neg", [](const std::complex<double>& z){ return -z; }}
        };
        return morph;
    }

    // Constants constructor
    TranscendentalComplex(const std::complex<double>& val, std::string label = "")
        : type_(NodeType::CONST), value_(val), label_(std::move(label)) {}

    // Operator overloads (build symbolic tree)
    TranscendentalComplex operator+(const TranscendentalComplex& other) const {
        return makeBinary(NodeType::ADD, *this, other);
    }
    TranscendentalComplex operator-(const TranscendentalComplex& other) const {
        return makeBinary(NodeType::SUB, *this, other);
    }
    TranscendentalComplex operator*(const TranscendentalComplex& other) const {
        return makeBinary(NodeType::MUL, *this, other);
    }
    TranscendentalComplex operator/(const TranscendentalComplex& other) const {
        return makeBinary(NodeType::DIV, *this, other);
    }

    // Apply a named morphism (sin, cos, exp...)
    static TranscendentalComplex morphism(const std::string& name, const TranscendentalComplex& arg) {
        if (!morphisms().count(name))
            throw std::runtime_error("Unknown morphism: " + name);
        TranscendentalComplex node(NodeType::UNARY, name);
        node.left_ = std::make_shared<TranscendentalComplex>(arg);
        return node;
    }

    // Numeric evaluation
    std::complex<double> getValue() const {
        switch (type_) {
            case NodeType::CONST: return value_;
            case NodeType::ADD: return left_->getValue() + right_->getValue();
            case NodeType::SUB: return left_->getValue() - right_->getValue();
            case NodeType::MUL: return left_->getValue() * right_->getValue();
            case NodeType::DIV: return left_->getValue() / right_->getValue();
            case NodeType::UNARY: return morphisms().at(label_)(left_->getValue());
        }
        return {};
    }

    // Pretty-print symbolic expression
    std::string toString() const {
        switch (type_) {
            case NodeType::CONST:
                return label_.empty() ? complexToStr(value_) : label_;
            case NodeType::ADD:
                return "(" + left_->toString() + " + " + right_->toString() + ")";
            case NodeType::SUB:
                return "(" + left_->toString() + " - " + right_->toString() + ")";
            case NodeType::MUL:
                return "(" + left_->toString() + " * " + right_->toString() + ")";
            case NodeType::DIV:
                return "(" + left_->toString() + " / " + right_->toString() + ")";
            case NodeType::UNARY:
                return label_ + "(" + left_->toString() + ")";
        }
        return {};
    }

private:
    NodeType type_;
    std::complex<double> value_{};
    std::string label_;
    std::shared_ptr<TranscendentalComplex> left_{}, right_{};

    TranscendentalComplex(NodeType t, const std::string& lbl = "")
        : type_(t), label_(lbl) {}

    static TranscendentalComplex makeBinary(NodeType type,
                                            const TranscendentalComplex& lhs,
                                            const TranscendentalComplex& rhs) {
        TranscendentalComplex node(type);
        node.left_  = std::make_shared<TranscendentalComplex>(lhs);
        node.right_ = std::make_shared<TranscendentalComplex>(rhs);
        return node;
    }

    static std::string complexToStr(const std::complex<double>& c) {
        std::ostringstream oss;
        oss << c;
        return oss.str();
    }
};

int main() {
    using TC = TranscendentalComplex;

    // Exact symbolic constants
    TC pi({M_PI, 0}, "π");
    TC e ({std::exp(1.0), 0}, "e");
    TC i ({0, 1}, "i");

    // Build symbolic expressions
    auto expr1 = TC::morphism("sin", pi / e);
    auto expr2 = TC::morphism("exp", i * pi);
    auto expr3 = (expr1 + e) * expr2;

    // Print symbolic form
    std::cout << "expr1: " << expr1.toString() << "\n";
    std::cout << "expr2: " << expr2.toString() << "\n";
    std::cout << "expr3: " << expr3.toString() << "\n";

    // Evaluate
    std::cout << "expr1 value = " << expr1.getValue() << "\n";
    std::cout << "expr2 value = " << expr2.getValue() << "\n";
    std::cout << "expr3 value = " << expr3.getValue() << "\n";
}
