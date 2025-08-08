/*
 * transcendental_symbolic.cpp
 * Build with: g++ -std=c++17 transcendental_symbolic.cpp -o transcendental_symbolic -lm
 *
 * Goal: lightweight symbolic complex expression system with rewrite rules (morphisms map + simplifier).
 */

#include <iostream>
#include <memory>
#include <complex>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <cassert>

using cplx = std::complex<double>;
constexpr double EPS = 1e-12;

static inline bool approxZero(const cplx& z) {
    return std::abs(z.real()) + std::abs(z.imag()) < 1e-12;
}

struct Expr;
using ExprPtr = std::shared_ptr<Expr>;
using RewriteFn = std::function<ExprPtr(const ExprPtr&)>; // returns new node if rewrote, or same node if no change

enum class NodeKind { CONST, VAR, ADD, SUB, MUL, DIV, POW, UNARY };

struct Expr {
    NodeKind kind;
    std::string label;      // used for CONST (symbolic label like "π") or VAR name or UNARY function name
    cplx const_value{};     // numeric value (for CONST nodes)
    ExprPtr left{}, right{}; // children (binary ops or unary: left used for argument)
    // Constructors
    static ExprPtr makeConst(cplx val, std::string lbl = "") {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::CONST;
        p->const_value = val;
        p->label = std::move(lbl);
        return p;
    }
    static ExprPtr makeVar(const std::string& name) {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::VAR;
        p->label = name;
        return p;
    }
    static ExprPtr makeUnary(const std::string& fname, ExprPtr arg) {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::UNARY;
        p->label = fname;
        p->left = std::move(arg);
        return p;
    }
    static ExprPtr makeBinary(NodeKind k, ExprPtr a, ExprPtr b) {
        assert(k == NodeKind::ADD || k == NodeKind::SUB || k == NodeKind::MUL || k == NodeKind::DIV || k == NodeKind::POW);
        auto p = std::make_shared<Expr>();
        p->kind = k;
        p->left = std::move(a);
        p->right = std::move(b);
        return p;
    }
};

// utility: convert complex to string
static std::string cplxToStr(const cplx& z) {
    std::ostringstream oss;
    oss << std::setprecision(12) << "(" << z.real() << (z.imag() >= 0 ? "+" : "") << z.imag() << "i)";
    return oss.str();
}

// pretty printer
std::string toString(const ExprPtr& e) {
    if (!e) return "null";
    switch (e->kind) {
        case NodeKind::CONST:
            if (!e->label.empty()) return e->label;
            return cplxToStr(e->const_value);
        case NodeKind::VAR:
            return e->label;
        case NodeKind::UNARY:
            return e->label + "(" + toString(e->left) + ")";
        case NodeKind::ADD:
            return "(" + toString(e->left) + " + " + toString(e->right) + ")";
        case NodeKind::SUB:
            return "(" + toString(e->left) + " - " + toString(e->right) + ")";
        case NodeKind::MUL:
            return "(" + toString(e->left) + " * " + toString(e->right) + ")";
        case NodeKind::DIV:
            return "(" + toString(e->left) + " / " + toString(e->right) + ")";
        case NodeKind::POW:
            return "(" + toString(e->left) + " ^ " + toString(e->right) + ")";
    }
    return "?";
}

// morphism registry: name -> evaluator (complex -> complex)
using MorphMap = std::map<std::string, std::function<cplx(const cplx&)>>;

MorphMap& morphisms() {
    static MorphMap mm = {
        {"sin", [](const cplx& z){ return std::sin(z); }},
        {"cos", [](const cplx& z){ return std::cos(z); }},
        {"exp", [](const cplx& z){ return std::exp(z); }},
        {"log", [](const cplx& z){ return std::log(z); }},
        {"sqrt",[](const cplx& z){ return std::sqrt(z); }},
        {"neg", [](const cplx& z){ return -z; }},
        // add more as needed
    };
    return mm;
}

// Helper to test if node is constant (symbol or numeric)
bool isConst(const ExprPtr& e) {
    return e && e->kind == NodeKind::CONST;
}

// Evaluate an Expr to a numeric complex value (resolves known labels like "π", "e", "i")
cplx evaluateNumeric(const ExprPtr& e);

// map known symbolic constants to numeric values
static inline bool tryLabelToValue(const std::string& lbl, cplx& out) {
    if (lbl == "π" || lbl == "pi") { out = cplx(M_PI, 0); return true; }
    if (lbl == "e") { out = cplx(std::exp(1.0), 0); return true; }
    if (lbl == "i") { out = cplx(0, 1); return true; }
    if (lbl == "0") { out = cplx(0,0); return true; }
    if (lbl == "1") { out = cplx(1,0); return true; }
    return false;
}

cplx evaluateNumeric(const ExprPtr& e) {
    if (!e) return cplx(0,0);
    switch (e->kind) {
        case NodeKind::CONST: {
            if (!e->label.empty()) {
                cplx v;
                if (tryLabelToValue(e->label, v)) return v;
                // if label unknown, fall back to provided numeric value
            }
            return e->const_value;
        }
        case NodeKind::VAR:
            throw std::runtime_error("Cannot numerically evaluate variable: " + e->label);
        case NodeKind::UNARY: {
            auto arg = evaluateNumeric(e->left);
            auto it = morphisms().find(e->label);
            if (it == morphisms().end()) throw std::runtime_error("Unknown morphism: " + e->label);
            return it->second(arg);
        }
        case NodeKind::ADD:
            return evaluateNumeric(e->left) + evaluateNumeric(e->right);
        case NodeKind::SUB:
            return evaluateNumeric(e->left) - evaluateNumeric(e->right);
        case NodeKind::MUL:
            return evaluateNumeric(e->left) * evaluateNumeric(e->right);
        case NodeKind::DIV:
            return evaluateNumeric(e->left) / evaluateNumeric(e->right);
        case NodeKind::POW: {
            cplx a = evaluateNumeric(e->left);
            cplx b = evaluateNumeric(e->right);
            // basic pow via std::pow
            return std::pow(a, b);
        }
    }
    return cplx(0,0);
}

// helpers to clone or return same pointer
static inline ExprPtr unchanged(const ExprPtr& n) { return n; }
static inline ExprPtr makeZero() { return Expr::makeConst(cplx(0,0), "0"); }
static inline ExprPtr makeOne()  { return Expr::makeConst(cplx(1,0), "1"); }

// Utilities to check if node is numeric constant approx zero/one
bool isNumericZero(const ExprPtr& n) {
    if (!n) return false;
    if (n->kind != NodeKind::CONST) return false;
    cplx val = n->const_value;
    if (!n->label.empty()) {
        if (tryLabelToValue(n->label, val)) {
            return approxZero(val);
        }
    }
    return approxZero(val);
}
bool isNumericOne(const ExprPtr& n) {
    if (!n) return false;
    if (n->kind != NodeKind::CONST) return false;
    cplx val = n->const_value;
    if (!n->label.empty()) {
        if (tryLabelToValue(n->label, val)) {
            return std::abs(val.real() - 1.0) + std::abs(val.imag()) < 1e-12;
        }
    }
    return std::abs(val.real() - 1.0) + std::abs(val.imag()) < 1e-12;
}

// Constant folding: if both children are numeric constants -> compute numeric and return CONST
ExprPtr tryConstantFoldBinary(NodeKind k, const ExprPtr& a, const ExprPtr& b) {
    if (!a || !b) return nullptr;
    if (a->kind == NodeKind::CONST && b->kind == NodeKind::CONST) {
        // try to map labels to numeric if present
        cplx A = a->const_value;
        cplx B = b->const_value;
        if (!a->label.empty()) tryLabelToValue(a->label, A);
        if (!b->label.empty()) tryLabelToValue(b->label, B);
        cplx res;
        switch (k) {
            case NodeKind::ADD: res = A + B; break;
            case NodeKind::SUB: res = A - B; break;
            case NodeKind::MUL: res = A * B; break;
            case NodeKind::DIV: res = A / B; break;
            case NodeKind::POW: res = std::pow(A, B); break;
            default: return nullptr;
        }
        // produce a CONST node with numeric value. We omit label to indicate numeric folded value.
        return Expr::makeConst(res, "");
    }
    return nullptr;
}

// ----------------------- Rewrite rules -----------------------
// Each rewrite rule examines a node and returns a new node (or the same node if no rewrite).
// Order matters: simpler algebraic reductions first, then pattern identities (like exp(i*pi) -> -1).

// Rule: x + 0 -> x
ExprPtr rule_add_zero(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::ADD) return n;
    if (isNumericZero(n->left)) return n->right;
    if (isNumericZero(n->right)) return n->left;
    // constant fold
    if (auto folded = tryConstantFoldBinary(NodeKind::ADD, n->left, n->right)) return folded;
    return n;
}
// Rule: x - 0 -> x, 0 - x -> -x
ExprPtr rule_sub_zero(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::SUB) return n;
    if (isNumericZero(n->right)) return n->left;
    if (isNumericZero(n->left)) {
        // 0 - x -> -x  => unary neg
        return Expr::makeUnary("neg", n->right);
    }
    if (auto folded = tryConstantFoldBinary(NodeKind::SUB, n->left, n->right)) return folded;
    return n;
}
// Rule: x * 1 -> x ; 1 * x -> x ; x * 0 -> 0
ExprPtr rule_mul_identity(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::MUL) return n;
    if (isNumericOne(n->left)) return n->right;
    if (isNumericOne(n->right)) return n->left;
    if (isNumericZero(n->left) || isNumericZero(n->right)) return makeZero();
    if (auto folded = tryConstantFoldBinary(NodeKind::MUL, n->left, n->right)) return folded;
    return n;
}
// Rule: x / 1 -> x ; 0 / x -> 0
ExprPtr rule_div_identity(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::DIV) return n;
    if (isNumericOne(n->right)) return n->left;
    if (isNumericZero(n->left)) return makeZero();
    if (auto folded = tryConstantFoldBinary(NodeKind::DIV, n->left, n->right)) return folded;
    return n;
}
// Rule: pow folding if both consts
ExprPtr rule_pow_fold(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::POW) return n;
    if (auto folded = tryConstantFoldBinary(NodeKind::POW, n->left, n->right)) return folded;
    return n;
}
// Rule: unary function applied to constant -> numeric fold (sin(0)->0, cos(0)->1, log(exp(x))->x handled separately)
ExprPtr rule_unary_const_fold(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::UNARY) return n;
    if (n->left && n->left->kind == NodeKind::CONST) {
        cplx value = n->left->const_value;
        if (!n->left->label.empty()) tryLabelToValue(n->left->label, value);
        auto it = morphisms().find(n->label);
        if (it == morphisms().end()) return n; // unknown morphism
        cplx res = it->second(value);
        // If result matches a known label exactly, we could assign label; we keep numeric folded value.
        return Expr::makeConst(res, "");
    }
    return n;
}

// Rule: log(exp(x)) -> x
ExprPtr rule_log_exp(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::UNARY) return n;
    if (n->label == "log" && n->left && n->left->kind == NodeKind::UNARY && n->left->label == "exp") {
        // return inner arg of exp
        return n->left->left;
    }
    return n;
}
// Rule: exp(log(x)) -> x  (careful domain; assume principal branch)
ExprPtr rule_exp_log(const ExprPtr& n) {
    if (!n) return n;
    // either UNARY exp(log(x)) or POW(e, log(x))
    if (n->kind == NodeKind::UNARY && n->label == "exp" && n->left && n->left->kind == NodeKind::UNARY && n->left->label == "log") {
        return n->left->left;
    }
    // POW(e, log(x)) -> x
    if (n->kind == NodeKind::POW && n->left && n->right) {
        // left equals e?
        bool leftIsE = false;
        if (n->left->kind == NodeKind::CONST && !n->left->label.empty()) {
            cplx v; leftIsE = tryLabelToValue(n->left->label, v) && std::abs(v.real() - std::exp(1.0)) < 1e-12;
        }
        if (leftIsE && n->right->kind == NodeKind::UNARY && n->right->label == "log") {
            return n->right->left;
        }
    }
    return n;
}

// Rule: exp(i*pi) -> -1   (Euler identity)
ExprPtr rule_euler_identity(const ExprPtr& n) {
    // matches UNARY exp where argument is MUL(i, π) or MUL(π, i)
    if (!n) return n;
    if (n->kind == NodeKind::UNARY && n->label == "exp" && n->left) {
        auto arg = n->left;
        if (arg->kind == NodeKind::MUL) {
            // check children
            auto a = arg->left;
            auto b = arg->right;
            bool a_is_i = (a->kind == NodeKind::CONST && a->label == "i");
            bool b_is_i = (b->kind == NodeKind::CONST && b->label == "i");
            bool a_is_pi = (a->kind == NodeKind::CONST && (a->label == "π" || a->label == "pi"));
            bool b_is_pi = (b->kind == NodeKind::CONST && (b->label == "π" || b->label == "pi"));
            if ((a_is_i && b_is_pi) || (b_is_i && a_is_pi)) {
                // exp(i*pi) == -1
                return Expr::makeConst(cplx(-1,0), "-1");
            }
        }
    }
    return n;
}

// Rule: sin(0)->0, cos(0)->1
ExprPtr rule_trig_zero(const ExprPtr& n) {
    if (!n || n->kind != NodeKind::UNARY) return n;
    if (n->left && isNumericZero(n->left)) {
        if (n->label == "sin") return makeZero();
        if (n->label == "cos") return makeOne();
    }
    return n;
}

// Generic rewrite that first attempts to rewrite children (recursively), then applies rules to current node.
ExprPtr rewriteOnce(const ExprPtr& node, const std::vector<RewriteFn>& rules);

// rewrite children recursively and build new node if children changed
ExprPtr rewriteChildren(const ExprPtr& n, const std::vector<RewriteFn>& rules) {
    if (!n) return n;
    // For CONST and VAR, nothing to recurse into
    if (n->kind == NodeKind::CONST || n->kind == NodeKind::VAR) return n;
    if (n->kind == NodeKind::UNARY) {
        ExprPtr left2 = rewriteOnce(n->left, rules);
        if (left2 != n->left) {
            return Expr::makeUnary(n->label, left2);
        }
        return n;
    }
    // binary
    ExprPtr L = rewriteOnce(n->left, rules);
    ExprPtr R = rewriteOnce(n->right, rules);
    if (L != n->left || R != n->right) {
        return Expr::makeBinary(n->kind, L, R);
    }
    return n;
}

ExprPtr rewriteOnce(const ExprPtr& node, const std::vector<RewriteFn>& rules) {
    if (!node) return node;
    // First rewrite children
    ExprPtr cur = rewriteChildren(node, rules);
    // Then attempt each rule in order, returning first successful rewrite (different pointer)
    for (const auto& r : rules) {
        ExprPtr next = r(cur);
        if (next != cur) {
            return next;
        }
    }
    return cur;
}

// Apply rewrite passes until fixed point or max iterations
ExprPtr simplify(const ExprPtr& root, const std::vector<RewriteFn>& rules, int max_passes = 50) {
    ExprPtr cur = root;
    for (int pass = 0; pass < max_passes; ++pass) {
        ExprPtr next = rewriteOnce(cur, rules);
        // repeat rewriteOnce on whole tree until no change (we need to fully traverse repeatedly)
        // But rewriteOnce only rewrites at one location; we build a full fixed point loop:
        // Repeatedly apply rewriteOnce globally until unchanged
        bool changed_any = false;
        // We'll perform a full-tree rewrite by calling rewriteOnce until no change.
        while (true) {
            ExprPtr candidate = rewriteOnce(cur, rules);
            if (candidate == cur) break;
            cur = candidate;
            changed_any = true;
        }
        if (!changed_any) break;
    }
    return cur;
}

// A small helper: apply simplifier pass with many rules: order matters
std::vector<RewriteFn> makeDefaultRules() {
    std::vector<RewriteFn> rules;
    // Local algebraic rules first
    rules.push_back(rule_add_zero);
    rules.push_back(rule_sub_zero);
    rules.push_back(rule_mul_identity);
    rules.push_back(rule_div_identity);
    rules.push_back(rule_pow_fold);
    rules.push_back(rule_unary_const_fold);
    // trig/log/exp simplifications
    rules.push_back(rule_trig_zero);
    rules.push_back(rule_log_exp);
    rules.push_back(rule_exp_log);
    rules.push_back(rule_euler_identity);
    // you can append more sophisticated pattern rewrites here
    return rules;
}

// ---------------- example usage ----------------

int main() {
    // Build some symbolic constants and variables
    auto PI = Expr::makeConst(cplx(0,0), "π");
    auto E  = Expr::makeConst(cplx(0,0), "e");
    auto I  = Expr::makeConst(cplx(0,1), "i");
    // also labels for 0,1
    auto ZERO = Expr::makeConst(cplx(0,0), "0");
    auto ONE  = Expr::makeConst(cplx(1,0), "1");

    // Example 1: exp(i * π) -> should simplify to -1
    auto euler = Expr::makeUnary("exp", Expr::makeBinary(NodeKind::MUL, I, PI));

    // Example 2: log(exp(x)) -> x
    auto x = Expr::makeVar("x");
    auto logexp = Expr::makeUnary("log", Expr::makeUnary("exp", x));

    // Example 3: complex folding: (pi + e) * 1  -> (π + e)
    auto sum_pi_e = Expr::makeBinary(NodeKind::ADD, PI, E);
    auto times_one  = Expr::makeBinary(NodeKind::MUL, sum_pi_e, ONE);

    // Example 4: exp(log( (e) ^ (log(y)) )) -> should reduce: pow(e, log(y)) -> y ; then exp(log(y)) -> y
    auto y = Expr::makeVar("y");
    auto logy = Expr::makeUnary("log", y);
    auto pow_e_logy = Expr::makeBinary(NodeKind::POW, E, logy);
    auto full = Expr::makeUnary("exp", Expr::makeUnary("log", pow_e_logy)); // exp(log(e ^ log(y)))

    // Example 5: sin(0) -> 0
    auto sin0 = Expr::makeUnary("sin", ZERO);

    // Compose examples
    std::vector<std::pair<std::string, ExprPtr>> examples = {
        {"Euler", euler},
        {"LogExp", logexp},
        {"TimesOne", times_one},
        {"PowExpLog", full},
        {"SinZero", sin0}
    };

    auto rules = makeDefaultRules();

    for (auto &ex : examples) {
        std::cout << "---- " << ex.first << " ----\n";
        std::cout << "Original:   " << toString(ex.second) << "\n";
        ExprPtr simp = simplify(ex.second, rules, 20);
        std::cout << "Simplified: " << toString(simp) << "\n";
        // attempt numeric evaluation if no variables present
        bool hasVar = false;
        std::function<void(const ExprPtr&)> checkVar = [&](const ExprPtr& n){
            if (!n || hasVar) return;
            if (n->kind == NodeKind::VAR) { hasVar = true; return; }
            if (n->left) checkVar(n->left);
            if (n->right) checkVar(n->right);
        };
        checkVar(simp);
        if (!hasVar) {
            try {
                cplx val = evaluateNumeric(simp);
                std::cout << "Numeric:    " << cplxToStr(val) << "\n";
            } catch (const std::exception& e) {
                std::cout << "Numeric eval failed: " << e.what() << "\n";
            }
        } else {
            std::cout << "Numeric:    contains variables -> skip numeric evaluation\n";
        }
        std::cout << "\n";
    }

    // Demonstrate more: simplify (pi + e) * 1, evaluate numeric approx
    auto simp2 = simplify(times_one, rules);
    std::cout << "Extra: " << toString(times_one) << " -> " << toString(simp2) << "\n";
    try {
        // create a numeric fold by converting pi and e labels to numeric when evaluating
        cplx val = evaluateNumeric(simp2);
        std::cout << "Numeric value: " << cplxToStr(val) << "\n";
    } catch (...) {
        std::cout << "Numeric evaluation error\n";
    }

    return 0;
}
