/*
 symbolic_patterns.cpp
 C++17 single-file: tree-based symbolic expressions + pattern matching + rewrite rules

 Build:
   g++ -std=c++17 symbolic_patterns.cpp -o symbolic_patterns -lm

*/

#include <iostream>
#include <memory>
#include <vector>
#include <map>
#include <functional>
#include <complex>
#include <string>
#include <sstream>
#include <cassert>
#include <iomanip>

using cplx = std::complex<double>;
constexpr double EPS = 1e-12;

enum class NodeKind { CONST, VAR, UNARY, ADD, SUB, MUL, DIV, POW };

// Forward
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

// Expression node
struct Expr {
    NodeKind kind;
    // For CONST and VAR and UNARY label; for BINARY nodes label unused
    std::string label;
    // numeric value (only used for CONST nodes if numeric provided)
    cplx value{0,0};
    // children: left used for unary/binary first arg; right for second child when binary
    ExprPtr left{};
    ExprPtr right{};

    // Constructors
    static ExprPtr Const(cplx val, std::string lbl = "") {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::CONST;
        p->value = val;
        p->label = std::move(lbl);
        return p;
    }
    static ExprPtr Var(const std::string& name) {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::VAR;
        p->label = name;
        return p;
    }
    static ExprPtr Unary(const std::string& fname, ExprPtr arg) {
        auto p = std::make_shared<Expr>();
        p->kind = NodeKind::UNARY;
        p->label = fname;
        p->left = std::move(arg);
        return p;
    }
    static ExprPtr Binary(NodeKind k, ExprPtr a, ExprPtr b) {
        assert(k==NodeKind::ADD || k==NodeKind::SUB || k==NodeKind::MUL || k==NodeKind::DIV || k==NodeKind::POW);
        auto p = std::make_shared<Expr>();
        p->kind = k;
        p->left = std::move(a);
        p->right = std::move(b);
        return p;
    }
};

// convenience factory functions
inline ExprPtr C(double r, double im=0.0, const std::string& lbl="") { return Expr::Const(cplx(r,im), lbl); }
inline ExprPtr PI() { return Expr::Const(cplx(0,0), "π"); }
inline ExprPtr E()  { return Expr::Const(cplx(0,0), "e"); }
inline ExprPtr I()  { return Expr::Const(cplx(0,1), "i"); }
inline ExprPtr ZERO(){ return Expr::Const(cplx(0,0), "0"); }
inline ExprPtr ONE(){ return Expr::Const(cplx(1,0), "1"); }
inline ExprPtr V(const std::string& n){ return Expr::Var(n); }
inline ExprPtr U(const std::string& f, ExprPtr a){ return Expr::Unary(f, a); }
inline ExprPtr B(NodeKind k, ExprPtr a, ExprPtr b){ return Expr::Binary(k, a, b); }

// Pretty printer
std::string toString(const ExprPtr& e) {
    if (!e) return "null";
    std::ostringstream oss;
    switch (e->kind) {
        case NodeKind::CONST:
            if (!e->label.empty()) return e->label;
            oss << std::setprecision(12) << "(" << e->value.real() << (e->value.imag()>=0 ? "+" : "") << e->value.imag() << "i)";
            return oss.str();
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

// Morphism registry - unary evaluators
using MorphMap = std::map<std::string, std::function<cplx(const cplx&)>>;
MorphMap& morphisms() {
    static MorphMap mm = {
        {"sin", [](const cplx& z){ return std::sin(z); }},
        {"cos", [](const cplx& z){ return std::cos(z); }},
        {"exp", [](const cplx& z){ return std::exp(z); }},
        {"log", [](const cplx& z){ return std::log(z); }},
        {"sqrt",[](const cplx& z){ return std::sqrt(z); }},
        {"neg", [](const cplx& z){ return -z; }}
    };
    return mm;
}

// label -> numeric mapping for common constants
bool labelToValue(const std::string& lbl, cplx &out) {
    if (lbl == "π" || lbl == "pi") { out = cplx(M_PI,0); return true; }
    if (lbl == "e") { out = cplx(std::exp(1.0),0); return true; }
    if (lbl == "i") { out = cplx(0,1); return true; }
    if (lbl == "0") { out = cplx(0,0); return true; }
    if (lbl == "1") { out = cplx(1,0); return true; }
    return false;
}

// Numeric evaluation of an expression (throws on free variables)
cplx evaluateNumeric(const ExprPtr& e) {
    if (!e) return cplx(0,0);
    switch (e->kind) {
        case NodeKind::CONST: {
            if (!e->label.empty()) {
                cplx v;
                if (labelToValue(e->label, v)) return v;
            }
            return e->value;
        }
        case NodeKind::VAR:
            throw std::runtime_error("Cannot numerically evaluate variable: " + e->label);
        case NodeKind::UNARY: {
            cplx a = evaluateNumeric(e->left);
            auto it = morphisms().find(e->label);
            if (it == morphisms().end()) throw std::runtime_error("Unknown morphism: " + e->label);
            return it->second(a);
        }
        case NodeKind::ADD:
            return evaluateNumeric(e->left) + evaluateNumeric(e->right);
        case NodeKind::SUB:
            return evaluateNumeric(e->left) - evaluateNumeric(e->right);
        case NodeKind::MUL:
            return evaluateNumeric(e->left) * evaluateNumeric(e->right);
        case NodeKind::DIV:
            return evaluateNumeric(e->left) / evaluateNumeric(e->right);
        case NodeKind::POW:
            return std::pow(evaluateNumeric(e->left), evaluateNumeric(e->right));
    }
    return cplx(0,0);
}

// -------- Pattern matching --------
// Bindings: map pattern variable name (without '?') -> matched ExprPtr
using Bindings = std::map<std::string, ExprPtr>;

// Try to match pattern p against subject s. On success, returns true and fills bindings.
// Pattern language: pattern nodes whose label starts with '?' are wildcards that bind the entire subtree.
// Example pattern: exp( ?x ) would be represented as Expr::Unary("exp", Var("?x")).
bool match(const ExprPtr& pattern, const ExprPtr& subject, Bindings& out) {
    if (!pattern || !subject) return pattern==subject;
    // wildcard variable: for CONST/VAR/UNARY etc if pattern->label starts with '?'
    auto isWildcard = [&](const ExprPtr& node)->bool {
        return (node->kind == NodeKind::VAR || node->kind == NodeKind::CONST) && !node->label.empty() && node->label[0]=='?';
    };
    if (isWildcard(pattern)) {
        std::string key = pattern->label.substr(1);
        // if already bound, check equality with current subject
        auto it = out.find(key);
        if (it != out.end()) {
            // pointer-equality or structural equality? We'll check structural string form equality for simplicity
            return toString(it->second) == toString(subject);
        } else {
            out[key] = subject;
            return true;
        }
    }
    // otherwise kinds must match
    if (pattern->kind != subject->kind) return false;
    switch (pattern->kind) {
        case NodeKind::CONST: {
            // if pattern has label (like "π") treat as name match; if not, compare numeric
            if (!pattern->label.empty()) {
                return pattern->label == subject->label;
            }
            // numeric compare
            return std::abs(pattern->value.real() - subject->value.real()) < EPS &&
                   std::abs(pattern->value.imag() - subject->value.imag()) < EPS;
        }
        case NodeKind::VAR:
            return pattern->label == subject->label;
        case NodeKind::UNARY:
            if (pattern->label != subject->label) return false;
            return match(pattern->left, subject->left, out);
        case NodeKind::ADD:
        case NodeKind::SUB:
        case NodeKind::MUL:
        case NodeKind::DIV:
        case NodeKind::POW:
            return match(pattern->left, subject->left, out) && match(pattern->right, subject->right, out);
    }
    return false;
}

// Build replacement tree by substituting wildcard bindings. If a replacement subtree contains pattern variable Var("?x"),
// we replace it by the binding out["x"].
ExprPtr substitute(const ExprPtr& repl, const Bindings& out) {
    if (!repl) return nullptr;
    // if repl is wildcard pattern variable, return binding
    if ((repl->kind == NodeKind::VAR || repl->kind == NodeKind::CONST) && !repl->label.empty() && repl->label[0]=='?') {
        std::string key = repl->label.substr(1);
        auto it = out.find(key);
        if (it != out.end()) return it->second;
        // unbound variable -> keep as-is (could also throw)
        return repl;
    }
    // otherwise recursively clone and substitute children
    if (repl->kind == NodeKind::CONST) {
        return Expr::Const(repl->value, repl->label);
    } else if (repl->kind == NodeKind::VAR) {
        return Expr::Var(repl->label);
    } else if (repl->kind == NodeKind::UNARY) {
        return Expr::Unary(repl->label, substitute(repl->left, out));
    } else {
        return Expr::Binary(repl->kind, substitute(repl->left, out), substitute(repl->right, out));
    }
}

// Search tree for first location where pattern matches; if found returns replaced subtree (not the whole tree).
// We implement a rewrite that attempts to match at the current node; if matched, returns the substituted node.
// Otherwise recursively attempt on children and rebuild node if a child changed.
ExprPtr applyRuleOnceAtNode(const ExprPtr& node, const ExprPtr& pattern, const ExprPtr& replacement) {
    if (!node) return node;
    Bindings b;
    if (match(pattern, node, b)) {
        // apply substitution
        return substitute(replacement, b);
    }
    // recurse into children and rebuild if necessary
    if (node->kind == NodeKind::UNARY) {
        ExprPtr newLeft = applyRuleOnceAtNode(node->left, pattern, replacement);
        if (newLeft != node->left) return Expr::Unary(node->label, newLeft);
        return node;
    } else if (node->kind==NodeKind::CONST || node->kind==NodeKind::VAR) {
        return node;
    } else {
        ExprPtr L = applyRuleOnceAtNode(node->left, pattern, replacement);
        if (L != node->left) return Expr::Binary(node->kind, L, node->right);
        ExprPtr R = applyRuleOnceAtNode(node->right, pattern, replacement);
        if (R != node->right) return Expr::Binary(node->kind, node->left, R);
        return node;
    }
}

// Top-level rewrite: apply list of rules (pattern->replacement) repeatedly until no change or iteration cap
using Rule = std::pair<ExprPtr, ExprPtr>; // (pattern, replacement)
ExprPtr rewriteFixedPoint(const ExprPtr& root, const std::vector<Rule>& rules, int maxIter=50) {
    ExprPtr cur = root;
    for (int it=0; it<maxIter; ++it) {
        bool changed = false;
        for (const auto& r : rules) {
            ExprPtr next = applyRuleOnceAtNode(cur, r.first, r.second);
            if (toString(next) != toString(cur)) { // structural change
                cur = next;
                changed = true;
                break; // restart rule list (first-match semantics)
            }
        }
        if (!changed) break;
    }
    return cur;
}

// Utility: constant folding for simple constant binary combos
ExprPtr tryConstantFoldBinary(const ExprPtr& node) {
    if (!node) return node;
    if (!(node->kind==NodeKind::ADD || node->kind==NodeKind::SUB || node->kind==NodeKind::MUL || node->kind==NodeKind::DIV || node->kind==NodeKind::POW)) return node;
    auto L = node->left;
    auto R = node->right;
    // both CONST and both numeric (resolve labels if available)
    auto resolveConst = [](const ExprPtr& c)->std::pair<bool,cplx>{
        if (!c) return {false, cplx(0,0)};
        if (c->kind == NodeKind::CONST) {
            cplx v = c->value;
            if (!c->label.empty()) {
                cplx tmp;
                if (labelToValue(c->label, tmp)) v = tmp;
            }
            return {true, v};
        }
        return {false, cplx(0,0)};
    };
    auto a = resolveConst(L); auto b = resolveConst(R);
    if (a.first && b.first) {
        cplx res;
        try {
            switch (node->kind) {
                case NodeKind::ADD: res = a.second + b.second; break;
                case NodeKind::SUB: res = a.second - b.second; break;
                case NodeKind::MUL: res = a.second * b.second; break;
                case NodeKind::DIV: res = a.second / b.second; break;
                case NodeKind::POW: res = std::pow(a.second, b.second); break;
                default: return node;
            }
        } catch (...) {
            return node;
        }
        // produce new CONST node (numeric)
        return Expr::Const(res, ""); // numeric const has empty label
    }
    return node;
}

// Helper: check numeric zero/one (resolve labels)
bool isNumericZero(const ExprPtr& n) {
    if (!n) return false;
    if (n->kind != NodeKind::CONST) return false;
    cplx v = n->value;
    if (!n->label.empty()) { if (!labelToValue(n->label, v)) return false; }
    return std::abs(v.real()) + std::abs(v.imag()) < 1e-12;
}
bool isNumericOne(const ExprPtr& n) {
    if (!n) return false;
    if (n->kind != NodeKind::CONST) return false;
    cplx v = n->value;
    if (!n->label.empty()) { if (!labelToValue(n->label, v)) return false; }
    return std::abs(v.real()-1.0) + std::abs(v.imag()) < 1e-12;
}

// Build the default rule set (patterns and replacements)
std::vector<Rule> defaultRules() {
    std::vector<Rule> rules;

    // 1) exp(i * π) -> -1
    // pattern: exp( ?a ) where ?a = ( * i π ) — but easier: impose exact pattern exp( i * π )
    ExprPtr pat_euler = U("exp", B(NodeKind::MUL, I(), PI()));
    ExprPtr repl_euler = C(-1,0,"-1");
    rules.emplace_back(pat_euler, repl_euler);

    // 2) log(exp(?x)) -> ?x
    ExprPtr pat_logexp = U("log", U("exp", V("?x")));
    ExprPtr repl_logexp = V("?x");
    rules.emplace_back(pat_logexp, repl_logexp);

    // 3) exp(log(?x)) -> ?x
    ExprPtr pat_exp_log = U("exp", U("log", V("?x")));
    ExprPtr repl_exp_log = V("?x");
    rules.emplace_back(pat_exp_log, repl_exp_log);

    // 4) sin(0) -> 0
    ExprPtr pat_sin0 = U("sin", C(0,0,"0"));
    ExprPtr repl_sin0 = C(0,0,"0");
    rules.emplace_back(pat_sin0, repl_sin0);

    // 5) cos(0) -> 1
    ExprPtr pat_cos0 = U("cos", C(0,0,"0"));
    ExprPtr repl_cos0 = C(1,0,"1");
    rules.emplace_back(pat_cos0, repl_cos0);

    // 6) x * 1 -> x
    ExprPtr pat_mul1a = B(NodeKind::MUL, V("?x"), C(1,0,"1"));
    ExprPtr pat_mul1b = B(NodeKind::MUL, C(1,0,"1"), V("?x"));
    rules.emplace_back(pat_mul1a, V("?x"));
    rules.emplace_back(pat_mul1b, V("?x"));

    // 7) x + 0 -> x  and 0 + x -> x
    ExprPtr pat_add0a = B(NodeKind::ADD, V("?x"), C(0,0,"0"));
    ExprPtr pat_add0b = B(NodeKind::ADD, C(0,0,"0"), V("?x"));
    rules.emplace_back(pat_add0a, V("?x"));
    rules.emplace_back(pat_add0b, V("?x"));

    // 8) constant folding for binary ops: we will implement these as pattern -> special replacement marker
    // We'll capture generic binary nodes and apply constant folding in a post-processing pass.
    // To keep simple, we don't add a pattern here.

    // 9) pow(e, log(?x)) -> ?x   (e ^ log(x) -> x)
    ExprPtr pat_pow_e_log = B(NodeKind::POW, E(), U("log", V("?x")));
    ExprPtr repl_pow_e_log = V("?x");
    rules.emplace_back(pat_pow_e_log, repl_pow_e_log);

    // Other rules can be appended similarly...
    return rules;
}

// Apply constant folding across the tree
ExprPtr constantFoldingWalk(const ExprPtr& node) {
    if (!node) return node;
    if (node->kind == NodeKind::CONST || node->kind == NodeKind::VAR) return node;
    if (node->kind == NodeKind::UNARY) {
        ExprPtr L = constantFoldingWalk(node->left);
        // if L is CONST -> evaluate function if known
        if (L->kind==NodeKind::CONST) {
            cplx val = L->value;
            if (!L->label.empty()) labelToValue(L->label, val);
            auto it = morphisms().find(node->label);
            if (it != morphisms().end()) {
                cplx res = it->second(val);
                return Expr::Const(res, "");
            }
        }
        return Expr::Unary(node->label, L);
    } else {
        ExprPtr L = constantFoldingWalk(node->left);
        ExprPtr R = constantFoldingWalk(node->right);
        ExprPtr rebuilt = Expr::Binary(node->kind, L, R);
        ExprPtr folded = tryConstantFoldBinary(rebuilt);
        return folded;
    }
}

// Helper to run rewrite rules then constant-fold and repeat until fixed point
ExprPtr simplify(const ExprPtr& root, const std::vector<Rule>& rules, int maxIter=50) {
    ExprPtr cur = root;
    for (int i=0;i<maxIter;++i) {
        ExprPtr next = rewriteFixedPoint(cur, rules, 1);
        // after rewriting apply constant folding across tree
        next = constantFoldingWalk(next);
        if (toString(next) == toString(cur)) break;
        cur = next;
    }
    return cur;
}

// ---------------- Example usage ----------------
int main() {
    auto rules = defaultRules();

    // Example A: Euler: exp(i * π) -> -1
    ExprPtr euler = U("exp", B(NodeKind::MUL, I(), PI()));
    std::cout << "Original Euler:   " << toString(euler) << "\n";
    ExprPtr simple_euler = simplify(euler, rules);
    std::cout << "Simplified Euler: " << toString(simple_euler) << "\n";
    try { std::cout << "Numeric: " << toString(Expr::Const(evaluateNumeric(simple_euler), "")) << "\n"; } catch (...) {}

    std::cout << "\n";

    // Example B: log(exp(x)) -> x
    ExprPtr x = V("x");
    ExprPtr logexp = U("log", U("exp", x));
    std::cout << "Original log(exp(x)): " << toString(logexp) << "\n";
    std::cout << "After rewrite:         " << toString(simplify(logexp, rules)) << "\n\n";

    // Example C: exp(log(e ^ log(y)))  -> should reduce to y (through pow and exp/log rules)
    ExprPtr y = V("y");
    ExprPtr logy = U("log", y);
    ExprPtr pow_e_logy = B(NodeKind::POW, E(), logy);            // e ^ log(y)
    ExprPtr wrapped = U("exp", U("log", pow_e_logy));             // exp(log(e^log(y)))
    std::cout << "Original complicated: " << toString(wrapped) << "\n";
    ExprPtr simplifiedWrapped = simplify(wrapped, rules);
    std::cout << "Simplified complicated: " << toString(simplifiedWrapped) << "\n\n";

    // Example D: algebraic cleanups: (π + e) * 1 -> (π + e)
    ExprPtr sum_pi_e = B(NodeKind::ADD, PI(), E());
    ExprPtr times_one = B(NodeKind::MUL, sum_pi_e, ONE());
    std::cout << "Original times_one: " << toString(times_one) << "\n";
    ExprPtr s_times_one = simplify(times_one, rules);
    std::cout << "Simplified times_one: " << toString(s_times_one) << "\n";
    try {
        cplx v = evaluateNumeric(s_times_one);
        std::cout << "Numeric value approx: (" << v.real() << (v.imag()>=0?"+":"") << v.imag() << "i)\n";
    } catch (const std::exception& ex) {
        std::cout << "Numeric eval failed: " << ex.what() << "\n";
    }

    std::cout << "\n";

    // Example E: sin(0) -> 0
    ExprPtr sin0 = U("sin", ZERO());
    std::cout << "sin(0) -> " << toString(simplify(sin0, rules)) << "\n";

    // You can append rules:
    // e.g. add rule: cos(π) -> -1  -> pattern U("cos", PI()) -> Const(-1)
    rules.emplace_back(U("cos", PI()), C(-1,0,"-1"));
    ExprPtr cospi = U("cos", PI());
    std::cout << "cos(pi) original: " << toString(cospi) << " -> " << toString(simplify(cospi, rules)) << "\n";

    return 0;
}
