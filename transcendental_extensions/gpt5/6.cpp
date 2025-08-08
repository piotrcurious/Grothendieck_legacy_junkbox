/*
 symbolic_engine.cpp
 C++17 single-file symbolic complex expressions with:
  - n-ary associative ADD / MUL (flattened + canonical order)
  - commutative pattern matching with sequence wildcards (?x, ?+x, ?*x)
  - pattern DSL helpers (no textual parser)
  - morphism registry (sin, cos, exp, log, ...)
  - rewrite engine with fixed point simplification
  - interning by structural key to accelerate comparisons

 Build:
   g++ -std=c++17 symbolic_engine.cpp -O2 -o symbolic_engine -lm
*/

#include <bits/stdc++.h>
using namespace std;
using cplx = complex<double>;
constexpr double EPS = 1e-12;

// ---------------------- Expression definitions ----------------------

enum class Kind { CONST, VAR, UNARY, NARY, POW }; // NARY covers ADD and MUL with op flag

enum class NaryOp { ADD, MUL };

struct Expr;
using ExprPtr = shared_ptr<Expr>;

struct Expr {
    Kind kind;
    // for CONST/VAR/UNARY: label (e.g., "π", "x", "sin")
    string label;
    // numeric value for CONST nodes (if numeric known)
    cplx value{0,0};
    // for UNARY or POW: child in left (and right for POW)
    ExprPtr left, right;
    // for NARY: operator and list of operands
    NaryOp nop;
    vector<ExprPtr> ops;

    // structural key cached (for interning/comparison)
    mutable string key_cache;

    Expr() = default;

    // constructors
    static ExprPtr makeConst(cplx v = {0,0}, const string &lbl = "") {
        auto p = make_shared<Expr>();
        p->kind = Kind::CONST;
        p->value = v;
        p->label = lbl;
        normalizeKey(*p);
        return p;
    }
    static ExprPtr makeVar(const string &name) {
        auto p = make_shared<Expr>();
        p->kind = Kind::VAR;
        p->label = name;
        normalizeKey(*p);
        return p;
    }
    static ExprPtr makeUnary(const string &fname, ExprPtr arg) {
        auto p = make_shared<Expr>();
        p->kind = Kind::UNARY;
        p->label = fname;
        p->left = move(arg);
        normalizeKey(*p);
        return p;
    }
    static ExprPtr makePow(ExprPtr a, ExprPtr b) {
        auto p = make_shared<Expr>();
        p->kind = Kind::POW;
        p->left = move(a);
        p->right = move(b);
        normalizeKey(*p);
        return p;
    }
    static ExprPtr makeNary(NaryOp op, vector<ExprPtr> operands) {
        auto p = make_shared<Expr>();
        p->kind = Kind::NARY;
        p->nop = op;
        p->ops = move(operands);
        normalizeKey(*p);
        return p;
    }

    // human readable
    string toString() const {
        switch (kind) {
            case Kind::CONST:
                if (!label.empty()) return label;
                {
                    ostringstream os; os<<fixed<<setprecision(12)<<"("<<value.real()<<(value.imag()>=0?"+":"")<<value.imag()<<"i)"; return os.str();
                }
            case Kind::VAR: return label;
            case Kind::UNARY: return label + "(" + left->toString() + ")";
            case Kind::POW: return "(" + left->toString() + " ^ " + right->toString() + ")";
            case Kind::NARY: {
                string op = (nop==NaryOp::ADD? " + " : " * ");
                string s = "(";
                for (size_t i=0;i<ops.size();++i) {
                    if (i) s += op;
                    s += ops[i]->toString();
                }
                s += ")";
                return s;
            }
        }
        return "?";
    }

    // structural key used for canonicalization & interning
    static string normalizeKey(Expr &e) {
        // produce a canonical string key and store in key_cache
        if (e.kind == Kind::CONST) {
            if (!e.label.empty()) { e.key_cache = "C:" + e.label; return e.key_cache; }
            ostringstream os; os<<setprecision(12)<<"N:"<<e.value.real()<<","<<e.value.imag();
            e.key_cache = os.str(); return e.key_cache;
        } else if (e.kind == Kind::VAR) {
            e.key_cache = "V:" + e.label; return e.key_cache;
        } else if (e.kind == Kind::UNARY) {
            e.key_cache = "U:" + e.label + "(" + (e.left? e.left->key_cache : "") + ")";
            return e.key_cache;
        } else if (e.kind == Kind::POW) {
            e.key_cache = "P(" + (e.left? e.left->key_cache : "") + "," + (e.right? e.right->key_cache : "") + ")";
            return e.key_cache;
        } else { // NARY
            // flatten nested same-op nodes and canonicalize operands (sort by key)
            vector<ExprPtr> flat;
            function<void(const ExprPtr&)> gather = [&](const ExprPtr& n){
                if (!n) return;
                if (n->kind==Kind::NARY && n->nop==e.nop) {
                    for (auto &c: n->ops) gather(c);
                } else flat.push_back(n);
            };
            // if ops already set, gather from them
            for (auto &c: e.ops) gather(c);
            // produce canonical order (sort by key - stable)
            sort(flat.begin(), flat.end(), [](const ExprPtr &a, const ExprPtr &b){
                return a->key_cache < b->key_cache;
            });
            // replace operands with canonical flat list
            e.ops = move(flat);
            // build key
            string k = (e.nop==NaryOp::ADD? "NA[" : "NM[");
            for (size_t i=0;i<e.ops.size();++i) {
                if (i) k.push_back(',');
                k += e.ops[i]->key_cache;
            }
            k.push_back(']');
            e.key_cache = k;
            return e.key_cache;
        }
    }
};

// ---------------------- Interning ----------------------
// Simple intern pool to return shared canonical nodes with identical structural key
struct Intern {
    unordered_map<string, ExprPtr> pool;
    ExprPtr intern(const ExprPtr &node) {
        if (!node) return nullptr;
        string key = node->key_cache;
        auto it = pool.find(key);
        if (it != pool.end()) return it->second;
        pool[key] = node;
        return node;
    }
} INTERN;

// helper factories that intern nodes
inline ExprPtr C(double r=0,double im=0,const string &lbl="") { return INTERN.intern(Expr::makeConst({r,im}, lbl)); }
inline ExprPtr V(const string &n) { return INTERN.intern(Expr::makeVar(n)); }
inline ExprPtr U(const string &f, ExprPtr a) { return INTERN.intern(Expr::makeUnary(f, a)); }
inline ExprPtr P(ExprPtr a, ExprPtr b) { return INTERN.intern(Expr::makePow(a,b)); }
inline ExprPtr N(NaryOp op, vector<ExprPtr> operands) { return INTERN.intern(Expr::makeNary(op, move(operands))); }
inline ExprPtr ADD(initializer_list<ExprPtr> l) { return N(NaryOp::ADD, vector<ExprPtr>(l)); }
inline ExprPtr MUL(initializer_list<ExprPtr> l) { return N(NaryOp::MUL, vector<ExprPtr>(l)); }
inline ExprPtr PI() { return C(0,0,"π"); }
inline ExprPtr EE() { return C(0,0,"e"); }
inline ExprPtr II() { return C(0,1,"i"); }
inline ExprPtr ZERO() { return C(0,0,"0"); }
inline ExprPtr ONE() { return C(1,0,"1"); }

// ---------------------- Morphism registry ----------------------
using MorphMap = unordered_map<string, function<cplx(const cplx&)>>;
MorphMap& morphisms() {
    static MorphMap mm = {
        {"sin", [](const cplx &z){ return sin(z); }},
        {"cos", [](const cplx &z){ return cos(z); }},
        {"exp", [](const cplx &z){ return exp(z); }},
        {"log", [](const cplx &z){ return log(z); }},
        {"sqrt",[](const cplx &z){ return sqrt(z); }},
        {"neg", [](const cplx &z){ return -z; }}
    };
    return mm;
}

// label -> numeric
bool labelToValue(const string &lbl, cplx &out) {
    if (lbl=="π" || lbl=="pi") { out = cplx(M_PI,0); return true; }
    if (lbl=="e") { out = cplx(exp(1.0),0); return true; }
    if (lbl=="i") { out = cplx(0,1); return true; }
    if (lbl=="0") { out = cplx(0,0); return true; }
    if (lbl=="1") { out = cplx(1,0); return true; }
    return false;
}

// ---------------------- Numeric evaluation ----------------------
cplx evaluate(const ExprPtr &e) {
    if (!e) return {0,0};
    switch (e->kind) {
        case Kind::CONST: {
            if (!e->label.empty()) {
                cplx v; if (labelToValue(e->label, v)) return v;
            }
            return e->value;
        }
        case Kind::VAR:
            throw runtime_error("Cannot evaluate variable: " + e->label);
        case Kind::UNARY: {
            cplx a = evaluate(e->left);
            auto it = morphisms().find(e->label);
            if (it==morphisms().end()) throw runtime_error("Unknown morphism: " + e->label);
            return it->second(a);
        }
        case Kind::POW: {
            cplx a=evaluate(e->left), b=evaluate(e->right);
            return pow(a,b);
        }
        case Kind::NARY: {
            if (e->nop==NaryOp::ADD) {
                cplx s{0,0};
                for (auto &c: e->ops) s += evaluate(c);
                return s;
            } else {
                cplx p{1,0};
                for (auto &c: e->ops) p *= evaluate(c);
                return p;
            }
        }
    }
    return {0,0};
}

// ---------------------- Pattern language ----------------------
//
// Pattern variables:
//   - single wildcard: Var("?x") or Const(label="?x") -> binds a single subtree to "x"
//   - sequence wildcard: Var("?+x") -> binds one-or-more operands in an n-ary node to "x" (vector binding)
//   - sequence wildcard zero-or-more: Var("?*x") -> binds zero-or-more operands
//
// Bindings store two maps:
//   single_bind: name -> ExprPtr
//   seq_bind: name -> vector<ExprPtr>
//
struct Bindings {
    unordered_map<string, ExprPtr> single;
    unordered_map<string, vector<ExprPtr>> seq;
    bool hasSingle(const string &k) const { return single.find(k)!=single.end(); }
    bool hasSeq(const string &k) const { return seq.find(k)!=seq.end(); }
};

// helper to detect wildcard kinds
enum class PatMode { NONE, SINGLE, SEQ_ONEPLUS, SEQ_ZEROORMORE };
static inline PatMode patternMode(const ExprPtr &p) {
    if (!p) return PatMode::NONE;
    if (p->kind==Kind::VAR && !p->label.empty() && p->label[0]=='?') {
        if (p->label.size()>=3 && p->label[1]=='+' ) return PatMode::SEQ_ONEPLUS; // ?+x
        if (p->label.size()>=3 && p->label[1]=='*' ) return PatMode::SEQ_ZEROORMORE; // ?*x
        return PatMode::SINGLE; // ?x
    }
    if (p->kind==Kind::CONST && !p->label.empty() && p->label[0]=='?') {
        // allow ?x as const node too
        if (p->label.size()>=3 && p->label[1]=='+' ) return PatMode::SEQ_ONEPLUS;
        if (p->label.size()>=3 && p->label[1]=='*' ) return PatMode::SEQ_ZEROORMORE;
        return PatMode::SINGLE;
    }
    return PatMode::NONE;
}
static inline string patternName(const ExprPtr &p) {
    if (!p) return "";
    if (p->label.empty()) return "";
    if (p->label[0] != '?') return "";
    if (p->label.size()>=3 && (p->label[1]=='+' || p->label[1]=='*')) return p->label.substr(2);
    return p->label.substr(1);
}

// structural equality quick check via key
static inline bool structurallyEqual(const ExprPtr &a, const ExprPtr &b) {
    if (!a || !b) return a==b;
    return a->key_cache == b->key_cache;
}

// Match pattern to subject. Returns true if match succeeded and fills bindings.
// This is a recursive matching that handles n-ary sequence wildcards with backtracking.
bool matchPattern(const ExprPtr &pat, const ExprPtr &subj, Bindings &B);

// helper: match list of pattern operands (may include seq wildcards) to list of subject operands
bool matchNaryList(const vector<ExprPtr> &pats, const vector<ExprPtr> &subs, Bindings &B) {
    // Backtracking index-based recursion
    function<bool(size_t,size_t)> dfs = [&](size_t pi, size_t si)->bool {
        // if both consumed -> success
        if (pi == pats.size() && si == subs.size()) return true;
        // if pattern consumed but subjects remain -> fail (unless trailing seq wildcard handles them)
        if (pi == pats.size()) return false;
        // if subjects consumed but patterns remain -> allow only seq-zero-or-more that can match empty
        if (si == subs.size()) {
            // Remaining patterns must all be ?*x (zero-or-more)
            for (size_t k = pi; k < pats.size(); ++k) {
                PatMode pm = patternMode(pats[k]);
                if (pm != PatMode::SEQ_ZEROORMORE) return false;
                // bind empty vector if not already bound
                string name = patternName(pats[k]);
                if (!B.hasSeq(name)) B.seq[name] = {};
            }
            return true;
        }

        // inspect current pattern item
        PatMode pm = patternMode(pats[pi]);
        if (pm == PatMode::SINGLE) {
            string name = patternName(pats[pi]);
            // bind single subtree
            if (B.hasSingle(name)) {
                // must be equal to previous binding
                if (!structurallyEqual(B.single[name], subs[si])) return false;
            } else {
                B.single[name] = subs[si];
            }
            return dfs(pi+1, si+1);
        } else if (pm == PatMode::SEQ_ONEPLUS || pm == PatMode::SEQ_ZEROORMORE) {
            string name = patternName(pats[pi]);
            // sequence wildcard can capture k >= (1 or 0) operands
            size_t mink = (pm==PatMode::SEQ_ONEPLUS)? 1 : 0;
            // If already bound, must match equal sequence
            if (B.hasSeq(name)) {
                const auto &boundseq = B.seq[name];
                size_t k = boundseq.size();
                if (si + k > subs.size()) return false;
                for (size_t t=0;t<k;++t) if (!structurallyEqual(boundseq[t], subs[si+t])) return false;
                return dfs(pi+1, si+k);
            } else {
                // try all possible k from mink..(subs.size()-si - remaining_min_required)
                // compute minimal required remaining slots for later pattern elements (conservative: 0)
                for (size_t k = mink; si + k <= subs.size(); ++k) {
                    // bind name -> subs[si .. si+k-1]
                    vector<ExprPtr> seq(subs.begin()+si, subs.begin()+si+k);
                    B.seq[name] = seq;
                    if (dfs(pi+1, si+k)) return true;
                    B.seq.erase(name);
                }
                return false;
            }
        } else {
            // normal pattern element: match single subtree recursively
            Bindings snapshot = B; // copy for backtracking
            if (!matchPattern(pats[pi], subs[si], B)) { B = snapshot; return false; }
            return dfs(pi+1, si+1);
        }
    };
    return dfs(0,0);
}

// main pattern match
bool matchPattern(const ExprPtr &pat, const ExprPtr &subj, Bindings &B) {
    if (!pat || !subj) return pat==subj;
    PatMode pm = patternMode(pat);
    if (pm == PatMode::SINGLE) {
        string name = patternName(pat);
        if (B.hasSingle(name)) return structurallyEqual(B.single[name], subj);
        B.single[name] = subj;
        return true;
    }
    if (pm == PatMode::SEQ_ONEPLUS || pm == PatMode::SEQ_ZEROORMORE) {
        // sequence wildcard at non-nary location only allowed if it binds the whole node as a single target (we treat as single)
        string name = patternName(pat);
        // single-binding of a sequence at non-nary context: represent as seq of one element (the whole subj)
        if (B.hasSeq(name)) {
            auto &seq = B.seq[name];
            if (seq.size()!=1) return false;
            return structurallyEqual(seq[0], subj);
        } else {
            B.seq[name] = {subj};
            return true;
        }
    }

    // non-wildcard cases: must have same kind
    if (pat->kind != subj->kind) return false;
    switch (pat->kind) {
        case Kind::CONST:
            if (!pat->label.empty()) {
                // label match
                return pat->label == subj->label;
            } else {
                // numeric compare
                cplx av = pat->value, bv = subj->value;
                if (!pat->label.empty()) labelToValue(pat->label, av);
                if (!subj->label.empty()) labelToValue(subj->label, bv);
                return abs(av.real()-bv.real())<EPS && abs(av.imag()-bv.imag())<EPS;
            }
        case Kind::VAR:
            return pat->label == subj->label;
        case Kind::UNARY:
            if (pat->label != subj->label) return false;
            return matchPattern(pat->left, subj->left, B);
        case Kind::POW:
            return matchPattern(pat->left, subj->left, B) && matchPattern(pat->right, subj->right, B);
        case Kind::NARY:
            if (pat->nop != subj->nop) return false;
            // special-case: if pattern NARY contains sequence wildcards ?* / ?+ we use matchNaryList
            return matchNaryList(pat->ops, subj->ops, B);
    }
    return false;
}

// substitute bindings into replacement pattern
// support splicing seq bindings into NARY replacements (if replacement contains Var("?+x") etc.)
ExprPtr substitute(const ExprPtr &repl, const Bindings &B);

// helper: when substituting into an NARY node, if one of replacement operands is a seq wildcard, splice its bound sequence
ExprPtr substituteIntoNary(NaryOp op, const vector<ExprPtr> &repl_ops, const Bindings &B) {
    vector<ExprPtr> out;
    for (auto &r : repl_ops) {
        PatMode pm = patternMode(r);
        if (pm==PatMode::SEQ_ONEPLUS || pm==PatMode::SEQ_ZEROORMORE) {
            string name = patternName(r);
            auto it = B.seq.find(name);
            if (it != B.seq.end()) {
                // insert each bound item
                for (auto &e : it->second) out.push_back(e);
                continue;
            } else {
                // unbound seq wildcard -> treat as empty for ?* or fail for ?+
                if (pm==PatMode::SEQ_ZEROORMORE) continue;
                else {
                    // ?+ unbound - can't substitute -> keep literal wildcard? we'll keep as variable itself
                    out.push_back(r);
                    continue;
                }
            }
        } else {
            // normal substitution
            out.push_back(substitute(r, B));
        }
    }
    // If after substitution we have single operand and op is ADD/MUL, keep as that operand (no n-ary wrapper)
    if (out.empty()) {
        // return neutral element
        return (op==NaryOp::ADD? ZERO() : ONE());
    }
    if (out.size()==1) return out[0];
    return N(op, out);
}

ExprPtr substitute(const ExprPtr &repl, const Bindings &B) {
    if (!repl) return nullptr;
    PatMode pm = patternMode(repl);
    if (pm == PatMode::SINGLE) {
        string name = patternName(repl);
        auto it = B.single.find(name);
        if (it != B.single.end()) return it->second;
        // not bound -> return original var as-is
        return repl;
    }
    if (pm == PatMode::SEQ_ONEPLUS || pm == PatMode::SEQ_ZEROORMORE) {
        string name = patternName(repl);
        auto it = B.seq.find(name);
        if (it != B.seq.end()) {
            // if seq bound to single element and we are substitute in non-nary context, return that element
            if (it->second.size()==1) return it->second[0];
            // otherwise we are substituting a sequence into non-nary context: wrap as MUL of elements (or ADD?) - ambiguous.
            // We choose to wrap as MUL(...) if each item is a factor, but it's ambiguous: leave as N-ary MUL by default
            return N(NaryOp::MUL, it->second);
        }
        // unbound -> represent empty sequence as neutral element
        return ONE();
    }

    // otherwise normal recursive substitute
    if (repl->kind == Kind::CONST) return C(repl->value.real(), repl->value.imag(), repl->label);
    if (repl->kind == Kind::VAR) return V(repl->label);
    if (repl->kind == Kind::UNARY) {
        return U(repl->label, substitute(repl->left, B));
    }
    if (repl->kind == Kind::POW) {
        return P(substitute(repl->left, B), substitute(repl->right, B));
    }
    if (repl->kind == Kind::NARY) {
        return substituteIntoNary(repl->nop, repl->ops, B);
    }
    return nullptr;
}

// ---------------------- Rewrite engine ----------------------
// Rule: pair<pattern, replacement>
using Rule = pair<ExprPtr, ExprPtr>;

// apply single rule once: search for first match (preorder leftmost-deepest) and replace; return new tree (or same tree)
ExprPtr applyRuleOnce(const ExprPtr &node, const Rule &rule) {
    if (!node) return node;
    Bindings B;
    if (matchPattern(rule.first, node, B)) {
        return substitute(rule.second, B);
    }
    // otherwise descend
    if (node->kind == Kind::UNARY) {
        ExprPtr nl = applyRuleOnce(node->left, rule);
        if (!structurallyEqual(nl, node->left)) return INTERN.intern(Expr::makeUnary(node->label, nl));
        return node;
    } else if (node->kind == Kind::POW) {
        ExprPtr nl = applyRuleOnce(node->left, rule);
        if (!structurallyEqual(nl, node->left)) return INTERN.intern(Expr::makePow(nl, node->right));
        ExprPtr nr = applyRuleOnce(node->right, rule);
        if (!structurallyEqual(nr, node->right)) return INTERN.intern(Expr::makePow(node->left, nr));
        return node;
    } else if (node->kind == Kind::NARY) {
        // try each operand
        for (size_t i=0;i<node->ops.size();++i) {
            ExprPtr child = node->ops[i];
            ExprPtr newChild = applyRuleOnce(child, rule);
            if (!structurallyEqual(newChild, child)) {
                auto newOps = node->ops;
                newOps[i] = newChild;
                return INTERN.intern(Expr::makeNary(node->nop, move(newOps)));
            }
        }
        return node;
    }
    return node;
}

// apply rules until fixed point
ExprPtr rewriteFixedPoint(const ExprPtr &root, const vector<Rule> &rules, int maxIter=50) {
    ExprPtr cur = root;
    for (int it=0; it<maxIter; ++it) {
        bool changed = false;
        for (auto &r : rules) {
            ExprPtr next = applyRuleOnce(cur, r);
            if (!structurallyEqual(next, cur)) { cur = next; changed = true; break; }
        }
        if (!changed) break;
    }
    return cur;
}

// ---------------------- Constant folding & algebraic simplifiers ----------------------
ExprPtr constantFold(const ExprPtr &node) {
    if (!node) return node;
    if (node->kind == Kind::CONST || node->kind == Kind::VAR) return node;
    if (node->kind == Kind::UNARY) {
        ExprPtr a = constantFold(node->left);
        if (a->kind == Kind::CONST) {
            cplx v = a->value;
            if (!a->label.empty()) labelToValue(a->label, v);
            auto it = morphisms().find(node->label);
            if (it!=morphisms().end()) {
                cplx r = it->second(v);
                return C(r.real(), r.imag(), "");
            }
        }
        return U(node->label, a);
    } else if (node->kind == Kind::POW) {
        ExprPtr A = constantFold(node->left);
        ExprPtr B = constantFold(node->right);
        if (A->kind==Kind::CONST && B->kind==Kind::CONST) {
            cplx a = A->value; if (!A->label.empty()) labelToValue(A->label, a);
            cplx b = B->value; if (!B->label.empty()) labelToValue(B->label, b);
            try {
                cplx r = pow(a,b);
                return C(r.real(), r.imag(), "");
            } catch (...) {}
        }
        return P(A,B);
    } else if (node->kind == Kind::NARY) {
        vector<ExprPtr> newops;
        for (auto &c: node->ops) newops.push_back(constantFold(c));
        // flatten same-op children & combine constants
        vector<ExprPtr> flat;
        cplx accumConst = (node->nop==NaryOp::ADD? cplx(0,0): cplx(1,0));
        bool hasAccumConst = false;
        for (auto &o: newops) {
            if (o->kind==Kind::NARY && o->nop==node->nop) {
                for (auto &g: o->ops) flat.push_back(g);
            } else flat.push_back(o);
        }
        // gather constants
        vector<ExprPtr> nonConst;
        for (auto &o: flat) {
            if (o->kind==Kind::CONST) {
                cplx v = o->value;
                if (!o->label.empty()) labelToValue(o->label, v);
                if (!hasAccumConst) { accumConst = v; hasAccumConst=true; }
                else accumConst = (node->nop==NaryOp::ADD? accumConst+v : accumConst * v);
            } else nonConst.push_back(o);
        }
        // If there is a constant neutral element and nonConst empty -> return const
        if (nonConst.empty()) {
            if (hasAccumConst) return C(accumConst.real(), accumConst.imag(), "");
            // else return neutral
            return (node->nop==NaryOp::ADD? ZERO() : ONE());
        }
        // if constant is neutral, and we have nonConst, include it only if necessary
        if (hasAccumConst) {
            bool isNeutral = (node->nop==NaryOp::ADD? abs(accumConst.real())+abs(accumConst.imag()) < EPS :
                                                  abs(accumConst.real()-1.0)+abs(accumConst.imag()) < EPS);
            if (!isNeutral) nonConst.push_back(C(accumConst.real(), accumConst.imag(), ""));
        }
        // if only one operand, return it
        if (nonConst.size()==1) return nonConst[0];
        return N(node->nop, nonConst);
    }
    return node;
}

// top-level simplify: rewrite rules then constant fold repeatedly
ExprPtr simplify(const ExprPtr &root, const vector<Rule> &rules, int maxIter=50) {
    ExprPtr cur = root;
    for (int i=0;i<maxIter;++i) {
        ExprPtr next = rewriteFixedPoint(cur, rules, 1);
        next = constantFold(next);
        if (next->key_cache == cur->key_cache) break;
        cur = next;
    }
    return cur;
}

// ---------------------- Pattern DSL helpers ----------------------
// Create wildcard patterns: ?x, ?+x, ?*x
inline ExprPtr W(const string &name) { return V("?" + name); }       // ?x
inline ExprPtr Wplus(const string &name) { return V("?+" + name); }  // ?+x
inline ExprPtr Wstar(const string &name) { return V("?*" + name); }  // ?*x

// ---------------------- Default rules ----------------------
vector<Rule> defaultRules() {
    vector<Rule> rules;
    // Euler: exp(i * π) -> -1
    rules.emplace_back(U("exp", MUL({II(), PI()})), C(-1,0,"-1"));
    // log(exp(?x)) -> ?x
    rules.emplace_back(U("log", U("exp", W("x"))), W("x"));
    // exp(log(?x)) -> ?x
    rules.emplace_back(U("exp", U("log", W("x"))), W("x"));
    // pow(e, log(?x)) -> ?x
    rules.emplace_back(P(EE(), U("log", W("x"))), W("x"));
    // sin(0) -> 0
    rules.emplace_back(U("sin", ZERO()), ZERO());
    // cos(0) -> 1
    rules.emplace_back(U("cos", ZERO()), ONE());
    // x * 1 -> x  and 1 * x -> x  (but with n-ary canonicalization these will be handled by constant folding; still add safe rules)
    rules.emplace_back(MUL({W("x"), ONE()}), W("x"));
    rules.emplace_back(MUL({ONE(), W("x")}), W("x"));
    // x + 0 -> x
    rules.emplace_back(ADD({W("x"), ZERO()}), W("x"));
    rules.emplace_back(ADD({ZERO(), W("x")}), W("x"));
    // associative pattern using sequence wildcard: (a + b + ?+rest) -> (?+rest + a + b) example not typical; it's just showing seq wildcards.
    // Example: combine like terms could be implemented with sophisticated patterns and canonicalization.
    return rules;
}

// ---------------------- Examples & tests ----------------------
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Prepare default rules
    vector<Rule> rules = defaultRules();

    // Example 1: Euler identity
    ExprPtr euler = U("exp", MUL({II(), PI()}));
    cout << "Original Euler:   " << euler->toString() << "\n";
    ExprPtr s = simplify(euler, rules);
    cout << "Simplified Euler: " << s->toString() << "\n";
    try { cplx v = evaluate(s); cout << "Numeric: ("<<v.real()<<(v.imag()>=0?"+":"")<<v.imag()<<"i)\n\n"; } catch(...) { cout<<"\n"; }

    // Example 2: log(exp(x)) -> x
    ExprPtr x = V("x");
    ExprPtr logexp = U("log", U("exp", x));
    cout << "Original log(exp(x)): " << logexp->toString() << "\n";
    cout << "Simplified:            " << simplify(logexp, rules)->toString() << "\n\n";

    // Example 3: associative/commutative matching with sequence wildcard
    // Pattern: ( ?a + ?+rest ) -> rewrite to ( ?+rest + ?a ) is trivial but demonstrates sequence matching.
    // Let's make expression: (1 + 2 + 3 + x)
    ExprPtr expr = ADD({ C(1,0,"1"), C(2,0,"2"), C(3,0,"3"), V("x") });
    cout << "Original sum: " << expr->toString() << "\n";
    // Pattern that captures first element as ?a and rest as ?+rest
    ExprPtr pat = ADD({ W("a"), Wplus("rest") });
    ExprPtr repl = ADD({ Wplus("rest"), W("a") }); // rotates first element to end
    // apply single rule
    Rule rotate = {pat, repl};
    ExprPtr rotated = applyRuleOnce(expr, rotate);
    cout << "After one rotation (applyRuleOnce): " << rotated->toString() << "\n";
    // repeatedly rotate until fixed point (but canonical sorting will often keep canonical order)
    cout << "\n";

    // Example 4: combine constant folding and n-ary product: (2 * 3 * x * 1) -> (6 * x)
    ExprPtr prod = MUL({ C(2,0,"2"), C(3,0,"3"), V("x"), ONE() });
    cout << "Original product: " << prod->toString() << "\n";
    ExprPtr sprod = simplify(prod, rules);
    cout << "Simplified product: " << sprod->toString() << "\n";
    try { cplx vp = evaluate(sprod); cout<<"Numeric eval: ("<<vp.real()<<(vp.imag()>=0?"+":"")<<vp.imag()<<"i)\n\n"; } catch(...) { cout<<"\n"; }

    // Example 5: pattern that matches a subset in commutative multiply: (a * b * ?+rest) -> replace a*b with single var ?z
    // E.g. multiply = (x * a * b * c) and pattern MUL({ Wplus("pref"), W("a"), W("b"), Wplus("suf") }) complicated.
    // To demonstrate: rule: ( ?*L , a, b, ?*R ) -> ( z, ?*L, ?*R ) replace a*b by z (we emulate by sequence wildcards)
    ExprPtr a = V("a"); ExprPtr b = V("b");
    ExprPtr mulexpr = MUL({ V("x"), a, b, V("y") });
    cout << "Original mulexpr: " << mulexpr->toString() << "\n";
    // pattern with ?*L, a, b, ?*R
    ExprPtr pat2 = MUL({ Wstar("L"), a, b, Wstar("R") });
    // replace a*b with single var Z and splice back L and R: replacement -> ( ?*L , Z, ?*R )
    ExprPtr repl2 = MUL({ Wstar("L"), V("Z"), Wstar("R") });
    // we need to provide binding for Z from matched a*b: we can add a second rewrite that after matching pattern will create binding Z from seq ["a","b"]
    // For demonstration, we do a manual match then custom substitute:
    Bindings B;
    bool ok = matchPattern(pat2, mulexpr, B);
    cout << "Match pattern ?*L a b ?*R -> " << (ok? "yes":"no") << "\n";
    if (ok) {
        // bind Z as product of first seq matched (assume a and b are somewhere, but here we create Z = MUL({a,b}))
        B.single["Z"] = MUL({ a, b });
        ExprPtr replaced = substitute(repl2, B);
        cout << "After custom replace: " << replaced->toString() << "\n";
    }
    cout << "\n";

    // Example 6: rewrite set in action -- add cos(pi)->-1 rule and simplify
    rules.emplace_back(U("cos", PI()), C(-1,0,"-1"));
    ExprPtr cospi = U("cos", PI());
    cout << "cos(pi) -> " << simplify(cospi, rules)->toString() << "\n";

    // Example 7: show sequence wildcard used inside replacement splicing constants
    ExprPtr sum2 = ADD({ V("a"), V("b"), V("c"), C(0,0,"0") });
    cout << "Original sum2: " << sum2->toString() << "\n";
    // rule: ( ?+prefix , 0 ) -> ( ?+prefix )  (strip trailing zero)
    Rule stripZero = { ADD({ Wplus("prefix"), ZERO() }), Wplus("prefix") };
    ExprPtr stripped = applyRuleOnce(sum2, stripZero);
    cout << "After strip zero: " << stripped->toString() << "\n";

    return 0;
}
