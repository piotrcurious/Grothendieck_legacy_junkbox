#ifndef GALOIS_MATH_H
#define GALOIS_MATH_H

#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

typedef std::vector<int> GFElement;

inline bool is_prime(int n) {
  if (n < 2) return false;
  for (int i = 2; i * i <= n; ++i) if (n % i == 0) return false;
  return true;
}

inline int gf_inv(int n, int p) {
  int a = n % p, b = p, x0 = 0, x1 = 1;
  while (a > 1) {
    if (b == 0) return 1;
    int q = a / b, t = b;
    b = a % b; a = t;
    t = x0; x0 = x1 - q * x0; x1 = t;
  }
  return (x1 < 0) ? x1 + p : x1;
}

inline bool gf_is_zero(const GFElement &a) {
  return a.empty() || (a.size() == 1 && a[0] == 0);
}

inline bool gf_is_one(const GFElement &a) {
  return a.size() == 1 && a[0] == 1;
}

inline GFElement gf_add(const GFElement &a, const GFElement &b, int p) {
  int n = (int)std::max(a.size(), b.size());
  GFElement res(n, 0);
  for (int i = 0; i < n; ++i) {
    int va = (i < (int)a.size()) ? a[i] : 0;
    int vb = (i < (int)b.size()) ? b[i] : 0;
    res[i] = (va + vb) % p;
  }
  while (res.size() > 1 && res.back() == 0) res.pop_back();
  return res;
}

inline GFElement gf_poly_sub(const GFElement &a, const GFElement &b, int p) {
  int n = (int)std::max(a.size(), b.size());
  GFElement res(n, 0);
  for (int i = 0; i < n; ++i) {
    int va = (i < (int)a.size()) ? a[i] : 0;
    int vb = (i < (int)b.size()) ? b[i] : 0;
    res[i] = (va - vb + p) % p;
  }
  while (res.size() > 1 && res.back() == 0) res.pop_back();
  return res;
}

inline GFElement gf_poly_mul(const GFElement &a, const GFElement &b, int p) {
  if (a.empty() || b.empty() || gf_is_zero(a) || gf_is_zero(b)) return {0};
  GFElement res(a.size() + b.size() - 1, 0);
  for (int i = 0; i < (int)a.size(); ++i) {
    for (int j = 0; j < (int)b.size(); ++j) {
      res[i + j] = (int)((res[i + j] + 1LL * a[i] * b[j]) % p);
    }
  }
  while (res.size() > 1 && res.back() == 0) res.pop_back();
  return res;
}

inline GFElement gf_poly_mod(GFElement a, const GFElement &g, int p) {
  if (gf_is_zero(g)) return a;
  int n = (int)g.size() - 1;
  int lead = g.back() % p;
  int inv_lead = gf_inv(lead, p);
  while ((int)a.size() > n) {
    int factor = (int)((1LL * a.back() * inv_lead) % p);
    int deg_diff = (int)a.size() - 1 - n;
    for (int i = 0; i <= n; ++i) {
      int idx = i + deg_diff;
      a[idx] = (int)((a[idx] - 1LL * factor * g[i] % p + p) % p);
    }
    while (!a.empty() && a.back() == 0) a.pop_back();
  }
  if (a.empty()) return {0};
  return a;
}

inline std::pair<GFElement, GFElement> gf_poly_div(GFElement a, const GFElement &g, int p) {
  if (gf_is_zero(g)) return {{0}, a};
  int n = (int)g.size() - 1;
  int lead = g.back() % p;
  int inv_lead = gf_inv(lead, p);
  GFElement q;
  while ((int)a.size() > n) {
    int factor = (int)((1LL * a.back() * inv_lead) % p);
    int deg_diff = (int)a.size() - 1 - n;
    if (q.size() <= (size_t)deg_diff) q.resize(deg_diff + 1, 0);
    q[deg_diff] = factor;
    for (int i = 0; i <= n; ++i) {
      int idx = i + deg_diff;
      a[idx] = (int)((a[idx] - 1LL * factor * g[i] % p + p) % p);
    }
    while (!a.empty() && a.back() == 0) a.pop_back();
  }
  if (a.empty()) a = {0};
  if (q.empty()) q = {0};
  return {q, a};
}

inline GFElement gf_multiply(const GFElement &a, const GFElement &b, const GFElement &g, int p) {
  return gf_poly_mod(gf_poly_mul(a, b, p), g, p);
}

inline int gf_element_eval(const GFElement &poly, int x, int p) {
  int res = 0;
  long long xp = 1;
  for (int c : poly) {
    res = (int)((res + 1LL * c * xp) % p);
    xp = (xp * x) % p;
  }
  return (res + p) % p;
}

inline GFElement gf_element_pow(GFElement base, long long exp, const GFElement &g, int p) {
  GFElement res = {1};
  while (exp > 0) {
    if (exp % 2 == 1) res = gf_multiply(res, base, g, p);
    base = gf_multiply(base, base, g, p);
    exp /= 2;
  }
  return res;
}

inline GFElement gf_poly_gcd(GFElement a, GFElement b, int p) {
  while (!gf_is_zero(b)) {
    GFElement r = gf_poly_mod(a, b, p);
    a = b; b = r;
  }
  if (!a.empty()) {
    int lead = a.back() % p;
    if (lead != 0) {
      int inv = gf_inv(lead, p);
      for (int &x : a) x = (int)((1LL * x * inv) % p);
    }
  }
  return a;
}

inline GFElement gf_poly_inv(GFElement a, const GFElement &g, int p) {
    GFElement s = {0}, old_s = {1};
    GFElement r = g, old_r = gf_poly_mod(a, g, p);
    while (!gf_is_zero(r)) {
        auto d = gf_poly_div(old_r, r, p);
        GFElement q = d.first;
        GFElement tr = r; r = d.second; old_r = tr;
        GFElement ts = s;
        s = gf_poly_sub(old_s, gf_poly_mul(q, s, p), p);
        s = gf_poly_mod(s, g, p);
        old_s = ts;
    }
    if (!gf_is_one(old_r) && !gf_is_zero(old_r)) {
        int inv_gcd = gf_inv(old_r[0], p);
        for(int &x : old_s) x = (int)((1LL * x * inv_gcd) % p);
    }
    return old_s;
}

inline bool gf_is_irreducible(const GFElement &poly, int p) {
  int n = (int)poly.size() - 1;
  if (n < 1) return false;
  if (n == 1) return true;
  if (!is_prime(p)) return false;

  GFElement rem_x_pow = {0, 1}; // x
  for (int i = 1; i <= n / 2; ++i) {
    GFElement next_rem = {1};
    GFElement base = rem_x_pow;
    int exp = p;
    while (exp > 0) {
        if (exp % 2 == 1) next_rem = gf_multiply(next_rem, base, poly, p);
        base = gf_multiply(base, base, poly, p);
        exp /= 2;
    }
    rem_x_pow = next_rem;

    GFElement neg_x(2, 0); neg_x[1] = (p - 1) % p;
    GFElement poly_diff = gf_add(rem_x_pow, neg_x, p);
    GFElement g = gf_poly_gcd(poly, poly_diff, p);
    if (!gf_is_one(g)) return false;
  }
  return true;
}

inline std::vector<int> gf_find_irreducible(int p, int n) {
  if (n == 1) return {0, 1};
  if (!is_prime(p)) return { (p % 2 == 0 ? 2 : 3), 0, 1 };

  static std::mt19937 gen(1337);
  std::uniform_int_distribution<int> dist(0, p - 1);

  std::vector<int> poly(n + 1, 0); poly[n] = 1;
  for (int attempt = 0; attempt < 2000; ++attempt) {
      for (int j = 0; j < n; ++j) poly[j] = dist(gen);
      if (gf_is_irreducible(poly, p)) return poly;
  }
  return poly;
}

inline GFElement gf_find_primitive(int p, int n, const std::vector<int> &g) {
  long long q = 1; for(int i=0; i<n; ++i) q *= p;
  long long q_m_1 = q - 1;
  std::vector<long long> factors; long long temp_q = q_m_1;
  for (long long i = 2; i * i <= temp_q; ++i) {
    if (temp_q % i == 0) { factors.push_back(i); while (temp_q % i == 0) temp_q /= i; }
  }
  if (temp_q > 1) factors.push_back(temp_q);

  static std::mt19937 gen(1338);
  std::uniform_int_distribution<int> dist(0, p - 1);

  for (int attempt = 0; attempt < 2000; ++attempt) {
    GFElement alpha(n);
    for (int j = 0; j < n; ++j) alpha[j] = dist(gen);
    if (gf_is_zero(alpha)) continue;
    bool is_p = true;
    for (long long f : factors) {
      long long exp = q_m_1 / f;
      GFElement res(1, 1); GFElement base = alpha;
      while (exp > 0) {
        if (exp % 2 == 1) res = gf_multiply(res, base, g, p);
        base = gf_multiply(base, base, g, p); exp /= 2;
      }
      if (gf_is_one(res)) { is_p = false; break; }
    }
    if (is_p) return alpha;
  }
  GFElement fallback(n, 0); if (n > 1) fallback[1] = 1; else fallback[0] = 1;
  return fallback;
}

#endif
