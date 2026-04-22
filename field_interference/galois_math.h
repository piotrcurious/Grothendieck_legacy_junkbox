#ifndef GALOIS_MATH_H
#define GALOIS_MATH_H

#include <vector>
#include <cmath>
#include <algorithm>

typedef std::vector<int> GFElement;

inline int gf_inv(int n, int p) {
  int a = n % p, b = p, x0 = 0, x1 = 1;
  while (a > 1) {
    int q = a / b, t = b;
    b = a % b; a = t;
    t = x0; x0 = x1 - q * x0; x1 = t;
  }
  return (x1 < 0) ? x1 + p : x1;
}

inline GFElement gf_add(const GFElement &a, const GFElement &b, int p) {
  int n = std::max(a.size(), b.size());
  GFElement res(n, 0);
  for (int i = 0; i < n; ++i) {
    int va = (i < (int)a.size()) ? a[i] : 0;
    int vb = (i < (int)b.size()) ? b[i] : 0;
    res[i] = (va + vb) % p;
  }
  return res;
}

inline GFElement gf_poly_mul(const GFElement &a, const GFElement &b, int p) {
  if (a.empty() || b.empty()) return {};
  GFElement res(a.size() + b.size() - 1, 0);
  for (int i = 0; i < (int)a.size(); ++i) {
    for (int j = 0; j < (int)b.size(); ++j) {
      res[i + j] = (res[i + j] + a[i] * b[j]) % p;
    }
  }
  while (res.size() > 1 && res.back() == 0) res.pop_back();
  return res;
}

inline GFElement gf_poly_mod(GFElement a, const GFElement &g, int p) {
  int n = g.size() - 1;
  int inv_lead = gf_inv(g.back(), p);
  while ((int)a.size() > n) {
    int factor = (a.back() * inv_lead) % p;
    int deg_diff = a.size() - 1 - n;
    for (int i = 0; i <= n; ++i) {
      int idx = i + deg_diff;
      a[idx] = (a[idx] - factor * g[i] % p + p) % p;
    }
    while (!a.empty() && a.back() == 0) a.pop_back();
  }
  if (a.empty()) return {0};
  return a;
}

inline GFElement gf_multiply(const GFElement &a, const GFElement &b, const GFElement &g, int p) {
  return gf_poly_mod(gf_poly_mul(a, b, p), g, p);
}

inline GFElement gf_poly_gcd(GFElement a, GFElement b, int p) {
  while (b.size() > 1 || (b.size() == 1 && b[0] != 0)) {
    GFElement r = gf_poly_mod(a, b, p);
    a = b; b = r;
  }
  // Normalize
  if (!a.empty()) {
    int inv = gf_inv(a.back(), p);
    for (int &x : a) x = (x * inv) % p;
  }
  return a;
}

inline bool gf_is_one(const GFElement &a) {
  return a.size() == 1 && a[0] == 1;
}

inline bool gf_is_zero(const GFElement &a) {
  return a.empty() || (a.size() == 1 && a[0] == 0);
}

inline bool gf_is_irreducible(const GFElement &poly, int p) {
  int n = poly.size() - 1;
  if (n < 1) return false;
  if (n == 1) return true;
  // Rabin's test or Ben-Or's: check gcd(poly, x^(p^i) - x)
  // For simplicity and our range (n <= 5), check all irreducible factors of degree up to n/2
  for (int i = 1; i <= n / 2; ++i) {
    // Compute x^(p^i) mod poly
    GFElement x_pow(2, 0); x_pow[1] = 1; // x
    GFElement res(1, 1); // 1
    long long exp = 1;
    for (int k = 0; k < i; ++k) exp *= p;

    GFElement base = x_pow;
    long long cur_exp = exp;
    GFElement rem_x_pow(1, 1);
    while (cur_exp > 0) {
      if (cur_exp % 2 == 1) rem_x_pow = gf_multiply(rem_x_pow, base, poly, p);
      base = gf_multiply(base, base, poly, p);
      cur_exp /= 2;
    }
    // gcd(poly, x^(p^i) - x)
    // poly_diff = rem_x_pow - x
    GFElement neg_x(2, 0); neg_x[1] = (p - 1) % p;
    GFElement poly_diff = gf_add(rem_x_pow, neg_x, p);

    GFElement g = gf_poly_gcd(poly, poly_diff, p);
    if (!gf_is_one(g)) return false;
  }
  return true;
}

inline std::vector<int> gf_find_irreducible(int p, int n) {
  if (n == 1) return {0, 1};
  std::vector<int> poly(n + 1, 0); poly[n] = 1;
  long long total = 1; for(int i=0; i<n; ++i) total *= p;
  if (total > 20000) total = 20000;
  for (long long i = 1; i < total; ++i) {
    long long temp = i;
    for (int j = 0; j < n; ++j) { poly[j] = temp % p; temp /= p; }
    if (gf_is_irreducible(poly, p)) return poly;
  }
  return poly;
}

inline GFElement gf_find_primitive(int p, int n, const std::vector<int> &g) {
  long long total = 1; for(int i=0; i<n; ++i) total *= p;
  long long q_minus_1 = total - 1;
  std::vector<long long> factors; long long temp_q = q_minus_1;
  for (long long i = 2; i * i <= temp_q; ++i) {
    if (temp_q % i == 0) { factors.push_back(i); while (temp_q % i == 0) temp_q /= i; }
  }
  if (temp_q > 1) factors.push_back(temp_q);
  for (long long i = 1; i < total; ++i) {
    GFElement alpha(n); long long temp = i;
    for (int j = 0; j < n; ++j) { alpha[j] = temp % p; temp /= p; }
    if (gf_is_zero(alpha)) continue;
    bool is_p = true;
    for (long long f : factors) {
      long long exp = q_minus_1 / f;
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
