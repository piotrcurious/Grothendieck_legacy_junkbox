#ifndef GALOIS_MATH_H
#define GALOIS_MATH_H

#include <vector>
#include <cmath>
#include <algorithm>

typedef std::vector<int> GFElement;

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

inline GFElement gf_multiply(const GFElement &a, const GFElement &b, const std::vector<int> &g, int p) {
  int n = g.size() - 1;
  std::vector<int> res(2 * n, 0);
  for (int i = 0; i < (int)a.size(); ++i) {
    for (int j = 0; j < (int)b.size(); ++j) {
      res[i + j] = (res[i + j] + a[i] * b[j]) % p;
    }
  }
  // Polynomial reduction modulo g
  for (int i = (int)res.size() - 1; i >= n; --i) {
    if (res[i] == 0) continue;
    int factor = res[i];
    // We assume g is monic (g[n] == 1). If not, we'd need inverse of g[n] mod p.
    // For our purposes, find_irreducible returns monic.
    for (int j = 0; j <= n; ++j) {
      res[i - n + j] = (res[i - n + j] - factor * g[j] % p + p) % p;
    }
  }
  res.resize(n);
  return res;
}

inline bool gf_is_zero(const GFElement &a) {
  for (int x : a) if (x != 0) return false;
  return true;
}

inline bool gf_is_one(const GFElement &a) {
  if (a.empty()) return false;
  if (a[0] != 1) return false;
  for (size_t i = 1; i < a.size(); ++i) if (a[i] != 0) return false;
  return true;
}

inline bool gf_is_irreducible(const std::vector<int> &poly, int p) {
  int n = poly.size() - 1;
  if (n < 1) return false;
  if (n == 1) return true;

  // Berlekamp's algorithm or Ben-Or's algorithm would be better for high n,
  // but for n <= 4, brute force or checking lower degree factors is fine.

  // Check roots (degree 1 factors)
  for (int i = 0; i < p; ++i) {
    long long val = 0;
    long long x_pow = 1;
    for (int c : poly) {
      val = (val + (long long)c * x_pow) % p;
      x_pow = (x_pow * i) % p;
    }
    if (val == 0) return false;
  }

  if (n == 4) {
    // Check for irreducible quadratic factors: x^2 + ax + b
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b < p; ++b) {
        // Only check irreducible quadratics (those with no roots)
        bool has_root = false;
        for (int i = 0; i < p; ++i) {
          if ((i * i + a * i + b) % p == 0) { has_root = true; break; }
        }
        if (has_root) continue;

        // Try dividing poly by (x^2 + ax + b)
        std::vector<int> rem = poly;
        for (int i = 4; i >= 2; --i) {
          int factor = rem[i];
          if (factor == 0) continue;
          rem[i - 1] = (rem[i - 1] - factor * a % p + p) % p;
          rem[i - 2] = (rem[i - 2] - factor * b % p + p) % p;
          rem[i] = 0;
        }
        if (rem[0] == 0 && rem[1] == 0) return false;
      }
    }
  }
  return true;
}

inline std::vector<int> gf_find_irreducible(int p, int n) {
  std::vector<int> poly(n + 1, 0);
  poly[n] = 1; // Monic

  // Try some simple ones first
  if (n == 1) return {0, 1}; // x is technically not irreducible if we want a field of size p? No, GF(p) is just Z/p.

  long long total_attempts = 1;
  for(int i=0; i<n; ++i) total_attempts *= p;

  for (long long i = 1; i < total_attempts; ++i) {
    long long temp = i;
    for (int j = 0; j < n; ++j) {
      poly[j] = temp % p;
      temp /= p;
    }
    if (gf_is_irreducible(poly, p)) return poly;
  }
  // Fallback (should not be reached for small p, n)
  return poly;
}

inline GFElement gf_find_primitive(int p, int n, const std::vector<int> &g) {
  long long total = 1;
  for(int i=0; i<n; ++i) total *= p;

  // Order of multiplicative group is q - 1
  long long q_minus_1 = total - 1;

  // Find prime factors of q_minus_1 for a faster check
  std::vector<long long> factors;
  long long temp_q = q_minus_1;
  for (long long i = 2; i * i <= temp_q; ++i) {
    if (temp_q % i == 0) {
      factors.push_back(i);
      while (temp_q % i == 0) temp_q /= i;
    }
  }
  if (temp_q > 1) factors.push_back(temp_q);

  for (long long i = 1; i < total; ++i) {
    GFElement alpha(n);
    long long temp = i;
    for (int j = 0; j < n; ++j) { alpha[j] = temp % p; temp /= p; }
    if (gf_is_zero(alpha)) continue;

    bool is_primitive = true;
    for (long long f : factors) {
      long long exp = q_minus_1 / f;
      // Compute alpha^exp
      GFElement res(n, 0); res[0] = 1;
      GFElement base = alpha;
      long long curr_exp = exp;
      while (curr_exp > 0) {
        if (curr_exp % 2 == 1) res = gf_multiply(res, base, g, p);
        base = gf_multiply(base, base, g, p);
        curr_exp /= 2;
      }
      if (gf_is_one(res)) {
        is_primitive = false;
        break;
      }
    }
    if (is_primitive) return alpha;
  }

  GFElement fallback(n, 0);
  if (n > 1) fallback[1] = 1; else fallback[0] = 1;
  return fallback;
}

#endif
