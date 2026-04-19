#ifndef GALOIS_MATH_H
#define GALOIS_MATH_H

#include <vector>
#include <cmath>
#include <algorithm>

typedef std::vector<int> GFElement;

inline GFElement gf_add(const GFElement &a, const GFElement &b, int p) {
  int n = a.size();
  GFElement res(n);
  for (int i = 0; i < n; ++i) res[i] = (a[i] + b[i]) % p;
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
  for (int i = (int)res.size() - 1; i >= n; --i) {
    if (res[i] == 0) continue;
    int factor = res[i];
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
  if (a.empty() || a[0] != 1) return false;
  for (size_t i = 1; i < a.size(); ++i) if (a[i] != 0) return false;
  return true;
}

inline bool gf_is_irreducible(const std::vector<int> &poly, int p) {
  int n = poly.size() - 1;
  if (n < 1) return false;
  if (n == 1) return true;
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
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b < p; ++b) {
        bool q_has_root = false;
        for (int i = 0; i < p; ++i) {
          if ((i * i + a * i + b) % p == 0) { q_has_root = true; break; }
        }
        if (q_has_root) continue;
        std::vector<int> rem = poly;
        for (int i = 4; i >= 2; --i) {
          int factor = rem[i];
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
  if (n == 1) return {0, 1};
  std::vector<int> poly(n + 1, 0);
  poly[n] = 1;
  int total_attempts = std::pow(p, n);
  for (int i = 1; i < total_attempts; ++i) {
    int temp = i;
    for (int j = 0; j < n; ++j) {
      poly[j] = temp % p;
      temp /= p;
    }
    if (gf_is_irreducible(poly, p)) return poly;
  }
  return poly;
}

inline GFElement gf_find_primitive(int p, int n, const std::vector<int> &g) {
  int total = std::pow(p, n);
  for (int i = 1; i < total; ++i) {
    GFElement alpha(n);
    int temp = i;
    for (int j = 0; j < n; ++j) { alpha[j] = temp % p; temp /= p; }
    if (gf_is_zero(alpha) || gf_is_one(alpha)) continue;
    GFElement current = alpha;
    int count = 1;
    while (count < total) {
      if (gf_is_one(current)) break;
      current = gf_multiply(current, alpha, g, p);
      count++;
    }
    if (count == total - 1) return alpha;
  }
  GFElement fallback(n, 0);
  if (n > 1) fallback[1] = 1; else fallback[0] = 1;
  return fallback;
}

#endif
