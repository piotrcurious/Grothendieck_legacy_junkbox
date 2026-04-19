// demo01.cpp
// Educational Demo: Finite Fields GF(p) and Extensions GF(p^n)
//
// This demo showcases the "interference" between:
// 1. The additive structure (vector space over GF(p))
// 2. The multiplicative structure (cyclic group of order p^n - 1)
//
// Build (Linux):
// g++ -std=c++17 -O2 demo01.cpp -o demo01 -lfltk -lfltk_gl -lGL -lGLU -lm

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Window.H>
#include <FL/gl.h>

#include <algorithm>
#include <cmath>
#include <complex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using cd = complex<double>;

// Represents an element of GF(p^n) as a polynomial of degree < n
typedef vector<int> GFElement;

GFElement add(const GFElement &a, const GFElement &b, int p) {
  int n = a.size();
  GFElement res(n);
  for (int i = 0; i < n; ++i)
    res[i] = (a[i] + b[i]) % p;
  return res;
}

GFElement multiply(const GFElement &a, const GFElement &b, const vector<int> &g,
                   int p) {
  int n = g.size() - 1;
  vector<int> res(2 * n, 0);
  for (int i = 0; i < (int)a.size(); ++i) {
    for (int j = 0; j < (int)b.size(); ++j) {
      res[i + j] = (res[i + j] + a[i] * b[j]) % p;
    }
  }
  for (int i = (int)res.size() - 1; i >= n; --i) {
    if (res[i] == 0)
      continue;
    int factor = res[i];
    for (int j = 0; j <= n; ++j) {
      res[i - n + j] = (res[i - n + j] - factor * g[j] % p + p) % p;
    }
  }
  res.resize(n);
  return res;
}

bool is_zero(const GFElement &a) {
  for (int x : a)
    if (x != 0)
      return false;
  return true;
}

bool is_one(const GFElement &a) {
  if (a.empty() || a[0] != 1)
    return false;
  for (size_t i = 1; i < a.size(); ++i)
    if (a[i] != 0)
      return false;
  return true;
}

bool is_irreducible(const vector<int> &poly, int p) {
  int n = poly.size() - 1;
  if (n < 1)
    return false;
  if (n == 1)
    return true;

  // Brute force check for roots in extensions is hard,
  // but for small p, n we can check for factors.
  // This is a simplified check for the demo purposes.
  // For degree 2 and 3, irreducibility is equivalent to having no roots.
  if (n <= 3) {
    for (int i = 0; i < p; ++i) {
      long long val = 0;
      long long x_pow = 1;
      for (int c : poly) {
        val = (val + (long long)c * x_pow) % p;
        x_pow = (x_pow * i) % p;
      }
      if (val == 0)
        return false;
    }
    return true;
  }

  // For degree 4, also need to check for degree 2 factors.
  // First, check for roots (degree 1 factors)
  for (int i = 0; i < p; ++i) {
    long long val = 0;
    long long x_pow = 1;
    for (int c : poly) {
      val = (val + (long long)c * x_pow) % p;
      x_pow = (x_pow * i) % p;
    }
    if (val == 0)
      return false;
  }

  if (n == 4) {
    // Check all monic quadratic polynomials x^2 + ax + b
    for (int a = 0; a < p; ++a) {
      for (int b = 0; b < p; ++b) {
        // Only check irreducible quadratics (those with no roots)
        bool q_has_root = false;
        for (int i = 0; i < p; ++i) {
          if ((i * i + a * i + b) % p == 0) {
            q_has_root = true;
            break;
          }
        }
        if (q_has_root)
          continue;

        // Polynomial division P(x) / (x^2 + ax + b)
        // P = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
        // We only care if the remainder is 0.
        // Synthetic division for quadratic divisor
        vector<int> rem = poly;
        for (int i = 4; i >= 2; --i) {
          int factor = rem[i];
          rem[i - 1] = (rem[i - 1] - factor * a % p + p) % p;
          rem[i - 2] = (rem[i - 2] - factor * b % p + p) % p;
          rem[i] = 0;
        }
        if (rem[0] == 0 && rem[1] == 0)
          return false;
      }
    }
  }

  return true;
}

vector<int> find_irreducible(int p, int n) {
  if (n == 1) return {0, 1}; // x

  // Try common candidates first
  vector<vector<int>> candidates;
  if (n == 2) candidates = {{1, 1, 1}, {1, 0, 1}, {2, 0, 1}, {1, 2, 1}};
  else if (n == 3) candidates = {{1, 1, 0, 1}, {1, 0, 1, 1}, {2, 1, 0, 1}};
  else candidates = {{1, 1, 0, 0, 1}, {1, 0, 0, 1, 1}};

  for (auto& c : candidates) {
    // Adjust coefficients mod p
    for (int& x : c) x = (x % p + p) % p;
    if (c.back() != 0 && is_irreducible(c, p)) return c;
  }

  // Brute force search
  vector<int> poly(n + 1, 0);
  poly[n] = 1;
  int total_attempts = pow(p, n);
  for (int i = 0; i < total_attempts; ++i) {
    int temp = i;
    for (int j = 0; j < n; ++j) {
      poly[j] = temp % p;
      temp /= p;
    }
    if (is_irreducible(poly, p)) return poly;
  }

  return candidates[0]; // Fallback
}

GFElement find_primitive_element(int p, int n, const vector<int> &g) {
  int total = pow(p, n);
  for (int i = 1; i < total; ++i) {
    GFElement alpha(n);
    int temp = i;
    for (int j = 0; j < n; ++j) {
      alpha[j] = temp % p;
      temp /= p;
    }
    if (is_zero(alpha) || is_one(alpha))
      continue;

    GFElement current = alpha;
    int count = 1;
    while (count < total) {
      if (is_one(current))
        break;
      current = multiply(current, alpha, g, p);
      count++;
    }
    if (count == total - 1)
      return alpha;
  }
  return {0, 1}; // Fallback to x
}

class GaloisGL : public Fl_Gl_Window {
public:
  int p = 3;
  int n = 2;
  vector<int> g = {1, 0, 1}; // x^2 + 1

  GaloisGL(int X, int Y, int W, int H, const char *L = 0)
      : Fl_Gl_Window(X, Y, W, H, L) {}

  void draw() override {
    if (!valid()) {
      valid(1);
      glEnable(GL_POINT_SMOOTH);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_logic();
  }

  void draw_logic() {
    int total = pow(p, n);

    // Dynamically find a valid irreducible polynomial for the field extension
    g = find_irreducible(p, n);

    vector<GFElement> elements;
    for (int i = 0; i < total; ++i) {
      GFElement e(n);
      int temp = i;
      for (int j = 0; j < n; ++j) {
        e[j] = temp % p;
        temp /= p;
      }
      elements.push_back(e);
    }

    auto map_to_2d = [&](const GFElement &e) {
      cd z(0, 0);
      for (int j = 0; j < n; ++j) {
        z += cd(e[j], 0) * polar(1.0, 2.0 * M_PI * j / n);
      }
      return z;
    };

    // 1. Draw Additive Lattice
    glBegin(GL_LINES);
    glColor4f(0.2f, 0.4f, 0.8f, 0.15f);
    for (int i = 0; i < total; ++i) {
      for (int j = i + 1; j < total; ++j) {
        int diffs = 0;
        for (int k = 0; k < n; ++k)
          if (elements[i][k] != elements[j][k])
            diffs++;
        if (diffs == 1) {
          cd z1 = map_to_2d(elements[i]), z2 = map_to_2d(elements[j]);
          glVertex2f(z1.real(), z1.imag());
          glVertex2f(z2.real(), z2.imag());
        }
      }
    }
    glEnd();

    // 2. Multiplicative Cycle (Generator orbit)
    if (total > 1) {
      GFElement alpha = find_primitive_element(p, n, g);
      GFElement current(n, 0);
      current[0] = 1; // 1

      glBegin(GL_LINE_STRIP);
      for (int i = 0; i < total; ++i) {
        float t = (float)i / (total - 1);
        glColor4f(1.0f, 0.8f - 0.4f * t, 0.0f, 0.7f);
        cd z = map_to_2d(current);
        glVertex2f(z.real(), z.imag());
        GFElement next = multiply(current, alpha, g, p);
        if (is_one(next)) {
          cd z0 = map_to_2d(next);
          glVertex2f(z0.real(), z0.imag());
          break;
        }
        current = next;
      }
      glEnd();
    }

    // 3. Elements
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < total; ++i) {
      cd z = map_to_2d(elements[i]);
      if (is_zero(elements[i]))
        glColor3f(1, 0, 0);
      else
        glColor3f(0, 1, 1);
      glVertex2f(z.real(), z.imag());
    }
    glEnd();
  }
};

int main() {
  Fl_Window *win = new Fl_Window(1100, 900, "Galois Theory: Additive vs Multiplicative Interference");
  GaloisGL *gl = new GaloisGL(10, 10, 880, 880);

  Fl_Group *panel = new Fl_Group(900, 10, 190, 880);
  panel->box(FL_UP_BOX);

  Fl_Value_Slider *s_p = new Fl_Value_Slider(910, 40, 170, 25, "Prime p");
  s_p->type(FL_HOR_NICE_SLIDER);
  s_p->bounds(2, 7);
  s_p->step(1);
  s_p->value(3);
  s_p->callback([](Fl_Widget *w, void *v) {
    GaloisGL *g = (GaloisGL *)v;
    int val = (int)((Fl_Value_Slider *)w)->value();
    // Simple prime check
    if (val == 4) val = 3;
    if (val == 6) val = 5;
    g->p = val;
    g->redraw();
  }, gl);

  Fl_Value_Slider *s_n = new Fl_Value_Slider(910, 90, 170, 25, "Extension n");
  s_n->type(FL_HOR_NICE_SLIDER);
  s_n->bounds(1, 4);
  s_n->step(1);
  s_n->value(2);
  s_n->callback([](Fl_Widget *w, void *v) {
    GaloisGL *g = (GaloisGL *)v;
    g->n = (int)((Fl_Value_Slider *)w)->value();
    g->redraw();
  }, gl);

  Fl_Box *info = new Fl_Box(910, 150, 170, 100, "Yellow line: Multiplicative orbit\nBlue lines: Additive lattice");
  info->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_TOP);

  panel->end();
  win->end();
  win->resizable(gl);
  win->show();
  return Fl::run();
}
