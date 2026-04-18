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
    GFElement alpha = {0, 1};   // x
    GFElement current = {1, 0}; // 1
    glBegin(GL_LINE_STRIP);
    glColor4f(1.0f, 0.8f, 0.0f, 0.6f);
    for (int i = 0; i < total; ++i) {
      cd z = map_to_2d(current);
      glVertex2f(z.real(), z.imag());
      current = multiply(current, alpha, g, p);
      if (is_zero(current))
        break;
    }
    glEnd();

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
  Fl_Window *win = new Fl_Window(
      900, 900, "Galois Theory: Additive vs Multiplicative Interference");
  GaloisGL *gl = new GaloisGL(10, 10, 880, 880);
  win->end();
  win->show();
  return Fl::run();
}
