// demo02.cpp
// Grothendieck Viewpoint: Varieties and Quotient Rings
//
// Shows how functions in a quotient ring GF(p)[x]/(g) are well-defined
// on the variety V(g) (the roots of g).
//
// Build (Linux):
// g++ -std=c++17 -O2 demo02.cpp -o demo02 -lfltk -lfltk_gl -lGL -lGLU -lm

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Window.H>
#include <FL/gl.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

struct GFpPoly {
  int p;
  vector<int> coeffs;

  GFpPoly(int _p) : p(_p) {}

  int eval(int x) const {
    int res = 0;
    for (int i = (int)coeffs.size() - 1; i >= 0; --i) {
      res = (res * x + coeffs[i]) % p;
    }
    return (res + p) % p;
  }
};

class VarietyGL : public Fl_Gl_Window {
public:
  int p = 11;
  vector<int> g = {-2, 0, 1}; // x^2 - 2
  vector<int> f = {1, 1};     // x + 1

  VarietyGL(int X, int Y, int W, int H, const char *L = 0)
      : Fl_Gl_Window(X, Y, W, H, L) {}

  void draw() override {
    if (!valid()) {
      valid(1);
    }
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-0.5, p - 0.5, -0.5, p - 0.5, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_logic();
  }

  void draw_logic() {
    GFpPoly poly_g(p);
    poly_g.coeffs = g;
    GFpPoly poly_f(p);
    poly_f.coeffs = f;

    // Draw all evaluations
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int x = 0; x < p; ++x) {
      int val = poly_f.eval(x);
      glColor3f(0.5, 0.5, 0.5);
      glVertex2f(x, val);
    }
    glEnd();

    // Highlight roots of g
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for (int x = 0; x < p; ++x) {
      if (poly_g.eval(x) == 0) {
        int val = poly_f.eval(x);
        glColor3f(1.0, 1.0, 0.0); // Yellow: Variety V(g)
        glVertex2f(x, val);
      }
    }
    glEnd();
  }
};

int main() {
  Fl_Window *win = new Fl_Window(800, 600, "Grothendieck Viewpoint Demo");
  VarietyGL *gl = new VarietyGL(10, 10, 780, 580);
  win->end();
  win->show();
  return Fl::run();
}
