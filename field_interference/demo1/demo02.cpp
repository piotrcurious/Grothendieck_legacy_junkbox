// demo02.cpp
// Grothendieck Viewpoint: Varieties and Quotient Rings
//
// Shows how functions in a quotient ring GF(p)[x]/(g) are well-defined
// on the variety V(g) (the roots of g).

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Window.H>
#include <FL/gl.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../complex_math.h"

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
  bool show_splitting = false;

  VarietyGL(int X, int Y, int W, int H, const char *L = 0)
      : Fl_Gl_Window(X, Y, W, H, L) {}

  void draw() override {
    if (!valid()) { valid(1); }
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(-0.5, p - 0.5, -0.5, p - 0.5, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_logic();
  }

  void draw_logic() {
    GFpPoly poly_g(p); poly_g.coeffs = g;
    GFpPoly poly_f(p); poly_f.coeffs = f;

    if (!show_splitting) {
      glPointSize(5.0f); glBegin(GL_POINTS);
      for (int x = 0; x < p; ++x) {
        int val = poly_f.eval(x); glColor3f(0.5, 0.5, 0.5); glVertex2f(x, val);
      }
      glEnd();
      glPointSize(10.0f); glBegin(GL_POINTS);
      for (int x = 0; x < p; ++x) {
        if (poly_g.eval(x) == 0) {
          int val = poly_f.eval(x); glColor3f(1.0, 1.0, 0.0); glVertex2f(x, val);
        }
      }
      glEnd();
    } else {
      glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-3, 3, -3, 3, -1, 1);
      glBegin(GL_LINES); glColor4f(1, 1, 1, 0.2f);
      glVertex2f(-10, 0); glVertex2f(10, 0); glVertex2f(0, -10); glVertex2f(0, 10);
      glEnd();
      vector<cd> g_coeffs; for (int c : g) g_coeffs.push_back(cd(c, 0));
      auto roots = dk_solve_roots(g_coeffs);
      glPointSize(12.0f); glBegin(GL_POINTS);
      for (auto &r : roots) {
        vector<cd> f_coeffs; for (int c : f) f_coeffs.push_back(cd(c, 0));
        cd fr = complex_eval_poly(f_coeffs, r);
        glColor3f(1, 1, 0); glVertex2f(r.real(), r.imag());
        glColor3f(0, 1, 1); glVertex2f(fr.real(), fr.imag());
      }
      glEnd();
      glLineWidth(1.5f); glBegin(GL_LINES);
      for (auto &r : roots) {
        vector<cd> f_coeffs; for (int c : f) f_coeffs.push_back(cd(c, 0));
        cd fr = complex_eval_poly(f_coeffs, r);
        glColor4f(1, 1, 1, 0.4f); glVertex2f(r.real(), r.imag()); glVertex2f(fr.real(), fr.imag());
      }
      glEnd();
    }
  }
};

struct UIContext { VarietyGL *gl; Fl_Input *g_in; Fl_Input *f_in; };

int main() {
  Fl_Window *win = new Fl_Window(1100, 800, "Grothendieck Viewpoint: Varieties and Quotients");
  VarietyGL *gl = new VarietyGL(10, 10, 780, 780);
  Fl_Group *ctrl = new Fl_Group(800, 10, 290, 780); ctrl->box(FL_UP_BOX);
  Fl_Value_Slider *s_p = new Fl_Value_Slider(810, 40, 270, 25, "Field Prime p");
  s_p->type(FL_HOR_NICE_SLIDER); s_p->bounds(2, 101); s_p->value(11);
  Fl_Input *g_inp = new Fl_Input(810, 100, 270, 25, "Poly g(x)"); g_inp->value("-2, 0, 1");
  Fl_Input *f_inp = new Fl_Input(810, 160, 270, 25, "Func f(x)"); f_inp->value("1, 1");
  UIContext *ctx = new UIContext{gl, g_inp, f_inp};
  auto upd = [](Fl_Widget *, void *v) {
    UIContext *c = (UIContext *)v;
    auto parse = [](string s) {
      vector<int> res; stringstream ss(s); string t;
      while(getline(ss, t, ',')) { try { res.push_back(stoi(t)); } catch(...) {} }
      return res;
    };
    c->gl->g = parse(c->g_in->value()); c->gl->f = parse(c->f_in->value()); c->gl->redraw();
  };
  s_p->callback([](Fl_Widget *w, void *v) { ((VarietyGL *)v)->p = (int)((Fl_Value_Slider *)w)->value(); ((VarietyGL *)v)->redraw(); }, gl);
  Fl_Button *upd_btn = new Fl_Button(810, 200, 270, 30, "Update Polynomials");
  upd_btn->callback(upd, ctx);
  Fl_Check_Button *split = new Fl_Check_Button(810, 250, 270, 25, "Splitting Field (C)");
  split->callback([](Fl_Widget *w, void *v) { ((VarietyGL *)v)->show_splitting = ((Fl_Check_Button *)w)->value(); ((VarietyGL *)v)->redraw(); }, gl);
  ctrl->end(); win->end(); win->resizable(gl); win->show(); return Fl::run();
}
