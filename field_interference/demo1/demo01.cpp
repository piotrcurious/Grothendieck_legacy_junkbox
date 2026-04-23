#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Check_Button.H>
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
#include "../galois_math.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace std;
using cd = complex<double>;

class GaloisGL : public Fl_Gl_Window {
public:
  int p = 3;
  int n = 2;
  bool show_inverse = false;

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
    int total = (int)pow(p, n);
    if (total > 5000) total = 5000;

    vector<int> g = gf_find_irreducible(p, n);

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
      if (n == 1) {
        z = cd(e[0] - p/2.0, 0);
      } else {
        for (int j = 0; j < n; ++j) {
          z += cd(e[j], 0) * polar(1.0, 2.0 * M_PI * j / n);
        }
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
    if (total > 1 && total < 2000) {
      GFElement alpha = gf_find_primitive(p, n, g);
      GFElement current(n, 0);
      current[0] = 1; // element '1'

      glBegin(GL_LINE_STRIP);
      for (int i = 0; i < total; ++i) {
        float t = (float)i / (total - 1);
        glColor4f(1.0f, 0.8f - 0.4f * t, 0.0f, 0.7f);
        cd z = map_to_2d(current);
        glVertex2f(z.real(), z.imag());
        GFElement next = gf_multiply(current, alpha, g, p);
        if (gf_is_one(next)) {
          cd z0 = map_to_2d(next);
          glVertex2f(z0.real(), z0.imag());
          break;
        }
        current = next;
      }
      glEnd();
    }

    // 3. Elements and potentially their inverses
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < total; ++i) {
      cd z = map_to_2d(elements[i]);
      if (gf_is_zero(elements[i]))
        glColor3f(1, 0, 0);
      else
        glColor3f(0, 1, 1);
      glVertex2f(z.real(), z.imag());

      if (show_inverse && !gf_is_zero(elements[i])) {
          GFElement inv_e = gf_poly_inv(elements[i], g, p);
          cd z_inv = map_to_2d(inv_e);
          glColor4f(1, 0.5, 0, 0.5);
          glVertex2f(z_inv.real(), z_inv.imag());
      }
    }
    glEnd();

    if (show_inverse) {
        glBegin(GL_LINES);
        glColor4f(1, 1, 0, 0.1);
        for (int i = 0; i < total; ++i) {
            if (gf_is_zero(elements[i])) continue;
            cd z = map_to_2d(elements[i]);
            GFElement inv_e = gf_poly_inv(elements[i], g, p);
            cd z_inv = map_to_2d(inv_e);
            glVertex2f(z.real(), z.imag());
            glVertex2f(z_inv.real(), z_inv.imag());
        }
        glEnd();
    }
  }
};

int main(int argc, char **argv) {
  Fl_Window *win = new Fl_Window(1100, 900, "Galois Theory: Additive vs Multiplicative Interference");
  GaloisGL *gl = new GaloisGL(10, 10, 880, 880);

  Fl_Group *panel = new Fl_Group(900, 10, 190, 880);
  panel->box(FL_UP_BOX);

  Fl_Value_Slider *s_p = new Fl_Value_Slider(910, 40, 170, 25, "Prime p");
  s_p->type(FL_HOR_NICE_SLIDER);
  s_p->bounds(2, 31);
  s_p->step(1);
  s_p->value(3);

  Fl_Box *info = new Fl_Box(910, 150, 170, 100, "Yellow line: Multiplicative orbit\nBlue lines: Additive lattice");
  info->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_TOP);

  Fl_Box *prime_warn = new Fl_Box(910, 260, 170, 30, "");
  prime_warn->labelcolor(FL_RED);
  auto check_p = [](Fl_Value_Slider *s, Fl_Box *b) {
    if (!is_prime((int)s->value())) b->label("Warning: p is not prime!");
    else b->label("");
  };
  check_p(s_p, prime_warn);
  s_p->callback([](Fl_Widget *w, void *v) {
    auto *gl_win = (GaloisGL *)v;
    gl_win->p = (int)((Fl_Value_Slider *)w)->value();
    Fl_Box *b = (Fl_Box *)w->parent()->child(3);
    if (!is_prime(gl_win->p)) b->label("Warning: p is not prime!");
    else b->label("");
    gl_win->redraw();
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

  Fl_Check_Button *chk_inv = new Fl_Check_Button(910, 300, 170, 25, "Show Inverses");
  chk_inv->callback([](Fl_Widget *w, void *v) {
      ((GaloisGL *)v)->show_inverse = ((Fl_Check_Button *)w)->value();
      ((GaloisGL *)v)->redraw();
  }, gl);

  panel->end();
  win->end();
  win->resizable(gl);
  win->show(argc, argv);
  return Fl::run();
}
