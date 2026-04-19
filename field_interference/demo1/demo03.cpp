// demo03.cpp
// C++ Educational Demo: Field Extensions, Companion Matrices and Ring Scopes
//
// This demo visualizes the hierarchy of number systems:
// Z (Integers) -> Q (Rationals) -> R (Reals) -> C (Complex)
// and the algebraic extensions Q(alpha) formed by adjoining roots of
// polynomials.
//
// Key Feature: Visualizes the relationship between a polynomial's companion
// matrix and its roots (eigenvalues).
//
// Build (Linux):
// g++ -std=c++17 -O2 demo03.cpp -o demo03 -lfltk -lfltk_gl -lGL -lGLU -lm

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using cd = complex<double>;

// --- Mathematical Utilities ---

struct IntPoly {
  vector<int> coeffs; // low to high
  IntPoly(int deg = 0) { coeffs.assign(max(1, deg + 1), 0); }
  int deg() const {
    for (int i = (int)coeffs.size() - 1; i >= 0; --i)
      if (coeffs[i] != 0)
        return i;
    return 0;
  }
  string to_string() const {
    if (coeffs.empty())
      return "0";
    ostringstream ss;
    bool first = true;
    for (int i = (int)coeffs.size() - 1; i >= 0; --i) {
      if (coeffs[i] == 0)
        continue;
      if (!first && coeffs[i] > 0)
        ss << " + ";
      else if (coeffs[i] < 0)
        ss << " - ";
      if (abs(coeffs[i]) != 1 || i == 0)
        ss << abs(coeffs[i]);
      if (i > 0) {
        ss << "x";
        if (i > 1)
          ss << "^" << i;
      }
      first = false;
    }
    return first ? "0" : ss.str();
  }
};

vector<double> companion_matrix(const IntPoly &P) {
  int n = P.deg();
  if (n < 1)
    return {};
  vector<double> M(n * n, 0.0);
  for (int i = 1; i < n; i++)
    M[i * n + (i - 1)] = 1.0;
  for (int i = 0; i < n; i++) {
    // Companion matrix for monic x^n + a_{n-1}x^{n-1} +   + a0
    // We assume P is monic for this educational demo
    double a = (i < (int)P.coeffs.size() - 1) ? (double)P.coeffs[i] : 0.0;
    M[i * n + (n - 1)] = -a / (double)P.coeffs.back();
  }
  return M;
}

vector<cd> durand_kerner_roots(const vector<cd> &coeffs) {
  int n = (int)coeffs.size() - 1;
  if (n <= 0)
    return {};
  vector<cd> roots(n);
  for (int i = 0; i < n; i++)
    roots[i] = std::polar(1.5, 2.0 * M_PI * i / n + 0.1);
  for (int iter = 0; iter < 200; ++iter) {
    double max_change = 0.0;
    for (int i = 0; i < n; ++i) {
      cd xi = roots[i];
      cd p = coeffs[n];
      for (int k = n - 1; k >= 0; --k)
        p = p * xi + coeffs[k];
      cd prod = 1.0;
      for (int j = 0; j < n; ++j)
        if (j != i)
          prod *= (xi - roots[j]);
      if (abs(prod) < 1e-18)
        prod = 1e-18;
      cd delta = p / prod;
      roots[i] -= delta;
      max_change = max(max_change, abs(delta));
    }
    if (max_change < 1e-12)
      break;
  }
  return roots;
}

// --- GUI Components ---

class ShowcaseGL : public Fl_Gl_Window {
public:
  IntPoly current_poly;
  vector<cd> roots;
  vector<double> matrix;

  ShowcaseGL(int X, int Y, int W, int H, const char *L = 0)
      : Fl_Gl_Window(X, Y, W, H, L), current_poly(2) {
    current_poly.coeffs = {-2, 0, 1}; // x^2 - 2
    update_data();
  }

  void update_data() {
    vector<cd> c_cd;
    for (int c : current_poly.coeffs)
      c_cd.push_back(cd(c, 0));
    roots = durand_kerner_roots(c_cd);
    matrix = companion_matrix(current_poly);
    redraw();
  }

  void draw() override {
    if (!valid()) {
      valid(1);
      glEnable(GL_POINT_SMOOTH);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    int W = w(), H = h();

    // 1. Complex Plane View (Roots)
    glViewport(0, 0, W / 2, H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_complex_plane();

    // 2. Companion Matrix View
    glViewport(W / 2, 0, W / 2, H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    draw_matrix_view();
  }

  void draw_complex_plane() {
    // Axes
    glBegin(GL_LINES);
    glColor4f(1, 1, 1, 0.2f);
    glVertex2f(-10, 0);
    glVertex2f(10, 0);
    glVertex2f(0, -10);
    glVertex2f(0, 10);
    glEnd();

    // Unit circle
    glBegin(GL_LINE_LOOP);
    glColor4f(0.5, 0.5, 1.0, 0.3f);
    for (int i = 0; i < 100; ++i) {
      double a = 2.0 * M_PI * i / 100.0;
      glVertex2f(cos(a), sin(a));
    }
    glEnd();

    // Roots
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for (auto &r : roots) {
      glColor3f(1, 1, 0);
      glVertex2f(r.real(), r.imag());
    }
    glEnd();

    draw_text(-2.3, 2.3, "Roots of " + current_poly.to_string());
  }

  void draw_matrix_view() {
    int n = current_poly.deg();
    if (n < 1)
      return;

    float size = min(0.6f / n, 0.15f);
    float start_x = 0.1f;
    float start_y = 0.85f;

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        float x = start_x + j * size;
        float y = start_y - i * size;
        draw_rect(x, y, size * 0.9f, size * 0.9f, 0.1f, 0.2f, 0.3f);
        double val = matrix[i * n + j];
        ostringstream ss;
        ss << fixed << setprecision(1) << val;
        draw_text(x + size * 0.1f, y - size * 0.5f, ss.str());
      }
    }
    draw_text(0.1, 0.92, "Companion Matrix M (Eigenvalues = Roots)");

    // List computed roots as eigenvalues
    float text_y = start_y - n * size - 0.1f;
    draw_text(0.1, text_y, "Eigenvalues computed via Durand-Kerner:");
    for (size_t i = 0; i < roots.size(); ++i) {
      ostringstream ss;
      ss << "λ_" << i << " = " << fixed << setprecision(3) << roots[i].real()
         << (roots[i].imag() >= 0 ? " + " : " - ") << abs(roots[i].imag()) << "i";
      draw_text(0.15, text_y - (i + 1) * 0.05f, ss.str());
    }
  }

  void draw_text(float x, float y, const string &s) {
    glColor3f(0.9, 0.9, 1.0);
    gl_font(FL_HELVETICA, 14);
    glRasterPos2f(x, y);
    gl_draw(s.c_str());
  }

  void draw_rect(float x, float y, float w, float h, float r, float g,
                 float b) {
    glColor3f(r, g, b);
    glBegin(GL_QUADS);
    glVertex2f(x, y);
    glVertex2f(x + w, y);
    glVertex2f(x + w, y - h);
    glVertex2f(x, y - h);
    glEnd();
  }
};

int main() {
  Fl_Window *win = new Fl_Window(
      1200, 600, "Algebraic Field Extensions: Companion Matrix & Roots");
  ShowcaseGL *gl = new ShowcaseGL(10, 10, 1180, 500);

  Fl_Group *grp = new Fl_Group(10, 520, 1180, 70);
  Fl_Input *inp = new Fl_Input(100, 530, 300, 30, "Coeffs:");
  inp->value("-2, 0, 1");
  inp->tooltip("Comma separated coefficients from low to high degree. e.g. -2, "
               "0, 1 for x^2 - 2");

  Fl_Button *btn = new Fl_Button(420, 530, 150, 30, "Update Polynomial");
  btn->callback(
      [](Fl_Widget *w, void *v) {
        ShowcaseGL *g = (ShowcaseGL *)v;
        Fl_Input *i = (Fl_Input *)w->parent()->child(0);
        string s = i->value();
        vector<int> co;
        stringstream ss(s);
        string t;
        while (getline(ss, t, ',')) {
          try {
            co.push_back(stoi(t));
          } catch (...) {
          }
        }
        if (co.size() >= 2) {
          g->current_poly.coeffs = co;
          g->update_data();
        }
      },
      gl);

  Fl_Box *help =
      new Fl_Box(600, 530, 500, 30,
                 "Yellow dots are roots in C. Matrix is the companion matrix.");
  help->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);

  grp->end();
  win->end();
  win->resizable(gl);
  win->show();
  return Fl::run();
}
