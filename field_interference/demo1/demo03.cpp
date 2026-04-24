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
#include <FL/fl_draw.H>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../complex_math.h"

using namespace std;

struct IntPoly {
  vector<int> coeffs;
  IntPoly(int deg = 0) { coeffs.assign(max(1, deg + 1), 0); }
  int deg() const {
    for (int i = (int)coeffs.size() - 1; i >= 0; --i) if (coeffs[i] != 0) return i;
    return 0;
  }
};

vector<double> get_companion_matrix(const IntPoly &P) {
  int n = P.deg();
  if (n < 1) return {};
  vector<double> M(n * n, 0.0);
  for (int i = 1; i < n; i++) M[i * n + (i - 1)] = 1.0;
  for (int i = 0; i < n; i++) {
    double a = (i < (int)P.coeffs.size() - 1) ? (double)P.coeffs[i] : 0.0;
    M[i * n + (n - 1)] = -a / (double)P.coeffs.back();
  }
  return M;
}

class ShowcaseGL : public Fl_Gl_Window {
public:
  IntPoly current_poly;
  vector<cd> roots;
  vector<double> matrix;
  double trace = 0, det = 0;

  ShowcaseGL(int X, int Y, int W, int H)
      : Fl_Gl_Window(X, Y, W, H), current_poly(2) {
    current_poly.coeffs = {-2, 0, 1};
    update_data();
  }

  void update_data() {
    vector<cd> c_cd; for (int c : current_poly.coeffs) c_cd.push_back(cd(c, 0));
    roots = dk_solve_roots(c_cd);
    matrix = get_companion_matrix(current_poly);
    int n = current_poly.deg();
    if (n >= 1) {
        trace = - (double)current_poly.coeffs[n-1] / current_poly.coeffs[n];
        det = (n % 2 == 0 ? 1 : -1) * (double)current_poly.coeffs[0] / current_poly.coeffs[n];
    }
    redraw();
  }

  void draw() override {
    if (!valid()) {
      valid(1); glEnable(GL_POINT_SMOOTH); glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
    glClearColor(0.05f, 0.05f, 0.08f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
    int W = w(), H = h();
    glViewport(0, 0, W / 2, H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_complex_plane();

    glViewport(W / 2, 0, W / 2, H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, 1, 0, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_matrix_view();
  }

  void draw_complex_plane() {
    glBegin(GL_LINES); glColor4f(1, 1, 1, 0.2f);
    glVertex2f(-10, 0); glVertex2f(10, 0); glVertex2f(0, -10); glVertex2f(0, 10);
    glEnd();
    glBegin(GL_LINE_LOOP); glColor4f(0.5, 0.5, 1.0, 0.3f);
    for (int i = 0; i < 100; ++i) { double a = 2.0 * M_PI * i / 100.0; glVertex2f(cos(a), sin(a)); }
    glEnd();
    glPointSize(10.0f); glBegin(GL_POINTS);
    for (auto &r : roots) { glColor3f(1, 1, 0); glVertex2f(r.real(), r.imag()); }
    glEnd();
    draw_text(-2.3, 2.3, "Roots of Polynomial");
  }

  void draw_matrix_view() {
    int n = current_poly.deg(); if (n < 1) return;
    float size = min(0.8f / n, 0.2f); float sx = 0.1f, sy = 0.85f;
    gl_font(FL_HELVETICA, 12);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        float x = sx + j * size, y = sy - i * size;
        glColor3f(0.1f, 0.2f, 0.3f); glBegin(GL_QUADS);
        glVertex2f(x, y); glVertex2f(x + size*0.9f, y); glVertex2f(x + size*0.9f, y - size*0.9f); glVertex2f(x, y - size*0.9f);
        glEnd();
        glColor3f(0.9, 0.9, 1.0);
        ostringstream ss; ss << fixed << setprecision(1) << matrix[i * n + j];
        glRasterPos2f(x + size*0.1f, y - size*0.6f);
        gl_draw(ss.str().c_str());
      }
    }
    draw_text(0.1, 0.92, "Companion Matrix M (Eigenvalues = Roots)");
    ostringstream ss; ss << fixed << setprecision(2) << "Trace: " << trace << "  Det: " << det;
    draw_text(0.1, 0.05, ss.str());
  }

  void draw_text(float x, float y, const string &s) {
    glColor3f(0.8, 0.8, 1.0); gl_font(FL_HELVETICA, 14);
    glRasterPos2f(x, y); gl_draw(s.c_str());
  }
};

struct UI { ShowcaseGL* gl; Fl_Input* in; };

int main(int argc, char **argv) {
  Fl_Window *win = new Fl_Window(1200, 600, "Companion Matrix & Eigenvalue Visualization");
  ShowcaseGL *gl = new ShowcaseGL(10, 10, 1180, 500);
  Fl_Group *grp = new Fl_Group(10, 520, 1180, 70);
  Fl_Input *inp = new Fl_Input(100, 530, 300, 30, "Coeffs:"); inp->value("-2, 0, 1");
  UI *ui = new UI{gl, inp};
  Fl_Button *upd = new Fl_Button(420, 530, 150, 30, "Update");
  upd->callback([](Fl_Widget *, void *v) {
    UI *u = (UI *)v; string s = u->in->value();
    for(auto &c:s) if(c==',') c=' ';
    stringstream ss(s); int val; vector<int> co;
    while(ss >> val) co.push_back(val);
    if (co.size() >= 2) { u->gl->current_poly.coeffs = co; u->gl->update_data(); }
  }, ui);
  grp->end(); win->end(); win->resizable(gl); win->show(); return Fl::run();
}
