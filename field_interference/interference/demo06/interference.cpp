// Unified Field Explorer (Enhanced Production Version)
//
// Description:
//   A high-performance C++ tool using FLTK and OpenGL to visualize:
//   1. Algebraic Numbers: Root density heatmaps of random polynomials.
//   2. Finite Fields: Visual representations of GF(p^n) and their multiplicative cycles.
//   3. Field Extensions: Lattice grids formed by adjoining roots.

#include <FL/Fl.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Browser.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Multiline_Output.H>
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Window.H>
#include <FL/fl_ask.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "../../galois_math.h"
#include "../../complex_math.h"

using namespace std;

static inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

vector<double> companion_matrix(const vector<int> &coeffs) {
  int n = (int)coeffs.size() - 1;
  if (n < 1) return {};
  vector<double> M(n * n, 0.0);
  for (int i = 1; i < n; ++i) M[i * n + (i - 1)] = 1.0;
  for (int i = 0; i < n; ++i) {
    double val = (i < (int)coeffs.size()) ? (double)coeffs[i] : 0.0;
    M[i * n + (n - 1)] = -val / coeffs.back();
  }
  return M;
}

string companion_matrix_to_csv(const vector<double> &M) {
  if (M.empty()) return "";
  int n = (int)sqrt((double)M.size());
  ostringstream ss;
  ss << fixed << setprecision(6);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      ss << setw(10) << M[r * n + c] << (c + 1 < n ? "," : "");
    }
    ss << "\n";
  }
  return ss.str();
}

// ------------------ Image Processing ------------------ //

vector<double> gaussian_kernel(double sigma, int &ksize_out) {
  if (sigma <= 0.0) { ksize_out = 1; return {1.0}; }
  int radius = (int)ceil(3.0 * sigma);
  int ksize = 2 * radius + 1;
  vector<double> k(ksize);
  double s2 = 2.0 * sigma * sigma;
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    double v = exp(-(i * i) / s2);
    k[i + radius] = v;
    sum += v;
  }
  for (auto &x : k) x /= sum;
  ksize_out = ksize;
  return k;
}

void separable_blur(vector<double> &grid, int res, double sigma) {
  if (sigma <= 0.0) return;
  int ksize;
  vector<double> k = gaussian_kernel(sigma, ksize);
  int radius = (ksize - 1) / 2;
  vector<double> tmp(res * res, 0.0);
  for (int y = 0; y < res; ++y) {
    for (int x = 0; x < res; ++x) {
      double s = 0;
      for (int j = -radius; j <= radius; ++j) {
        int xx = std::max(0, std::min(res - 1, x + j));
        s += k[j + radius] * grid[y * res + xx];
      }
      tmp[y * res + x] = s;
    }
  }
  for (int x = 0; x < res; ++x) {
    for (int y = 0; y < res; ++y) {
      double s = 0;
      for (int j = -radius; j <= radius; ++j) {
        int yy = std::max(0, std::min(res - 1, y + j));
        s += k[j + radius] * tmp[yy * res + x];
      }
      grid[y * res + x] = s;
    }
  }
}

void magma_colormap(double t, unsigned char &r, unsigned char &g, unsigned char &b) {
  t = clamp01(t);
  r = (unsigned char)(pow(t, 0.4) * 255.0);
  g = (unsigned char)(pow(t, 1.9) * 180.0);
  b = (unsigned char)(pow(1.0 - t, 1.2) * 220.0);
  if (t > 0.98) { r = 255; g = 255; b = 230; }
}

// ------------------ Visualization Widget ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
  bool mode_algebraic = true;
  bool show_edges = true;
  int max_degree = 5;
  int max_coeff = 5;
  double sigma = 1.0;
  int p_prime = 3;
  int n_ext = 2;

  const int grid_res = 512;
  vector<double> heat;
  vector<unsigned char> image;
  std::mt19937 rng;
  bool dirty_compute = true;
  GLuint tex_id = 0;

  vector<int> adjoin_coeffs;
  vector<cd> adjoin_roots;
  int chosen_root_index = -1;
  cd chosen_alpha;
  vector<double> companion_M;

  UnifiedGL(int X, int Y, int W, int H, const char *L = 0)
      : Fl_Gl_Window(X, Y, W, H, L) {
    heat.resize(grid_res * grid_res);
    image.resize(grid_res * grid_res * 3);
    rng.seed(std::random_device{}());
    end();
  }

  bool map_to_grid(const cd &z, int &ix, int &iy) {
    double re = z.real(), im = z.imag(), limit = 2.0;
    if (re < -limit || re > limit || im < -limit || im > limit) return false;
    ix = (int)((re + limit) / (2.0 * limit) * (grid_res - 1));
    iy = (int)((im + limit) / (2.0 * limit) * (grid_res - 1));
    return true;
  }

  void compute_algebraic() {
    std::fill(heat.begin(), heat.end(), 0.0);
    std::uniform_int_distribution<int> coeff_dist(-max_coeff, max_coeff);
    int samples = 50000;
    for (int i = 0; i < samples; ++i) {
      std::uniform_int_distribution<int> deg_dist(1, max_degree);
      int d = deg_dist(rng);
      vector<cd> coeffs(d + 1);
      for (int k = 0; k <= d; ++k) {
        int c = coeff_dist(rng);
        if (k == d && c == 0) c = 1;
        coeffs[k] = cd((double)c, 0.0);
      }
      auto roots = dk_solve_roots(coeffs);
      for (const auto &r : roots) {
        int ix, iy;
        if (map_to_grid(r, ix, iy)) heat[iy * grid_res + ix] += 1.0;
      }
    }
    separable_blur(heat, grid_res, sigma);
    double maxv = 0.0;
    for (double v : heat) if (v > maxv) maxv = v;
    if (maxv < 1e-9) maxv = 1.0;
    for (int i = 0; i < grid_res * grid_res; ++i) {
      double t = log(1.0 + heat[i] * 10.0) / log(1.0 + maxv * 10.0);
      magma_colormap(t, image[i * 3 + 0], image[i * 3 + 1], image[i * 3 + 2]);
    }
  }

  void draw() override {
    if (!valid()) {
      valid(1);
      glDisable(GL_DEPTH_TEST);
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      if (tex_id == 0) glGenTextures(1, &tex_id);
    }
    if (dirty_compute && mode_algebraic) {
      compute_algebraic();
      glBindTexture(GL_TEXTURE_2D, tex_id);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, grid_res, grid_res, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data());
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
      glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
      dirty_compute = false;
    }
    glClearColor(0.06f, 0.06f, 0.06f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (mode_algebraic) render_algebraic_view();
    else render_finite_field_math();
    draw_overlays();
  }

  void render_algebraic_view() {
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex_id);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2f(-1, -1);
    glTexCoord2f(1, 0); glVertex2f(1, -1);
    glTexCoord2f(1, 1); glVertex2f(1, 1);
    glTexCoord2f(0, 1); glVertex2f(-1, 1);
    glEnd();
    glDisable(GL_TEXTURE_2D);
  }

  void render_finite_field_math() {
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    int p = max(2, p_prime), n = max(1, n_ext);
    int N = (int)pow(p, n); if (N > 10000) N = 10000;
    vector<cd> elems;
    for (int i = 0; i < N; ++i) {
      int t = i; cd z(0, 0);
      for (int j = 0; j < n; ++j) {
        int coeff = t % p; t /= p;
        z += polar(1.0, 2.0 * M_PI * j / n) * (double)coeff;
      }
      elems.push_back(z);
    }
    double maxr = 0; for (auto &z : elems) maxr = max(maxr, abs(z));
    if (maxr == 0) maxr = 1.0;
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for (const auto &z : elems) {
      float hue = (float)(arg(z) / (2.0 * M_PI) + 0.5), mag = (float)(abs(z) / maxr);
      glColor4f(0.2f + 0.8f * mag, 0.6f * hue, 1.0f - 0.5f * mag, 1.0f);
      glVertex2f((float)z.real(), (float)z.imag());
    }
    glEnd();
  }

  void draw_overlays() {
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1, 1, 1, 0.2f);
    glVertex2f(-1, 0); glVertex2f(1, 0); glVertex2f(0, -1); glVertex2f(0, 1);
    glEnd();
    glBegin(GL_LINE_LOOP); for (int i = 0; i < 80; ++i) {
      double th = 2.0 * M_PI * i / 80.0; glVertex2f(0.5f * cos(th), 0.5f * sin(th));
    } glEnd();
    if (!adjoin_coeffs.empty() && chosen_root_index >= 0) {
      glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-2.0, 2.0, -2.0, 2.0, -1, 1);
      int deg = (int)adjoin_coeffs.size() - 1, bound = 2;
      glPointSize(5.0f); glBegin(GL_POINTS);
      int total = (int)pow((2 * bound + 1), deg);
      for (int idx = 0; idx < total && idx < 5000; ++idx) {
        int t = idx; vector<cd> pc(deg);
        for (int k = 0; k < deg; ++k) { pc[k] = cd((t % (2 * bound + 1)) - bound, 0); t /= (2 * bound + 1); }
        cd z = complex_eval_poly(pc, chosen_alpha);
        glColor4f(1, 1, 1, 0.7f); glVertex2f((float)z.real(), (float)z.imag());
      }
      glEnd();
    }
  }

  void trigger_regen() { dirty_compute = true; redraw(); }
};

struct AdjoinContext { UnifiedGL *gl; Fl_Input *inp; Fl_Browser *list; Fl_Multiline_Output *mat; };

int main(int argc, char **argv) {
  Fl_Window *win = new Fl_Window(1300, 860, "Unified Field Explorer");
  UnifiedGL *gl = new UnifiedGL(10, 10, 900, 840);
  Fl_Group *panel = new Fl_Group(920, 10, 370, 840);
  Fl_Choice *mode_choice = new Fl_Choice(1020, 50, 200, 25, "Mode:");
  mode_choice->add("Algebraic"); mode_choice->add("Finite Field"); mode_choice->value(0);
  mode_choice->callback([](Fl_Widget *w, void *v) { UnifiedGL *g = (UnifiedGL *)v; g->mode_algebraic = (((Fl_Choice *)w)->value() == 0); g->trigger_regen(); }, gl);
  Fl_Value_Slider *s1 = new Fl_Value_Slider(1020, 90, 260, 20, "Deg/P");
  s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 15); s1->value(5);
  s1->callback([](Fl_Widget *w, void *v) { UnifiedGL *g = (UnifiedGL *)v; g->max_degree = g->p_prime = (int)((Fl_Value_Slider *)w)->value(); g->trigger_regen(); }, gl);
  Fl_Value_Slider *s2 = new Fl_Value_Slider(1020, 120, 260, 20, "Coeff/N");
  s2->type(FL_HOR_NICE_SLIDER); s2->bounds(1, 20); s2->value(5);
  s2->callback([](Fl_Widget *w, void *v) { UnifiedGL *g = (UnifiedGL *)v; g->max_coeff = g->n_ext = (int)((Fl_Value_Slider *)w)->value(); g->trigger_regen(); }, gl);
  Fl_Input *inp_poly = new Fl_Input(1000, 290, 280, 25, "Poly:"); inp_poly->value("1,0,1");
  Fl_Browser *br_roots = new Fl_Browser(930, 360, 350, 100, "Roots:"); br_roots->type(FL_HOLD_BROWSER);
  Fl_Multiline_Output *out_matrix = new Fl_Multiline_Output(930, 500, 350, 180, "Companion Matrix"); out_matrix->textfont(FL_COURIER);
  AdjoinContext *ctx = new AdjoinContext{gl, inp_poly, br_roots, out_matrix};
  Fl_Button *btn_adjoin = new Fl_Button(930, 325, 350, 25, "Adjoin Polynomial");
  btn_adjoin->callback([](Fl_Widget *, void *v) {
    auto c = (AdjoinContext *)v; string s = c->inp->value(), tmp; vector<int> coeffs;
    for (char ch : s) { if (ch == ',' || ch == ' ') { if (!tmp.empty()) { coeffs.push_back(stoi(tmp)); tmp.clear(); } } else tmp.push_back(ch); }
    if (!tmp.empty()) coeffs.push_back(stoi(tmp));
    if (coeffs.size() < 2) return;
    c->gl->adjoin_coeffs = coeffs; vector<cd> c_cd; for (int k : coeffs) c_cd.push_back(cd((double)k, 0));
    c->gl->adjoin_roots = dk_solve_roots(c_cd); c->list->clear();
    for (size_t i = 0; i < c->gl->adjoin_roots.size(); ++i) { ostringstream ss; ss << "r" << i << ": " << fixed << setprecision(3) << c->gl->adjoin_roots[i]; c->list->add(ss.str().c_str()); }
    if (!c->gl->adjoin_roots.empty()) { c->list->select(1); c->gl->chosen_root_index = 0; c->gl->chosen_alpha = c->gl->adjoin_roots[0]; }
    c->gl->companion_M = companion_matrix(coeffs); c->mat->value(companion_matrix_to_csv(c->gl->companion_M).c_str()); c->gl->redraw();
  }, ctx);
  br_roots->callback([](Fl_Widget *w, void *v) { AdjoinContext *c = (AdjoinContext *)v; int sel = ((Fl_Browser *)w)->value(); if (sel > 0) { c->gl->chosen_root_index = sel - 1; c->gl->chosen_alpha = c->gl->adjoin_roots[sel - 1]; c->gl->redraw(); } }, ctx);
  panel->end(); win->end(); win->resizable(gl); win->show(argc, argv); gl->trigger_regen(); return Fl::run();
}
