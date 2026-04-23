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
#include <FL/gl.h>
#include <GL/glu.h>

#include <algorithm>
#include <atomic>
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
#include "../../interference_core.h"
#include "../../mappings.h"

using namespace std;

static inline double clamp01(double x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }

vector<double> companion_matrix(const vector<int> &coeffs) {
  int n = (int)coeffs.size() - 1;
  if (n < 1) return {};
  vector<double> M(n * n, 0.0);
  for (int i = 1; i < n; ++i) M[i * n + (i - 1)] = 1.0;
  for (int i = 0; i < n; ++i) {
    double val = (i < (int)coeffs.size()) ? (double)coeffs[i] : 0.0;
    M[i * n + (n - 1)] = -val;
  }
  return M;
}

string companion_matrix_to_csv(const vector<double> &M) {
  if (M.empty()) return "";
  int n = (int)sqrt((double)M.size());
  ostringstream ss; ss << fixed << setprecision(12);
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) { ss << M[r * n + c] << (c + 1 < n ? "," : ""); }
    ss << "\n";
  }
  return ss.str();
}

vector<double> gaussian_kernel(double sigma, int &ksize_out) {
  if (sigma <= 0.0) { ksize_out = 1; return {1.0}; }
  int radius = (int)ceil(3.0 * sigma);
  int ksize = 2 * radius + 1;
  vector<double> k(ksize);
  double s2 = 2.0 * sigma * sigma, sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    double v = exp(-(i * i) / s2); k[i + radius] = v; sum += v;
  }
  for (auto &x : k) x /= sum;
  ksize_out = ksize; return k;
}

void separable_blur(vector<double> &grid, int res, double sigma) {
  if (sigma <= 0.0) return;
  int ksize; vector<double> k = gaussian_kernel(sigma, ksize);
  int radius = (ksize - 1) / 2;
  vector<double> tmp(res * res, 0.0);
  for (int y = 0; y < res; ++y) {
    for (int x = 0; x < res; ++x) {
      double s = 0;
      for (int j = -radius; j <= radius; ++j) {
        int xx = clamp(x + j, 0, res - 1); s += k[j + radius] * grid[y * res + xx];
      }
      tmp[y * res + x] = s;
    }
  }
  for (int x = 0; x < res; ++x) {
    for (int y = 0; y < res; ++y) {
      double s = 0;
      for (int j = -radius; j <= radius; ++j) {
        int yy = clamp(y + j, 0, res - 1); s += k[j + radius] * tmp[yy * res + x];
      }
      grid[y * res + x] = s;
    }
  }
}

void magma_colormap(double t, unsigned char &r, unsigned char &g, unsigned char &b) {
  t = clamp01(t);
  double rr = pow(t, 0.4) * 255.0, gg = pow(t, 1.9) * 180.0, bb = pow(1.0 - t, 1.2) * 220.0;
  if (t > 0.95) { rr = 255; gg = 255; bb = 220; }
  r = (unsigned char)clamp01(rr / 255.0) * 255;
  g = (unsigned char)clamp01(gg / 255.0) * 255;
  b = (unsigned char)clamp01(bb / 255.0) * 255;
}

class UnifiedGL : public Fl_Gl_Window {
public:
  bool mode_algebraic = true;
  int max_degree = 3, max_coeff = 5;
  double sigma = 1.0;
  int prime_p = 3, ext_n = 2;
  const int grid_res = 512;
  vector<double> heat;
  vector<unsigned char> image;
  std::mt19937 rng;
  bool dirty_compute = true;
  bool filter_real = false, filter_unit = false, filter_quadratic = false, filter_high_degree = false;
  vector<int> adjoin_coeffs;
  vector<cd> adjoin_roots;
  int chosen_root_index = -1;
  cd chosen_alpha;
  vector<double> companion_M;

  UnifiedGL(int X, int Y, int W, int H, const char *L = 0) : Fl_Gl_Window(X, Y, W, H, L) {
    heat.resize(grid_res * grid_res); image.resize(grid_res * grid_res * 3);
    rng.seed(std::random_device{}()); end();
  }

  bool map_to_grid(const cd &z, int &ix, int &iy) {
    double re = z.real(), im = z.imag(), limit = 2.0;
    if (re < -limit || re > limit || im < -limit || im > limit) return false;
    double nx = (re + limit) / (2.0 * limit), ny = (im + limit) / (2.0 * limit);
    ix = (int)(nx * (grid_res - 1)); iy = (int)(ny * (grid_res - 1)); return true;
  }

  void compute_algebraic() {
    int samples = (max_degree > 6) ? 20000 : 50000;
    interference::compute_threaded(samples, grid_res, max_degree, max_coeff, heat, [](cd r){ return r; });
    separable_blur(heat, grid_res, sigma);
    double maxv = 0.0;
    for (double v : heat) if (v > maxv) maxv = v;
    if (maxv < 1e-9) maxv = 1.0;
    for (int i = 0; i < grid_res * grid_res; ++i) {
      double t = log(1.0 + heat[i] * 10.0) / log(1.0 + maxv * 10.0);
      magma_colormap(t, image[i * 3], image[i * 3 + 1], image[i * 3 + 2]);
    }
  }

  void draw() override {
    if (!valid()) { valid(1); glDisable(GL_DEPTH_TEST); glPixelStorei(GL_UNPACK_ALIGNMENT, 1); }
    if (dirty_compute && mode_algebraic) { compute_algebraic(); dirty_compute = false; }
    glClearColor(0.06f, 0.06f, 0.06f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
    if (mode_algebraic) {
      glMatrixMode(GL_PROJECTION); glLoadIdentity(); glMatrixMode(GL_MODELVIEW); glLoadIdentity();
      glRasterPos2f(-1.0f, -1.0f); glPixelZoom((float)w() / grid_res, (float)h() / grid_res);
      glDrawPixels(grid_res, grid_res, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    } else render_finite_view();
    draw_overlays();
  }

  void render_finite_view() {
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    int p = max(2, prime_p), n = max(1, ext_n); int N = (int)pow(p, n); if (N > 100000) N = 100000;
    glPointSize(5.0f); glBegin(GL_POINTS);
    for (int i = 0; i < N; ++i) {
      int t = i; cd z(0, 0);
      for (int j = 0; j < n; ++j) { z += polar(1.0, 2.0 * M_PI * j / n) * (double)(t % p); t /= p; }
      float hue = (float)(arg(z) / (2.0 * M_PI) + 0.5), mag = (float)(abs(z) / (p*n));
      glColor3f(0.2f + 0.8f * mag, 0.6f * hue, 1.0f - 0.5f * mag); glVertex2f((float)z.real(), (float)z.imag());
    }
    glEnd();
  }

  void draw_overlays() {
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1, 1, -1, 1, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
    glVertex2f(-1.0f, 0.0f); glVertex2f(1.0f, 0.0f); glVertex2f(0.0f, -1.0f); glVertex2f(0.0f, 1.0f); glEnd();
    glBegin(GL_LINE_LOOP); glColor4f(1.0f, 1.0f, 1.0f, 0.2f);
    for (int i = 0; i < 64; ++i) { double theta = 2.0 * M_PI * i / 64.0; glVertex2f((float)(0.5 * cos(theta)), (float)(0.5 * sin(theta))); }
    glEnd();
    if (!adjoin_coeffs.empty() && chosen_root_index >= 0) {
      glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-2.0, 2.0, -2.0, 2.0, -1, 1);
      int deg = (int)adjoin_coeffs.size() - 1; int total = (int)pow(3, deg); int cap = min(total, 5000);
      glPointSize(4.0f); glBegin(GL_POINTS); glColor3f(0.2f, 1.0f, 0.2f);
      for (int idx = 0; idx < cap; ++idx) {
        int t = idx; vector<int> pi_int(deg); for (int k = 0; k < deg; ++k) { pi_int[k] = (double)((t % 3) - 1); t /= 3; }
        vector<cd> pc; for(int x:pi_int) pc.push_back(cd(x,0));
        cd z = complex_eval_poly(pc, chosen_alpha); glVertex2f((float)z.real(), (float)z.imag());
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
  mode_choice->callback([](Fl_Widget *w, void *v) {
    UnifiedGL *gl = (UnifiedGL *)v; gl->mode_algebraic = (((Fl_Choice *)w)->value() == 0); gl->trigger_regen();
  }, gl);
  Fl_Value_Slider *s1 = new Fl_Value_Slider(1020, 90, 260, 20, "Deg / P");
  s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 15); s1->step(1); s1->value(5);
  s1->callback([](Fl_Widget *w, void *v) {
    UnifiedGL *gl = (UnifiedGL *)v; int val = (int)((Fl_Value_Slider *)w)->value();
    gl->max_degree = val; gl->prime_p = max(2, val); gl->trigger_regen();
  }, gl);
  Fl_Value_Slider *s2 = new Fl_Value_Slider(1020, 120, 260, 20, "Coeff / N");
  s2->type(FL_HOR_NICE_SLIDER); s2->bounds(1, 20); s2->step(1); s2->value(5);
  s2->callback([](Fl_Widget *w, void *v) {
    UnifiedGL *gl = (UnifiedGL *)v; int val = (int)((Fl_Value_Slider *)w)->value();
    gl->max_coeff = val; gl->ext_n = max(1, val); gl->trigger_regen();
  }, gl);
  Fl_Input *inp_poly = new Fl_Input(1000, 340, 280, 25, "Poly:"); inp_poly->value("1,0,1");
  Fl_Browser *br_roots = new Fl_Browser(930, 420, 350, 120, "Roots"); br_roots->type(FL_HOLD_BROWSER);
  Fl_Multiline_Output *out_mat = new Fl_Multiline_Output(930, 570, 350, 200, "Matrix"); out_mat->textfont(FL_COURIER);
  AdjoinContext *ctx = new AdjoinContext{gl, inp_poly, br_roots, out_mat};
  Fl_Button *btn = new Fl_Button(930, 380, 350, 25, "Adjoin");
  btn->callback([](Fl_Widget *, void *v) {
    AdjoinContext *c = (AdjoinContext *)v; string s = c->inp->value(), tmp; vector<int> co;
    for (char ch : s) { if (ch == ',' || ch == ' ') { if (!tmp.empty()) { co.push_back(stoi(tmp)); tmp.clear(); } } else tmp.push_back(ch); }
    if (!tmp.empty()) co.push_back(stoi(tmp)); if (co.size() < 2) return;
    c->gl->adjoin_coeffs = co; vector<cd> c_cd; for (int k : co) c_cd.push_back(cd((double)k, 0));
    c->gl->adjoin_roots = dk_solve_roots(c_cd); c->list->clear();
    for (size_t i = 0; i < c->gl->adjoin_roots.size(); ++i) {
      ostringstream ss; ss << i << ": " << c->gl->adjoin_roots[i]; c->list->add(ss.str().c_str());
    }
    if (!c->gl->adjoin_roots.empty()) { c->list->select(1); c->gl->chosen_root_index = 0; c->gl->chosen_alpha = c->gl->adjoin_roots[0]; }
    c->gl->companion_M = companion_matrix(co); c->mat->value(companion_matrix_to_csv(c->gl->companion_M).c_str()); c->gl->redraw();
  }, ctx);
  panel->end(); win->end(); win->resizable(gl); win->show(argc, argv); return Fl::run();
}
