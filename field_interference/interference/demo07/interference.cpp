// Unified Field Explorer (High-Performance Edition)
//
// Features:
//   - Hardware-Accelerated Root Density Heatmaps
//   - Lattice Resonance Analysis (Rational, Gaussian, Eisenstein)
//   - Advanced Mappings (Log-Polar, Mobius, Euler Space)
//   - Finite Field Structure Exploration

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
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Window.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include "../../galois_math.h"
#include "../../complex_math.h"

using namespace std;

vector<cd> get_anchors(cd z, int n = 8) {
  vector<cd> res;
  double x = z.real();
  double a = floor(x);
  vector<double> convs; convs.push_back(a);
  double rem = x - a;
  for (int i = 0; i < n; ++i) {
    if (abs(rem) < 1e-11) break;
    double inv = 1.0 / rem;
    a = floor(inv); convs.push_back(a);
    rem = inv - a;
  }
  for (int i = 1; i <= (int)convs.size(); ++i) {
    double p = 1, q = 0;
    for (int j = i - 1; j >= 0; --j) { double np = convs[j] * p + q; q = p; p = np; }
    res.push_back(cd(p / q, 0));
  }
  int re = (int)round(z.real()), im = (int)round(z.imag());
  for(int i=-1; i<=1; ++i) for(int j=-1; j<=1; ++j) {
    res.push_back(cd(re+i, im+j));
    res.push_back(cd(re+i + 0.5*(im+j), sqrt(3.0)/2.0*(im+j)));
  }
  return res;
}

class UnifiedGL : public Fl_Gl_Window {
public:
  bool mode_algebraic = true;
  bool mode_resonance = false;
  bool show_edges = true;
  int mapping_type = 0;
  double v_x = 0, v_y = 0, v_zoom = 1.0;
  int last_mx = 0, last_my = 0;
  int max_deg = 5, max_c = 5, p_prime = 3, n_ext = 2;
  int quality_samples = 40000;
  double l_scale = 1.0;
  const int res = 512;
  vector<unsigned char> tex_data;
  GLuint tex_id = 0;
  bool dirty_compute = true;
  vector<int> adj_coeffs;
  vector<cd> adj_roots;
  cd chosen_alpha = 0;

  UnifiedGL(int X, int Y, int W, int H) : Fl_Gl_Window(X, Y, W, H) {
    tex_data.resize(res * res * 3, 0);
  }

  cd apply_mapping(cd z) {
    if (mapping_type == 1) {
      double r = abs(z); double t = arg(z);
      return cd(log(max(1e-15, r)), t);
    }
    if (mapping_type == 2) {
      return (z - cd(0, 1)) / (z + cd(0, 1) + 1e-15);
    }
    if (mapping_type == 3) {
      return exp(cd(0, 1) * M_PI * z / l_scale);
    }
    return z;
  }

  void magma_color(double t, unsigned char *rgb) {
    t = max(0.0, min(1.0, t));
    rgb[0] = (unsigned char)(pow(t, 0.4) * 255);
    rgb[1] = (unsigned char)(pow(t, 1.8) * 180);
    rgb[2] = (unsigned char)(pow(1.0 - t, 1.2) * 220);
  }

  void compute_heatmap() {
    vector<double> heat(res * res, 0.0);
    mt19937 rng(1337); uniform_int_distribution<int> c_dist(-max_c, max_c);
    int num_samples = mode_resonance ? quality_samples / 4 : quality_samples;
    for (int i = 0; i < num_samples; ++i) {
      int d = (i % max_deg) + 1; vector<cd> c(d + 1);
      for (int k = 0; k <= d; ++k) c[k] = (double)c_dist(rng);
      if (abs(c.back()) < 0.1) c.back() = 1.0;
      for (auto &r : dk_solve_roots(c)) {
        double val = 1.0;
        if (mode_resonance) {
          auto anchors = get_anchors(r);
          double min_d = 1e9;
          for (auto &an : anchors) min_d = min(min_d, abs(r - an));
          val = -log10(min_d + 1e-15);
        }
        cd mr = apply_mapping(r);
        int ix = (int)((mr.real() + 2.0) / 4.0 * res), iy = (int)((mr.imag() + 2.0) / 4.0 * res);
        if (ix >= 0 && ix < res && iy >= 0 && iy < res) {
          if (mode_resonance) heat[iy * res + ix] = max(heat[iy * res + ix], val);
          else heat[iy * res + ix] += 1.0;
        }
      }
    }
    double mv = 0; for (double v : heat) mv = max(mv, v);
    for (int i = 0; i < res * res; ++i) {
      double t = (mv > 0) ? (mode_resonance ? (heat[i]/mv) : log(1.0 + heat[i]*5)/log(1.0 + mv*5)) : 0;
      magma_color(t, &tex_data[i * 3]);
    }
    if (tex_id == 0) glGenTextures(1, &tex_id);
    glBindTexture(GL_TEXTURE_2D, tex_id);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, res, res, 0, GL_RGB, GL_UNSIGNED_BYTE, tex_data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  }

  int handle(int e) override {
    if (e == FL_PUSH) { last_mx = Fl::event_x(); last_my = Fl::event_y(); return 1; }
    if (e == FL_DRAG) {
      double dx = Fl::event_x() - last_mx, dy = Fl::event_y() - last_my;
      if (Fl::event_button() == 1) { v_x -= (dx / w()) * (4.0 / v_zoom); v_y += (dy / h()) * (4.0 / v_zoom); }
      else { v_zoom *= (1.0 - dy / 100.0); }
      last_mx = Fl::event_x(); last_my = Fl::event_y(); redraw(); return 1;
    }
    if (e == FL_MOUSEWHEEL) { v_zoom *= (1.0 - Fl::event_dy() * 0.1); redraw(); return 1; }
    return Fl_Gl_Window::handle(e);
  }

  void draw() override {
    if (!valid()) { valid(1); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); }
    if (dirty_compute && mode_algebraic) { compute_heatmap(); dirty_compute = false; }
    glClearColor(0.02f, 0.02f, 0.05f, 1.0f); glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION); glLoadIdentity();
    double aspect = (double)w() / h();
    double hw = 2.0 * aspect / v_zoom, hh = 2.0 / v_zoom;
    glOrtho(v_x - hw, v_x + hw, v_y - hh, v_y + hh, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    if (mode_algebraic) {
      glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex_id); glColor4f(1, 1, 1, 1);
      glBegin(GL_QUADS); glTexCoord2f(0, 0); glVertex2f(-2, -2); glTexCoord2f(1, 0); glVertex2f(2, -2); glTexCoord2f(1, 1); glVertex2f(2, 2); glTexCoord2f(0, 1); glVertex2f(-2, 2); glEnd(); glDisable(GL_TEXTURE_2D);
    } else render_finite();
    draw_extension_lattice(); draw_overlays();
  }

  void render_finite() {
    int p = max(2, p_prime), n = max(1, n_ext);
    int N = (int)pow(p, n); if (N > 10000) N = 10000;
    vector<int> irred = gf_find_irreducible(p, n); GFElement alpha = gf_find_primitive(p, n, irred);
    auto map_to_2d = [&](const GFElement &e) {
      cd z(0, 0); for (int j = 0; j < n; ++j) z += polar(l_scale, 2 * M_PI * j / n) * (double)e[j];
      return apply_mapping(z);
    };
    if (show_edges && N < 5000) {
      GFElement curr(n, 0); curr[0] = 1; glLineWidth(1.0f); glBegin(GL_LINE_STRIP);
      for (int k = 0; k < min(N, 1200); ++k) {
        cd z = map_to_2d(curr); glColor4f(0.3f, 0.7f, 1.0f, 0.5f * (1.0f - (float)k / 1200)); glVertex2f(z.real(), z.imag());
        curr = gf_multiply(curr, alpha, irred, p); if (gf_is_one(curr)) break;
      }
      glEnd();
    }
    glPointSize(4.0f); glBegin(GL_POINTS);
    for (int i = 0; i < N; ++i) {
      int t = i; GFElement e(n); for (int j = 0; j < n; ++j) { e[j] = t % p; t /= p; }
      cd z = map_to_2d(e); glColor3f(0.4f, 0.6f, 1.0f); glVertex2f(z.real(), z.imag());
    }
    glEnd();
  }

  void draw_extension_lattice() {
    if (adj_coeffs.size() < 2) return;
    int deg = adj_coeffs.size() - 1, b = 3; glPointSize(6.0f); glBegin(GL_POINTS);
    int total = (int)pow(2 * b + 1, min(deg, 3));
    for (int i = 0; i < total; ++i) {
      int t = i; vector<cd> pc(deg);
      for (int k = 0; k < deg; ++k) { pc[k] = cd((t % (2 * b + 1)) - b, 0); t /= (2 * b + 1); }
      cd z = apply_mapping(complex_eval_poly(pc, chosen_alpha)); glColor4f(1, 0.8f, 0.1f, 0.9f); glVertex2f(z.real(), z.imag());
    }
    glEnd();
  }

  void draw_overlays() {
    glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1, 1, 1, 0.2f);
    glVertex2f(-10, 0); glVertex2f(10, 0); glVertex2f(0, -10); glVertex2f(0, 10);
    glEnd();
    glBegin(GL_LINE_LOOP); for (int i = 0; i < 100; ++i) {
      cd z = apply_mapping(polar(1.0, 2 * M_PI * i / 100.0)); glVertex2f(z.real(), z.imag());
    } glEnd();
  }
};

struct UI { UnifiedGL *gl; Fl_Input *poly; Fl_Browser *roots; };

int main(int argc, char **argv) {
  Fl_Window *win = new Fl_Window(1200, 800, "Unified Field Explorer - Advanced");
  UnifiedGL *gl = new UnifiedGL(10, 10, 850, 780);
  Fl_Group *ctrl = new Fl_Group(870, 10, 320, 780);
  Fl_Choice *mode = new Fl_Choice(970, 40, 200, 25, "Mode");
  mode->add("Algebraic"); mode->add("Finite Field"); mode->value(0);
  Fl_Choice *map = new Fl_Choice(970, 75, 200, 25, "Mapping");
  map->add("Standard"); map->add("Log-Polar"); map->add("Mobius"); map->add("Euler Space"); map->value(0);
  Fl_Check_Button *res_btn = new Fl_Check_Button(970, 110, 200, 25, "Lattice Resonance");
  Fl_Value_Slider *sq = new Fl_Value_Slider(970, 145, 200, 20, "Samples");
  sq->type(FL_HOR_NICE_SLIDER); sq->bounds(1000, 200000); sq->step(1000); sq->value(40000);
  Fl_Value_Slider *s1 = new Fl_Value_Slider(970, 175, 200, 20, "Deg/P");
  Fl_Box *prime_warn = new Fl_Box(970, 195, 200, 15, "");
  prime_warn->labelcolor(FL_RED);
  prime_warn->labelsize(12);

  s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 31); s1->value(5);
  Fl_Value_Slider *s2 = new Fl_Value_Slider(970, 205, 200, 20, "Cof/N");
  s2->type(FL_HOR_NICE_SLIDER); s2->bounds(1, 15); s2->value(5);
  Fl_Input *poly = new Fl_Input(970, 265, 200, 25, "Adjoin Poly"); poly->value("1,0,1");
  Fl_Browser *roots = new Fl_Browser(870, 335, 310, 150, "Roots (Alpha)"); roots->type(FL_HOLD_BROWSER);
  UI *ui = new UI{gl, poly, roots};
  auto upd = [](Fl_Widget *, void *v) {
    UI *u = (UI *)v; string s = u->poly->value(), t; vector<int> co; stringstream ss(s);
    while (getline(ss, t, ',')) { try { co.push_back(stoi(t)); } catch (...) {} }
    if (co.size() < 2) return;
    u->gl->adj_coeffs = co; vector<cd> cdc; for (int k : co) cdc.push_back((double)k);
    u->gl->adj_roots = dk_solve_roots(cdc); u->roots->clear();
    for (auto &r : u->gl->adj_roots) { ostringstream os; os << r; u->roots->add(os.str().c_str()); }
    if (!u->gl->adj_roots.empty()) { u->roots->select(1); u->gl->chosen_alpha = u->gl->adj_roots[0]; }
    u->gl->redraw();
  };
  mode->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->mode_algebraic = (((Fl_Choice *)w)->value() == 0); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw(); }, gl);
  map->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->mapping_type = ((Fl_Choice *)w)->value(); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw(); }, gl);
  res_btn->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->mode_resonance = ((Fl_Check_Button *)w)->value(); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw(); }, gl);
  sq->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->quality_samples = (int)((Fl_Value_Slider *)w)->value(); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw(); }, gl);
  s1->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->max_deg = ((UnifiedGL *)v)->p_prime = (int)((Fl_Value_Slider *)w)->value(); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw();
    Fl_Box *b = (Fl_Box *)w->parent()->child(w->parent()->find(w) + 1);
    UnifiedGL *gl_ptr = (UnifiedGL *)v;
    if (!gl_ptr->mode_algebraic && !is_prime(gl_ptr->p_prime)) b->label("Warning: p not prime"); else b->label("");}, gl);
  s2->callback([](Fl_Widget *w, void *v) { ((UnifiedGL *)v)->max_c = ((UnifiedGL *)v)->n_ext = (int)((Fl_Value_Slider *)w)->value(); ((UnifiedGL *)v)->dirty_compute = true; ((UnifiedGL *)v)->redraw(); }, gl);
  Fl_Button *apply_btn = new Fl_Button(870, 295, 310, 30, "Apply Adjunction"); apply_btn->callback(upd, ui);
  roots->callback([](Fl_Widget *w, void *v) { UnifiedGL *gl = (UnifiedGL *)v; int s = ((Fl_Browser *)w)->value(); if (s > 0) { gl->chosen_alpha = gl->adj_roots[s - 1]; gl->redraw(); } }, gl);
  ctrl->end(); win->end(); win->resizable(gl); win->show(); return Fl::run();
}
