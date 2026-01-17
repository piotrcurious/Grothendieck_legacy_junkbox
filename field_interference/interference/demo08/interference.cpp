// Unified Field Explorer (Synchronized Viewport Edition)
//
// Description:
//   A high-performance C++ tool using FLTK and OpenGL.
//   Synchronized coordinate systems for:
//   1. Algebraic Numbers (Heatmaps)
//   2. Finite Fields (Lattices/Cycles)
//   3. Field Extensions (Lattices)
//
// Features: 
//   - Panning (Left Click)
//   - Zooming (Scroll / Right Click)
//   - Synchronized Texture-mapped Heatmaps

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Browser.H>
#include <FL/Fl_Multiline_Output.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <random>
#include <sstream>
#include <algorithm>
#include <iomanip>

using namespace std;
using cd = complex<double>;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------ Math & Processing ------------------ //

vector<cd> durand_kerner(const vector<cd>& input_coeffs) {
    vector<cd> coeffs = input_coeffs;
    while (coeffs.size() > 1 && abs(coeffs.back()) < 1e-9) coeffs.pop_back();
    int n = (int)coeffs.size() - 1;
    if (n <= 0) return {};
    vector<cd> roots(n);
    cd leading = coeffs[n];
    for (auto& c : coeffs) c /= leading;
    double radius = 1.0;
    for (int i = 0; i < n; ++i) radius = max(radius, 1.0 + abs(coeffs[i]));
    for (int i = 0; i < n; ++i) roots[i] = polar(radius, 2.0 * M_PI * i / n + 0.1);
    for (int iter = 0; iter < 100; ++iter) {
        double max_change = 0.0;
        for (int i = 0; i < n; ++i) {
            cd p_val = coeffs[n];
            for (int k = n - 1; k >= 0; --k) p_val = p_val * roots[i] + coeffs[k];
            cd prod = 1.0;
            for (int j = 0; j < n; ++j) if (i != j) prod *= (roots[i] - roots[j]);
            cd delta = p_val / (prod == 0.0 ? 1e-18 : prod);
            roots[i] -= delta;
            max_change = max(max_change, abs(delta));
        }
        if (max_change < 1e-10) break;
    }
    return roots;
}

void magma_colormap(double t, unsigned char* rgb) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    rgb[0] = (unsigned char)(pow(t, 0.4) * 255.0);
    rgb[1] = (unsigned char)(pow(t, 1.9) * 180.0);
    rgb[2] = (unsigned char)(pow(1.0 - t, 1.2) * 220.0);
}

// ------------------ Main Widget ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
    bool mode_algebraic = true;
    bool show_edges = true;
    
    // Viewport State
    double v_x = 0, v_y = 0, v_zoom = 1.0;
    int last_mx = 0, last_my = 0;

    // Simulation Params
    int max_deg = 5, max_c = 5;
    int p_prime = 3, n_ext = 2;
    double l_scale = 1.0;

    // Heatmap / Texture state
    const int res = 512;
    vector<unsigned char> tex_data;
    GLuint tex_id = 0;
    bool dirty_compute = true;

    // Algebraic details
    vector<int> adj_coeffs;
    vector<cd> adj_roots;
    cd chosen_alpha = 0;

    UnifiedGL(int X, int Y, int W, int H) : Fl_Gl_Window(X,Y,W,H) {
        tex_data.resize(res * res * 3, 0);
    }

    void compute_heatmap() {
        vector<double> heat(res * res, 0.0);
        mt19937 rng(1337);
        uniform_int_distribution<int> c_dist(-max_c, max_c);
        
        for (int i = 0; i < 20000; ++i) {
            int d = (i % max_deg) + 1;
            vector<cd> coeffs(d + 1);
            for (int k = 0; k <= d; ++k) coeffs[k] = (double)c_dist(rng);
            if (abs(coeffs.back()) < 0.5) coeffs.back() = 1.0;
            
            for (auto& r : durand_kerner(coeffs)) {
                // Map [-2, 2] to [0, res]
                int ix = (int)((r.real() + 2.0) / 4.0 * res);
                int iy = (int)((r.imag() + 2.0) / 4.0 * res);
                if (ix >= 0 && ix < res && iy >= 0 && iy < res) heat[iy * res + ix] += 1.0;
            }
        }

        double max_v = 0;
        for (double v : heat) max_v = max(max_v, v);
        for (int i = 0; i < res * res; ++i) {
            double t = log(1.0 + heat[i]) / log(1.0 + max_v);
            magma_colormap(t, &tex_data[i * 3]);
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
            double dx = (Fl::event_x() - last_mx);
            double dy = (Fl::event_y() - last_my);
            if (Fl::event_button() == 1) { // Pan
                v_x -= (dx / w()) * (4.0 / v_zoom);
                v_y += (dy / h()) * (4.0 / v_zoom);
            } else { // Zoom
                v_zoom *= (1.0 - dy / 100.0);
            }
            last_mx = Fl::event_x(); last_my = Fl::event_y();
            redraw(); return 1;
        }
        if (e == FL_MOUSEWHEEL) {
            v_zoom *= (1.0 - Fl::event_dy() * 0.1);
            redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }

    void draw() override {
        if (!valid()) {
            valid(1);
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        if (dirty_compute) { compute_heatmap(); dirty_compute = false; }

        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Synchronized Projection
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        double aspect = (double)w() / h();
        double hw = 2.0 * aspect / v_zoom;
        double hh = 2.0 / v_zoom;
        glOrtho(v_x - hw, v_x + hw, v_y - hh, v_y + hh, -1, 1);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        if (mode_algebraic) {
            // Render Heatmap as a world-space Quad spanning [-2, 2]
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, tex_id);
            glColor4f(1, 1, 1, 1);
            glBegin(GL_QUADS);
            glTexCoord2f(0,0); glVertex2f(-2,-2);
            glTexCoord2f(1,0); glVertex2f( 2,-2);
            glTexCoord2f(1,1); glVertex2f( 2, 2);
            glTexCoord2f(0,1); glVertex2f(-2, 2);
            glEnd();
            glDisable(GL_TEXTURE_2D);
        } else {
            render_finite();
        }
        
        draw_extension_lattice();
        draw_axes();
    }

    void render_finite() {
        int N = (int)pow(p_prime, n_ext); if (N > 10000) N = 10000;
        glPointSize(4.0f); glBegin(GL_POINTS);
        for (int i = 0; i < N; ++i) {
            int t = i; cd z(0, 0);
            for (int j = 0; j < n_ext; ++j) {
                z += polar(l_scale, 2.0 * M_PI * j / n_ext) * (double)(t % p_prime);
                t /= p_prime;
            }
            glColor3f(0.4f, 0.6f, 1.0f);
            glVertex2f(z.real(), z.imag());
        }
        glEnd();
    }

    void draw_extension_lattice() {
        if (adj_coeffs.size() < 2) return;
        int deg = adj_coeffs.size() - 1, b = 3;
        glPointSize(5.0f); glBegin(GL_POINTS);
        int total = (int)pow(2*b+1, min(deg, 3));
        for (int i = 0; i < total; ++i) {
            int t = i; cd z(0,0), p(1,0);
            for (int k = 0; k < min(deg, 3); ++k) {
                int c = (t % (2*b+1)) - b; t /= (2*b+1);
                z += (double)c * pow(chosen_alpha, k);
            }
            glColor4f(1, 0.8f, 0.2f, 0.7f);
            glVertex2f(z.real(), z.imag());
        }
        glEnd();
    }

    void draw_axes() {
        glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1,1,1,0.2f);
        glVertex2f(-10,0); glVertex2f(10,0); glVertex2f(0,-10); glVertex2f(0,10);
        glEnd();
        glBegin(GL_LINE_LOOP); for(int i=0; i<64; ++i) {
            double a = 2*M_PI*i/64.0; glVertex2f(cos(a), sin(a));
        } glEnd();
    }
};

struct Context { UnifiedGL* gl; Fl_Input* in; Fl_Browser* br; };

int main(int argc, char** argv) {
    Fl_Window* win = new Fl_Window(1200, 800, "Galois Visualizer (Sync View)");
    UnifiedGL* gl = new UnifiedGL(10, 10, 850, 780);
    
    Fl_Group* g = new Fl_Group(870, 10, 320, 780);
    Fl_Choice* m = new Fl_Choice(950, 40, 200, 25, "Mode");
    m->add("Algebraic"); m->add("Finite"); m->value(0);
    m->callback([](Fl_Widget* w, void* v){ 
        ((UnifiedGL*)v)->mode_algebraic = (((Fl_Choice*)w)->value()==0); ((UnifiedGL*)v)->redraw(); 
    }, gl);

    Fl_Value_Slider* s1 = new Fl_Value_Slider(950, 80, 200, 20, "Deg/P");
    s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 20); s1->value(5);
    s1->callback([](Fl_Widget* w, void* v){
        UnifiedGL* gl = (UnifiedGL*)v; gl->max_deg = gl->p_prime = (int)((Fl_Value_Slider*)w)->value();
        gl->dirty_compute = true; gl->redraw();
    }, gl);

    Fl_Input* in = new Fl_Input(950, 120, 200, 25, "Poly"); in->value("1,0,1");
    Fl_Browser* br = new Fl_Browser(870, 180, 310, 150, "Roots");
    br->type(FL_HOLD_BROWSER);
    
    Context* ctx = new Context{gl, in, br};
    
    Fl_Button* b = new Fl_Button(870, 150, 310, 25, "Adjoin Root");
    b->callback([](Fl_Widget*, void* v){
        Context* c = (Context*)v; string s = c->in->value(), t; vector<int> co;
        stringstream ss(s); while(getline(ss, t, ',')) co.push_back(stoi(t));
        c->gl->adj_coeffs = co; vector<cd> cd_co; for(int k:co) cd_co.push_back((double)k);
        c->gl->adj_roots = durand_kerner(cd_co);
        c->br->clear();
        for(auto& r : c->gl->adj_roots) {
            ostringstream os; os << r; c->br->add(os.str().c_str());
        }
        if(!c->gl->adj_roots.empty()) { c->br->select(1); c->gl->chosen_alpha = c->gl->adj_roots[0]; }
        c->gl->redraw();
    }, ctx);

    br->callback([](Fl_Widget* w, void* v){
        UnifiedGL* gl = (UnifiedGL*)v; int s = ((Fl_Browser*)w)->value();
        if(s > 0) { gl->chosen_alpha = gl->adj_roots[s-1]; gl->redraw(); }
    }, gl);

    g->end(); win->end(); win->resizable(gl); win->show();
    return Fl::run();
}
