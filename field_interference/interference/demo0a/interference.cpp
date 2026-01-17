// Unified Field Explorer (Restored Professional Version)
//
// Features:
//   - Synchronized GL Viewport (Pan/Zoom/Drag)
//   - High-Fidelity Algebraic Heatmaps (Gaussian Blur + Magma Colormap)
//   - Finite Field GF(p^n) Multiplicative Cycle Rendering
//   - Field Extension Lattice Generator with Root Selection
//   - Extension Presets (Cyclotomic, Radicals, etc.)
//   - Full Companion Matrix Analysis

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

// ------------------ Mathematical Utilities ------------------ //

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
    for (int iter = 0; iter < 150; ++iter) {
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
        if (max_change < 1e-11) break;
    }
    return roots;
}

vector<double> companion_matrix(const vector<int>& coeffs) {
    int n = (int)coeffs.size() - 1;
    if (n < 1) return {};
    vector<double> M(n * n, 0.0);
    for (int i = 1; i < n; ++i) M[i * n + (i - 1)] = 1.0;
    for (int i = 0; i < n; ++i) M[i * n + (n - 1)] = -(double)coeffs[i] / coeffs[n];
    return M;
}

void separable_blur(vector<double>& grid, int res, double sigma) {
    if (sigma <= 0.0) return;
    int radius = (int)ceil(3.0 * sigma);
    int ksize = 2 * radius + 1;
    vector<double> kernel(ksize);
    double sum = 0, s2 = 2.0 * sigma * sigma;
    for (int i = -radius; i <= radius; ++i) {
        kernel[i + radius] = exp(-(i * i) / s2);
        sum += kernel[i + radius];
    }
    for (double& v : kernel) v /= sum;

    vector<double> temp(res * res, 0.0);
    for (int y = 0; y < res; ++y) {
        for (int x = 0; x < res; ++x) {
            double s = 0;
            for (int j = -radius; j <= radius; ++j) {
                int nx = max(0, min(res - 1, x + j));
                s += grid[y * res + nx] * kernel[j + radius];
            }
            temp[y * res + x] = s;
        }
    }
    for (int x = 0; x < res; ++x) {
        for (int y = 0; y < res; ++y) {
            double s = 0;
            for (int j = -radius; j <= radius; ++j) {
                int ny = max(0, min(res - 1, y + j));
                s += temp[ny * res + x] * kernel[j + radius];
            }
            grid[y * res + x] = s;
        }
    }
}

void magma_colormap(double t, unsigned char* rgb) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    rgb[0] = (unsigned char)(pow(t, 0.45) * 255.0);
    rgb[1] = (unsigned char)(pow(t, 1.8) * 200.0);
    rgb[2] = (unsigned char)(pow(1.0 - t, 1.1) * 230.0);
}

// ------------------ Main Visualizer ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
    bool mode_algebraic = true;
    bool show_edges = true;
    
    // Viewport
    double v_x = 0, v_y = 0, v_zoom = 1.0;
    int last_mx = 0, last_my = 0;

    // Simulation Params
    int max_deg = 5, max_c = 5;
    int p_prime = 3, n_ext = 2;
    double l_scale = 1.0, blur_sigma = 0.8;

    // Heatmap / Texture
    const int res = 512;
    vector<unsigned char> tex_data;
    GLuint tex_id = 0;
    bool dirty_compute = true;

    // Galois / Extensions
    vector<int> adj_coeffs;
    vector<cd> adj_roots;
    cd chosen_alpha = 0;
    int chosen_idx = -1;

    UnifiedGL(int X, int Y, int W, int H) : Fl_Gl_Window(X,Y,W,H) {
        tex_data.resize(res * res * 3, 0);
    }

    void compute_heatmap() {
        vector<double> heat(res * res, 0.0);
        mt19937 rng(1337);
        uniform_int_distribution<int> c_dist(-max_c, max_c);
        
        int samples = 40000;
        for (int i = 0; i < samples; ++i) {
            int d = (i % max_deg) + 1;
            vector<cd> coeffs(d + 1);
            for (int k = 0; k <= d; ++k) coeffs[k] = (double)c_dist(rng);
            if (abs(coeffs.back()) < 0.5) coeffs.back() = 1.0;
            
            auto roots = durand_kerner(coeffs);
            for (auto& r : roots) {
                int ix = (int)((r.real() + 2.0) / 4.0 * res);
                int iy = (int)((r.imag() + 2.0) / 4.0 * res);
                if (ix >= 0 && ix < res && iy >= 0 && iy < res) heat[iy * res + ix] += 1.0;
            }
        }

        separable_blur(heat, res, blur_sigma);

        double max_v = 0;
        for (double v : heat) max_v = max(max_v, v);
        for (int i = 0; i < res * res; ++i) {
            double t = log(1.0 + heat[i] * 5.0) / log(1.0 + max_v * 5.0);
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
            if (Fl::event_button() == 1) { 
                v_x -= (dx / w()) * (4.0 / v_zoom);
                v_y += (dy / h()) * (4.0 / v_zoom);
            } else { 
                v_zoom *= (1.0 - dy / 100.0);
            }
            last_mx = Fl::event_x(); last_my = Fl::event_y();
            redraw(); return 1;
        }
        if (e == FL_MOUSEWHEEL) {
            v_zoom *= (1.0 - Fl::event_dy() * 0.1);
            if (v_zoom < 0.01) v_zoom = 0.01;
            redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }

    void draw() override {
        if (!valid()) {
            valid(1); glLoadIdentity();
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        if (dirty_compute && mode_algebraic) { compute_heatmap(); dirty_compute = false; }

        glClearColor(0.03f, 0.03f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        double aspect = (double)w() / h();
        double hw = 2.0 * aspect / v_zoom;
        double hh = 2.0 / v_zoom;
        glOrtho(v_x - hw, v_x + hw, v_y - hh, v_y + hh, -1, 1);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        if (mode_algebraic) {
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
        int p = max(2, p_prime), n = max(1, n_ext);
        int N = (int)pow(p, n); if (N > 15000) N = 15000;
        vector<cd> points(N);
        for (int i = 0; i < N; ++i) {
            int t = i; cd z(0, 0);
            for (int j = 0; j < n; ++j) {
                z += polar(l_scale, 2.0 * M_PI * j / n) * (double)(t % p);
                t /= p;
            }
            points[i] = z;
        }

        if (show_edges && N < 5000) {
            glLineWidth(1.2f); glBegin(GL_LINE_STRIP);
            int curr = 1, step = (p == 2) ? 1 : p - 1;
            for (int k = 0; k < min(N, 1200); ++k) {
                float alpha = 0.5f * (1.0f - (float)k/1200);
                glColor4f(0.3f, 0.7f, 1.0f, alpha);
                glVertex2f(points[curr].real(), points[curr].imag());
                curr = (curr * step) % N; if (curr == 0) curr = 1;
            }
            glEnd();
        }

        glPointSize(5.0f * (float)sqrt(v_zoom)); 
        glBegin(GL_POINTS);
        for (auto& z : points) {
            float hue = (float)(arg(z)/(2*M_PI) + 0.5);
            glColor4f(0.5f + 0.5f * hue, 0.6f, 1.0f, 0.9f);
            glVertex2f(z.real(), z.imag());
        }
        glEnd();
    }

    void draw_extension_lattice() {
        if (adj_coeffs.size() < 2 || chosen_idx < 0) return;
        int deg = adj_coeffs.size() - 1, b = 3;
        glPointSize(7.0f * (float)sqrt(v_zoom)); 
        glBegin(GL_POINTS);
        int total = (int)pow(2*b+1, min(deg, 3));
        for (int i = 0; i < total; ++i) {
            int t = i; cd z(0,0);
            for (int k = 0; k < min(deg, 3); ++k) {
                int c = (t % (2*b+1)) - b; t /= (2*b+1);
                z += (double)c * pow(chosen_alpha, k);
            }
            glColor4f(1.0f, 0.8f, 0.1f, 0.9f);
            glVertex2f(z.real(), z.imag());
        }
        glEnd();
    }

    void draw_axes() {
        glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1,1,1,0.25f);
        glVertex2f(-20,0); glVertex2f(20,0); glVertex2f(0,-20); glVertex2f(0,20);
        glEnd();
        glBegin(GL_LINE_LOOP); glColor4f(1,1,0.5f,0.3f);
        for(int i=0; i<128; ++i) {
            double a = 2*M_PI*i/128.0; glVertex2f(cos(a), sin(a));
        } glEnd();
    }
};

struct UIContext { 
    UnifiedGL* gl; 
    Fl_Input* poly_in; 
    Fl_Browser* root_br; 
    Fl_Multiline_Output* mat_out; 
};

int main(int argc, char** argv) {
    Fl_Window* win = new Fl_Window(1300, 850, "Unified Galois Explorer Pro");
    UnifiedGL* gl = new UnifiedGL(10, 10, 920, 830);
    
    Fl_Group* ctrl = new Fl_Group(940, 10, 350, 830);
    
    Fl_Box* b1 = new Fl_Box(940, 15, 350, 25, "--- Global Configuration ---"); b1->labelfont(FL_BOLD);
    
    Fl_Choice* mode = new Fl_Choice(1040, 45, 240, 25, "Mode");
    mode->add("Algebraic (Heatmap)"); mode->add("Finite Field GF(p^n)"); mode->value(0);
    mode->callback([](Fl_Widget* w, void* v){
        UnifiedGL* g = (UnifiedGL*)v; g->mode_algebraic = (((Fl_Choice*)w)->value() == 0); g->redraw();
    }, gl);

    Fl_Button* reset = new Fl_Button(1040, 75, 240, 25, "Reset Viewport");
    reset->callback([](Fl_Widget*, void* v){ 
        UnifiedGL* g = (UnifiedGL*)v; g->v_x = 0; g->v_y = 0; g->v_zoom = 1.0; g->redraw(); 
    }, gl);

    Fl_Value_Slider* s_deg = new Fl_Value_Slider(1040, 110, 240, 20, "Deg / P");
    s_deg->type(FL_HOR_NICE_SLIDER); s_deg->bounds(1, 31); s_deg->step(1); s_deg->value(5);
    s_deg->callback([](Fl_Widget* w, void* v){
        UnifiedGL* g = (UnifiedGL*)v; g->max_deg = g->p_prime = (int)((Fl_Value_Slider*)w)->value();
        g->dirty_compute = true; g->redraw();
    }, gl);

    Fl_Value_Slider* s_cof = new Fl_Value_Slider(1040, 140, 240, 20, "Coeff / N");
    s_cof->type(FL_HOR_NICE_SLIDER); s_cof->bounds(1, 20); s_cof->step(1); s_cof->value(5);
    s_cof->callback([](Fl_Widget* w, void* v){
        UnifiedGL* g = (UnifiedGL*)v; g->max_c = g->n_ext = (int)((Fl_Value_Slider*)w)->value();
        g->dirty_compute = true; g->redraw();
    }, gl);

    Fl_Value_Slider* s_lat = new Fl_Value_Slider(1040, 170, 240, 20, "Lattice Scale");
    s_lat->type(FL_HOR_NICE_SLIDER); s_lat->bounds(0.1, 5.0); s_lat->value(1.0);
    s_lat->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->l_scale = ((Fl_Value_Slider*)w)->value(); ((UnifiedGL*)v)->redraw(); }, gl);

    Fl_Check_Button* edges = new Fl_Check_Button(1040, 195, 240, 20, "Show Field Edges");
    edges->value(1); edges->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->show_edges = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->redraw(); }, gl);

    Fl_Box* b2 = new Fl_Box(940, 230, 350, 25, "--- Galois Extensions ---"); b2->labelfont(FL_BOLD);
    
    Fl_Choice* presets = new Fl_Choice(1040, 260, 240, 25, "Presets");
    presets->add("Gaussian (x^2+1)"); presets->add("Eisenstein (x^2+x+1)");
    presets->add("Golden (x^2-x-1)"); presets->add("Silver (x^2-2x-1)");
    presets->add("CubeRoot2 (x^3-2)"); presets->add("Cyclo5 (x^4+x^3+x^2+x+1)");
    
    Fl_Input* poly = new Fl_Input(1040, 290, 240, 25, "Coeffs (c0,c1..)"); poly->value("1,0,1");
    
    Fl_Browser* roots = new Fl_Browser(950, 355, 330, 120, "Extension Generator (alpha)"); 
    roots->type(FL_HOLD_BROWSER);
    
    Fl_Multiline_Output* mat = new Fl_Multiline_Output(950, 515, 330, 200, "Companion Matrix"); 
    mat->textfont(FL_COURIER); mat->textsize(12);

    UIContext* ctx = new UIContext{gl, poly, roots, mat};

    auto calc_cb = [](Fl_Widget*, void* v){
        UIContext* c = (UIContext*)v; string s = c->poly_in->value(), t; vector<int> co;
        stringstream ss(s); while(getline(ss, t, ',')) { try{co.push_back(stoi(t));}catch(...){} }
        if(co.size() < 2) return;
        c->gl->adj_coeffs = co; vector<cd> cd_co; for(int k:co) cd_co.push_back((double)k);
        c->gl->adj_roots = durand_kerner(cd_co);
        c->root_br->clear();
        for(size_t i=0; i<c->gl->adj_roots.size(); ++i) {
            ostringstream os; os << "Root " << i << ": " << fixed << setprecision(3) << c->gl->adj_roots[i];
            c->root_br->add(os.str().c_str());
        }
        if(!c->gl->adj_roots.empty()){ c->root_br->select(1); c->gl->chosen_idx = 0; c->gl->chosen_alpha = c->gl->adj_roots[0]; }
        
        auto M = companion_matrix(co);
        int n = co.size()-1; ostringstream ms; ms << fixed << setprecision(3);
        for(int r=0; r<n; ++r) {
            for(int cl=0; cl<n; ++cl) ms << setw(8) << M[r*n+cl] << (cl==n-1?"":",");
            ms << "\n";
        }
        c->mat_out->value(ms.str().c_str());
        c->gl->redraw();
    };

    Fl_Button* adj = new Fl_Button(950, 320, 330, 25, "Apply Polynomial Extension");
    adj->callback(calc_cb, ctx);

    presets->callback([](Fl_Widget* w, void* v){
        UIContext* c = (UIContext*)v; int val = ((Fl_Choice*)w)->value();
        if(val==0) c->poly_in->value("1,0,1"); 
        else if(val==1) c->poly_in->value("1,1,1");
        else if(val==2) c->poly_in->value("-1,-1,1");
        else if(val==3) c->poly_in->value("-1,-2,1");
        else if(val==4) c->poly_in->value("-2,0,0,1");
        else if(val==5) c->poly_in->value("1,1,1,1,1");
    }, ctx);

    roots->callback([](Fl_Widget* w, void* v){
        UIContext* c = (UIContext*)v; int s = ((Fl_Browser*)w)->value();
        if(s > 0) { c->gl->chosen_idx = s-1; c->gl->chosen_alpha = c->gl->adj_roots[s-1]; c->gl->redraw(); }
    }, ctx);

    ctrl->end(); win->end(); win->resizable(gl); win->show();
    return Fl::run();
}
