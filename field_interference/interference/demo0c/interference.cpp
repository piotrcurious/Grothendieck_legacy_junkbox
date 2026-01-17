// Unified Field Explorer (Professional 3D Edition)
//
// Features:
//   - High-Precision 3D Viewport with Depth Blending
//   - Basis-Staged Z-Axis Tower for Field Extensions
//   - Helical Multiplicative Flow for Finite Fields
//   - Structured UI with Mathematical Presets
//   - Professional Magma-to-Neon Color Mapping

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

// ------------------ Numerical Core ------------------ //

vector<cd> solve_roots(const vector<cd>& coeffs) {
    int n = (int)coeffs.size() - 1;
    if (n <= 0) return {};
    vector<cd> roots(n);
    for (int i = 0; i < n; ++i) roots[i] = polar(1.5, 2.0 * M_PI * i / n + 0.2);
    for (int iter = 0; iter < 100; ++iter) {
        double max_err = 0;
        for (int i = 0; i < n; ++i) {
            cd val = coeffs[n], deriv = 0;
            for (int j = n - 1; j >= 0; --j) val = val * roots[i] + coeffs[j];
            cd denom = 1.0;
            for (int j = 0; j < n; ++j) if (i != j) denom *= (roots[i] - roots[j]);
            cd delta = val / (denom == 0.0 ? 1e-15 : denom);
            roots[i] -= delta;
            max_err = max(max_err, abs(delta));
        }
        if (max_err < 1e-10) break;
    }
    return roots;
}

vector<double> get_companion(const vector<int>& c) {
    int n = (int)c.size() - 1;
    if (n < 1) return {};
    vector<double> M(n * n, 0.0);
    for (int i = 1; i < n; ++i) M[i * n + (i - 1)] = 1.0;
    for (int i = 0; i < n; ++i) M[i * n + (n - 1)] = -(double)c[i] / c[n];
    return M;
}

// ------------------ Rendering Engine ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
    bool mode_algebraic = true;
    bool show_edges = true;
    float v_x = 0, v_y = 0, v_zoom = 1.0, rot_x = 25, rot_y = -35;
    int last_mx = 0, last_my = 0;

    int max_deg = 5, max_c = 5;
    int p_prime = 3, n_ext = 2;
    float tower_height = 0.4f;

    const int res = 512;
    vector<unsigned char> tex_data;
    GLuint tex_id = 0;
    bool dirty_compute = true;

    vector<int> adj_coeffs;
    vector<cd> adj_roots;
    cd chosen_alpha = 0;
    int chosen_idx = -1;

    UnifiedGL(int X, int Y, int W, int H) : Fl_Gl_Window(X,Y,W,H) {
        tex_data.resize(res * res * 3, 0);
    }

    void compute_heatmap() {
        vector<double> heat(res * res, 0.0);
        mt19937 rng(42);
        uniform_int_distribution<int> c_dist(-max_c, max_c);
        for (int i = 0; i < 25000; ++i) {
            int d = (i % max_deg) + 1;
            vector<cd> c(d + 1);
            for (int k = 0; k <= d; ++k) c[k] = (double)c_dist(rng);
            if (abs(c.back()) < 0.1) c.back() = 1.0;
            for (auto& r : solve_roots(c)) {
                int ix = (int)((r.real() + 2.0) / 4.0 * res);
                int iy = (int)((r.imag() + 2.0) / 4.0 * res);
                if (ix >= 0 && ix < res && iy >= 0 && iy < res) heat[iy * res + ix] += 1.0;
            }
        }
        double mv = 0; for (double v : heat) mv = max(mv, v);
        for (int i = 0; i < res * res; ++i) {
            double t = log(1.0 + heat[i]*10) / log(1.0 + mv*10);
            tex_data[i*3] = (unsigned char)(pow(t, 0.4) * 255);
            tex_data[i*3+1] = (unsigned char)(pow(t, 1.5) * 180);
            tex_data[i*3+2] = (unsigned char)(pow(1.0-t, 1.2) * 200 + t * 50);
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
            int dx = Fl::event_x() - last_mx, dy = Fl::event_y() - last_my;
            if (Fl::event_button() == 1) { v_x += dx*0.01f/v_zoom; v_y -= dy*0.01f/v_zoom; }
            else { rot_x += dy; rot_y += dx; }
            last_mx = Fl::event_x(); last_my = Fl::event_y(); redraw(); return 1;
        }
        if (e == FL_MOUSEWHEEL) { v_zoom *= (1.0f - Fl::event_dy()*0.1f); redraw(); return 1; }
        return Fl_Gl_Window::handle(e);
    }

    void draw() override {
        if (!valid()) {
            valid(1); glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        }
        if (dirty_compute && mode_algebraic) { compute_heatmap(); dirty_compute = false; }
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(40.0, (float)w()/h(), 0.1, 100.0);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glTranslatef(v_x, v_y, -6.0f / v_zoom);
        glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0);

        // Grid Plane
        glLineWidth(1.0f); glBegin(GL_LINES);
        for(int i=-5; i<=5; ++i) {
            glColor4f(0.2f, 0.2f, 0.4f, 0.2f);
            glVertex3f(i, -5, 0); glVertex3f(i, 5, 0);
            glVertex3f(-5, i, 0); glVertex3f(5, i, 0);
        } glEnd();

        if (mode_algebraic) {
            glEnable(GL_TEXTURE_2D); glBindTexture(GL_TEXTURE_2D, tex_id);
            glColor4f(1,1,1,0.8f); glBegin(GL_QUADS);
            glTexCoord2f(0,0); glVertex3f(-2,-2, 0);
            glTexCoord2f(1,0); glVertex3f( 2,-2, 0);
            glTexCoord2f(1,1); glVertex3f( 2, 2, 0);
            glTexCoord2f(0,1); glVertex3f(-2, 2, 0);
            glEnd(); glDisable(GL_TEXTURE_2D);
        } else {
            render_gf_3d();
        }
        render_extension_3d();
    }

    void render_gf_3d() {
        int p = max(2, p_prime), n = max(1, n_ext);
        int N = (int)pow(p, n); if (N > 8000) N = 8000;
        vector<cd> pts(N);
        for (int i=0; i<N; ++i) {
            int t=i; cd z(0,0);
            for (int j=0; j<n; ++j) { z += polar(1.0, 2*M_PI*j/n) * (double)(t%p); t/=p; }
            pts[i] = z;
        }
        if (show_edges && N < 3000) {
            glLineWidth(1.5f); glBegin(GL_LINE_STRIP);
            int curr = 1, step = (p==2?1:p-1);
            for (int k=0; k<min(N, 1500); ++k) {
                float a = 0.7f * (1.0f - (float)k/1500);
                glColor4f(0.3f, 0.8f, 1.0f, a);
                glVertex3f((float)pts[curr].real(), (float)pts[curr].imag(), k*0.001f);
                curr = (curr * step) % N; if (curr == 0) curr = 1;
            } glEnd();
        }
        glPointSize(3.0f); glBegin(GL_POINTS);
        for(auto& z : pts) { glColor4f(0.4f, 0.5f, 1.0f, 0.6f); glVertex3f((float)z.real(), (float)z.imag(), 0); }
        glEnd();
    }

    void render_extension_3d() {
        if (adj_coeffs.size() < 2 || chosen_idx < 0) return;
        int deg = adj_coeffs.size()-1; int b = 2;
        glPointSize(5.0f); glBegin(GL_POINTS);
        int total = (int)pow(2*b+1, min(deg, 4));
        for (int i=0; i<total; ++i) {
            int t = i; cd z(0,0); float zh = 0;
            for (int k=0; k<min(deg, 4); ++k) {
                int c = (t % (2*b+1)) - b; t /= (2*b+1);
                z += (double)c * pow(chosen_alpha, k);
                zh += c * tower_height * (k + 1);
            }
            glColor4f(1.0f, 0.8f, 0.1f, 0.9f);
            glVertex3f((float)z.real(), (float)z.imag(), zh);
        } glEnd();
    }
};

struct UI {
    UnifiedGL* gl; Fl_Input* poly; Fl_Browser* roots; Fl_Multiline_Output* mat;
};

int main(int argc, char** argv) {
    Fl_Window* win = new Fl_Window(1250, 850, "Unified Galois Explorer - Master");
    UnifiedGL* gl = new UnifiedGL(10, 10, 880, 830);
    
    Fl_Group* g1 = new Fl_Group(900, 10, 340, 240, "Simulation Parameters");
    g1->box(FL_ENGRAVED_FRAME); g1->align(FL_ALIGN_TOP_LEFT);
    
    Fl_Choice* mode = new Fl_Choice(1000, 30, 230, 25, "Mode");
    mode->add("Complex Plane"); mode->add("Finite GF(p^n)"); mode->value(0);
    
    Fl_Value_Slider* sd = new Fl_Value_Slider(1000, 65, 230, 20, "Deg/P");
    sd->type(FL_HOR_NICE_SLIDER); sd->bounds(1, 31); sd->step(1); sd->value(5);
    
    Fl_Value_Slider* sc = new Fl_Value_Slider(1000, 95, 230, 20, "Cof/N");
    sc->type(FL_HOR_NICE_SLIDER); sc->bounds(1, 15); sc->step(1); sc->value(5);

    Fl_Value_Slider* sh = new Fl_Value_Slider(1000, 125, 230, 20, "Z-Step");
    sh->type(FL_HOR_NICE_SLIDER); sh->bounds(0.01, 1.0); sh->value(0.4);

    Fl_Check_Button* cb = new Fl_Check_Button(1000, 155, 230, 25, "Show 3D Flow");
    cb->value(1);

    Fl_Button* reset = new Fl_Button(910, 195, 320, 30, "Reset Camera & View");
    g1->end();

    Fl_Group* g2 = new Fl_Group(900, 260, 340, 580, "Galois Tower Builder");
    g2->box(FL_ENGRAVED_FRAME); g2->align(FL_ALIGN_TOP_LEFT);
    
    Fl_Choice* pre = new Fl_Choice(1000, 280, 230, 25, "Presets");
    pre->add("Gaussian (x^2+1)"); pre->add("Eisenstein (x^2+x+1)");
    pre->add("Silver (x^2-2x-1)"); pre->add("Cube Root 2 (x^3-2)");
    pre->add("Cyclotomic 5");

    Fl_Input* pi = new Fl_Input(1000, 315, 230, 25, "Poly"); pi->value("1,0,1");
    Fl_Button* run = new Fl_Button(910, 350, 320, 30, "Calculate & Adjoin");
    
    Fl_Browser* rb = new Fl_Browser(910, 405, 320, 120, "Basis Generators (Alpha)");
    rb->type(FL_HOLD_BROWSER);
    
    Fl_Multiline_Output* mo = new Fl_Multiline_Output(910, 555, 320, 275, "Transformation Matrix");
    mo->textfont(FL_COURIER); mo->textsize(11);
    g2->end();

    UI* ui = new UI{gl, pi, rb, mo};

    auto update_cb = [](Fl_Widget*, void* v){
        UI* u = (UI*)v; stringstream ss(u->poly->value()); string t; vector<int> co;
        while(getline(ss, t, ',')) { try{co.push_back(stoi(t));}catch(...){} }
        if(co.size()<2) return;
        u->gl->adj_coeffs = co; vector<cd> cdc; for(int k:co) cdc.push_back((double)k);
        u->gl->adj_roots = solve_roots(cdc); u->roots->clear();
        for(size_t i=0; i<u->gl->adj_roots.size(); ++i){
            ostringstream os; os << "α[" << i << "]: " << fixed << setprecision(2) << u->gl->adj_roots[i];
            u->roots->add(os.str().c_str());
        }
        if(!u->gl->adj_roots.empty()){ u->roots->select(1); u->gl->chosen_idx=0; u->gl->chosen_alpha=u->gl->adj_roots[0]; }
        auto M = get_companion(co); int n = co.size()-1; ostringstream ms;
        for(int r=0; r<n; ++r){ for(int c=0; c<n; ++c) ms << setw(7) << M[r*n+c] << (c==n-1?"":","); ms << "\n"; }
        u->mat->value(ms.str().c_str()); u->gl->redraw();
    };

    mode->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->mode_algebraic = (((Fl_Choice*)w)->value()==0); ((UnifiedGL*)v)->redraw(); }, gl);
    sd->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->max_deg = ((UnifiedGL*)v)->p_prime = (int)((Fl_Value_Slider*)w)->value(); ((UnifiedGL*)v)->dirty_compute=true; ((UnifiedGL*)v)->redraw(); }, gl);
    sc->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->max_c = ((UnifiedGL*)v)->n_ext = (int)((Fl_Value_Slider*)w)->value(); ((UnifiedGL*)v)->dirty_compute=true; ((UnifiedGL*)v)->redraw(); }, gl);
    sh->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->tower_height = (float)((Fl_Value_Slider*)w)->value(); ((UnifiedGL*)v)->redraw(); }, gl);
    cb->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->show_edges = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->redraw(); }, gl);
    reset->callback([](Fl_Widget*, void* v){ auto g=(UnifiedGL*)v; g->v_x=0; g->v_y=0; g->v_zoom=1; g->rot_x=25; g->rot_y=-35; g->redraw(); }, gl);
    run->callback(update_cb, ui);
    rb->callback([](Fl_Widget* w, void* v){ UI* u=(UI*)v; int s=((Fl_Browser*)w)->value(); if(s>0){ u->gl->chosen_idx=s-1; u->gl->chosen_alpha=u->gl->adj_roots[s-1]; u->gl->redraw(); } }, ui);
    
    pre->callback([](Fl_Widget* w, void* v){
        UI* u = (UI*)v; int i = ((Fl_Choice*)w)->value();
        if(i==0) u->poly->value("1,0,1"); if(i==1) u->poly->value("1,1,1");
        if(i==2) u->poly->value("-1,-2,1"); if(i==3) u->poly->value("-2,0,0,1");
        if(i==4) u->poly->value("1,1,1,1,1");
    }, ui);

    win->end(); win->resizable(gl); win->show();
    return Fl::run();
}
