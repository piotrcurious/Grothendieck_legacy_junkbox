// Unified Field Explorer (3D Enhanced Version)
//
// Features:
//   - 3D Synchronized Viewport (Rotate: Right-Click, Pan: Left-Click, Zoom: Scroll)
//   - Z-Axis Stacking for Extension Lattices (Visualizing Degree as Height)
//   - Helical Multiplicative Cycles for Finite Fields
//   - High-Fidelity Heatmaps on a 3D Plane
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

// ------------------ Math Utils ------------------ //

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

void magma_colormap(double t, unsigned char* rgb) {
    t = t < 0 ? 0 : (t > 1 ? 1 : t);
    rgb[0] = (unsigned char)(pow(t, 0.45) * 255.0);
    rgb[1] = (unsigned char)(pow(t, 1.8) * 200.0);
    rgb[2] = (unsigned char)(pow(1.0 - t, 1.1) * 230.0);
}

// ------------------ Visualizer ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
    bool mode_algebraic = true;
    bool show_edges = true;
    
    // Viewport 3D State
    float v_x = 0, v_y = 0, v_zoom = 1.0;
    float rot_x = 0, rot_y = 0;
    int last_mx = 0, last_my = 0;

    // Simulation
    int max_deg = 5, max_c = 5;
    int p_prime = 3, n_ext = 2;
    double l_scale = 1.0;

    // Heatmap
    const int res = 512;
    vector<unsigned char> tex_data;
    GLuint tex_id = 0;
    bool dirty_compute = true;

    // Extensions
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
        
        for (int i = 0; i < 30000; ++i) {
            int d = (i % max_deg) + 1;
            vector<cd> coeffs(d + 1);
            for (int k = 0; k <= d; ++k) coeffs[k] = (double)c_dist(rng);
            if (abs(coeffs.back()) < 0.5) coeffs.back() = 1.0;
            for (auto& r : durand_kerner(coeffs)) {
                int ix = (int)((r.real() + 2.0) / 4.0 * res);
                int iy = (int)((r.imag() + 2.0) / 4.0 * res);
                if (ix >= 0 && ix < res && iy >= 0 && iy < res) heat[iy * res + ix] += 1.0;
            }
        }
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
            int dx = Fl::event_x() - last_mx;
            int dy = Fl::event_y() - last_my;
            if (Fl::event_button() == 1) { // Pan
                v_x += (dx / (float)w()) * (4.0f / v_zoom);
                v_y -= (dy / (float)h()) * (4.0f / v_zoom);
            } else if (Fl::event_button() == 3) { // Rotate
                rot_x += dy * 0.5f;
                rot_y += dx * 0.5f;
            }
            last_mx = Fl::event_x(); last_my = Fl::event_y();
            redraw(); return 1;
        }
        if (e == FL_MOUSEWHEEL) {
            v_zoom *= (1.0f - Fl::event_dy() * 0.1f);
            if (v_zoom < 0.01f) v_zoom = 0.01f;
            redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }

    void draw() override {
        if (!valid()) {
            valid(1);
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        if (dirty_compute && mode_algebraic) { compute_heatmap(); dirty_compute = false; }

        glClearColor(0.02f, 0.02f, 0.04f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        float aspect = (float)w() / h();
        gluPerspective(45.0, aspect, 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        glTranslatef(v_x, v_y, -5.0f / v_zoom);
        glRotatef(rot_x, 1, 0, 0);
        glRotatef(rot_y, 0, 1, 0);

        draw_axes();

        if (mode_algebraic) {
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, tex_id);
            glColor4f(1, 1, 1, 1);
            glBegin(GL_QUADS);
            glTexCoord2f(0,0); glVertex3f(-2,-2, 0);
            glTexCoord2f(1,0); glVertex3f( 2,-2, 0);
            glTexCoord2f(1,1); glVertex3f( 2, 2, 0);
            glTexCoord2f(0,1); glVertex3f(-2, 2, 0);
            glEnd();
            glDisable(GL_TEXTURE_2D);
        } else {
            render_finite_3d();
        }
        
        draw_extension_lattice_3d();
    }

    void render_finite_3d() {
        int p = max(2, p_prime), n = max(1, n_ext);
        int N = (int)pow(p, n); if (N > 10000) N = 10000;
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
            glLineWidth(1.5f); glBegin(GL_LINE_STRIP);
            int curr = 1, step = (p == 2) ? 1 : p - 1;
            for (int k = 0; k < min(N, 1000); ++k) {
                float alpha = 0.6f * (1.0f - (float)k/1000);
                float z_offset = (float)k * 0.002f; // Helical lift
                glColor4f(0.2f, 0.8f, 1.0f, alpha);
                glVertex3f((float)points[curr].real(), (float)points[curr].imag(), z_offset);
                curr = (curr * step) % N; if (curr == 0) curr = 1;
            }
            glEnd();
        }

        glPointSize(4.0f); glBegin(GL_POINTS);
        for (auto& z : points) {
            glColor4f(0.4f, 0.6f, 1.0f, 0.8f);
            glVertex3f((float)z.real(), (float)z.imag(), 0);
        }
        glEnd();
    }

    void draw_extension_lattice_3d() {
        if (adj_coeffs.size() < 2 || chosen_idx < 0) return;
        int deg = adj_coeffs.size() - 1, b = 3;
        glPointSize(6.0f); glBegin(GL_POINTS);
        int total = (int)pow(2*b+1, min(deg, 3));
        for (int i = 0; i < total; ++i) {
            int t = i; cd z(0,0);
            float z_stack = 0;
            for (int k = 0; k < min(deg, 3); ++k) {
                int c = (t % (2*b+1)) - b; t /= (2*b+1);
                z += (double)c * pow(chosen_alpha, k);
                z_stack += (float)c * 0.2f; // Depth per basis element
            }
            glColor4f(1.0f, 0.7f, 0.0f, 0.9f);
            glVertex3f((float)z.real(), (float)z.imag(), z_stack);
        }
        glEnd();
    }

    void draw_axes() {
        glLineWidth(1.0f); glBegin(GL_LINES);
        glColor4f(1,0,0,0.3f); glVertex3f(0,0,0); glVertex3f(5,0,0); // X
        glColor4f(0,1,0,0.3f); glVertex3f(0,0,0); glVertex3f(0,5,0); // Y
        glColor4f(0,0,1,0.3f); glVertex3f(0,0,0); glVertex3f(0,0,5); // Z
        glEnd();
        // Unit Circle
        glBegin(GL_LINE_LOOP); glColor4f(1,1,1,0.2f);
        for(int i=0; i<100; ++i) {
            double a = 2*M_PI*i/100.0; glVertex3f((float)cos(a), (float)sin(a), 0);
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
    Fl_Window* win = new Fl_Window(1300, 850, "Unified Galois Explorer 3D");
    UnifiedGL* gl = new UnifiedGL(10, 10, 920, 830);
    
    Fl_Group* ctrl = new Fl_Group(940, 10, 350, 830);
    new Fl_Box(940, 15, 350, 25, "--- 3D Global Controls ---");
    
    Fl_Choice* mode = new Fl_Choice(1040, 45, 240, 25, "Mode");
    mode->add("Algebraic (Plane)"); mode->add("Finite Field (Helical)"); mode->value(0);
    mode->callback([](Fl_Widget* w, void* v){
        UnifiedGL* g = (UnifiedGL*)v; g->mode_algebraic = (((Fl_Choice*)w)->value() == 0); g->redraw();
    }, gl);

    Fl_Button* reset = new Fl_Button(1040, 75, 240, 25, "Reset Camera");
    reset->callback([](Fl_Widget*, void* v){ 
        UnifiedGL* g = (UnifiedGL*)v; g->v_x=0; g->v_y=0; g->v_zoom=1.0f; g->rot_x=0; g->rot_y=0; g->redraw(); 
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

    Fl_Check_Button* edges = new Fl_Check_Button(1040, 175, 240, 20, "Show 3D Helices");
    edges->value(1); edges->callback([](Fl_Widget* w, void* v){ ((UnifiedGL*)v)->show_edges = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->redraw(); }, gl);

    new Fl_Box(940, 210, 350, 25, "--- Galois 3D Tower ---");
    Fl_Input* poly = new Fl_Input(1040, 240, 240, 25, "Extension"); poly->value("1,0,1");
    Fl_Browser* roots = new Fl_Browser(950, 305, 330, 120, "Generator Roots"); roots->type(FL_HOLD_BROWSER);
    Fl_Multiline_Output* mat = new Fl_Multiline_Output(950, 465, 330, 200, "Companion Matrix"); mat->textfont(FL_COURIER);

    UIContext* ctx = new UIContext{gl, poly, roots, mat};

    Fl_Button* adj = new Fl_Button(950, 270, 330, 25, "Adjoin & Build Tower");
    adj->callback([](Fl_Widget*, void* v){
        UIContext* c = (UIContext*)v; string s = c->poly_in->value(), t; vector<int> co;
        stringstream ss(s); while(getline(ss, t, ',')) { try{co.push_back(stoi(t));}catch(...){} }
        if(co.size() < 2) return;
        c->gl->adj_coeffs = co; vector<cd> cd_co; for(int k:co) cd_co.push_back((double)k);
        c->gl->adj_roots = durand_kerner(cd_co);
        c->root_br->clear();
        for(size_t i=0; i<c->gl->adj_roots.size(); ++i) {
            ostringstream os; os << "Root " << i << ": " << c->gl->adj_roots[i];
            c->root_br->add(os.str().c_str());
        }
        if(!c->gl->adj_roots.empty()){ c->root_br->select(1); c->gl->chosen_idx = 0; c->gl->chosen_alpha = c->gl->adj_roots[0]; }
        auto M = companion_matrix(co); int n = co.size()-1; ostringstream ms;
        for(int r=0; r<n; ++r) {
            for(int cl=0; cl<n; ++cl) ms << setw(8) << M[r*n+cl] << (cl==n-1?"":",");
            ms << "\n";
        }
        c->mat_out->value(ms.str().c_str());
        c->gl->redraw();
    }, ctx);

    roots->callback([](Fl_Widget* w, void* v){
        UIContext* c = (UIContext*)v; int s = ((Fl_Browser*)w)->value();
        if(s > 0) { c->gl->chosen_idx = s-1; c->gl->chosen_alpha = c->gl->adj_roots[s-1]; c->gl->redraw(); }
    }, ctx);

    ctrl->end(); win->end(); win->resizable(gl); win->show();
    return Fl::run();
}
