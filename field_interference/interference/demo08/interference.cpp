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

#include "../../galois_math.h"
#include "../../complex_math.h"
#include "../../interference_core.h"
#include "../../mappings.h"

using namespace std;

class UnifiedGL : public Fl_Gl_Window {
public:
    bool mode_algebraic = true;
    bool show_edges = true;
    bool show_inverse = false;
    double v_x = 0, v_y = 0, v_zoom = 1.0;
    int last_mx = 0, last_my = 0;
    int max_deg = 5, max_c = 5, p_prime = 3, n_ext = 2;
    double l_scale = 1.0;
    int mapping_type = 0;
    const int res = 512;
    vector<unsigned char> tex_data;
    GLuint tex_id = 0;
    bool dirty_compute = true;
    vector<int> adj_coeffs;
    vector<cd> adj_roots;
    cd chosen_alpha = 0;
    std::vector<double> heat;

    UnifiedGL(int X, int Y, int W, int H) : Fl_Gl_Window(X,Y,W,H) {
        tex_data.resize(res * res * 3, 0);
        heat.resize(res * res);
    }

    void magma_color(double t, unsigned char* rgb) {
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        rgb[0] = (unsigned char)(pow(t, 0.4) * 255.0);
        rgb[1] = (unsigned char)(pow(t, 1.9) * 180.0);
        rgb[2] = (unsigned char)(pow(1.0 - t, 1.2) * 220.0);
    }

    void compute_heatmap() {
        auto mapper = [&](cd z) {
            cd mz = interference::apply_mapping(z, (interference::MappingType)mapping_type, l_scale);
            if (show_inverse) mz = 1.0 / (mz + 1e-15);
            return mz;
        };
        interference::compute_threaded(40000, res, max_deg, max_c, heat, mapper);

        double max_v = 0;
        for (double v : heat) max_v = max(max_v, v);
        for (int i = 0; i < res * res; ++i) {
            double t = (max_v > 0) ? log(1.0 + heat[i]*5.0) / log(1.0 + max_v*5.0) : 0;
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
            double dx = (Fl::event_x() - last_mx);
            double dy = (Fl::event_y() - last_my);
            if (Fl::event_button() == 1) {
                v_x -= (dx / w()) * (4.0 / v_zoom);
                v_y += (dy / h()) * (4.0 / v_zoom);
            } else {
                v_zoom *= (1.0 - dy / 100.0);
            }
            last_mx = Fl::event_x(); last_my = Fl::event_y(); redraw(); return 1;
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
        if (dirty_compute && mode_algebraic) { compute_heatmap(); dirty_compute = false; }
        glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
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
        int N = (int)pow(p, n); if (N > 10000) N = 10000;
        glPointSize(4.0f); glBegin(GL_POINTS);
        for (int i = 0; i < N; ++i) {
            int t = i; cd z(0, 0);
            for (int j = 0; j < n; ++j) {
                z += polar(l_scale, 2.0 * M_PI * j / n) * (double)(t % p);
                t /= p;
            }
            cd mz = interference::apply_mapping(z, (interference::MappingType)mapping_type, l_scale);
            if (show_inverse) mz = 1.0 / (mz + 1e-15);
            glColor3f(0.4f, 0.6f, 1.0f);
            glVertex2f((float)mz.real(), (float)mz.imag());
        }
        glEnd();
    }

    void draw_extension_lattice() {
        if (adj_coeffs.size() < 2) return;
        int deg = adj_coeffs.size() - 1, b = 3;
        glPointSize(5.0f); glBegin(GL_POINTS);
        int total = (int)pow(2*b+1, min(deg, 3));
        for (int i = 0; i < total; ++i) {
            int t = i; vector<cd> pc;
            for (int k = 0; k < deg; ++k) { pc.push_back(cd((double)((t % (2*b+1)) - b), 0)); t /= (2*b+1); }
            cd z = complex_eval_poly(pc, chosen_alpha);
            cd mz = interference::apply_mapping(z, (interference::MappingType)mapping_type, l_scale);
            if (show_inverse) mz = 1.0 / (mz + 1e-15);
            glColor4f(1, 0.8f, 0.2f, 0.7f);
            glVertex2f((float)mz.real(), (float)mz.imag());
        }
        glEnd();
    }

    void draw_axes() {
        glLineWidth(1.0f); glBegin(GL_LINES); glColor4f(1,1,1,0.2f);
        glVertex2f(-10,0); glVertex2f(10,0); glVertex2f(0,-10); glVertex2f(0,10);
        glEnd();
        glBegin(GL_LINE_LOOP); for(int i=0; i<64; ++i) {
            cd z = interference::apply_mapping(polar(1.0, 2*M_PI*i/64.0), (interference::MappingType)mapping_type, l_scale);
            if (show_inverse) z = 1.0 / (z + 1e-15);
            glVertex2f((float)z.real(), (float)z.imag());
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

    Fl_Choice* map = new Fl_Choice(950, 75, 200, 25, "Mapping");
    map->add("Standard"); map->add("Log-Polar"); map->add("Mobius"); map->add("Euler Space"); map->add("Reciprocal"); map->value(0);
    map->callback([](Fl_Widget* w, void* v){
        ((UnifiedGL*)v)->mapping_type = ((Fl_Choice*)w)->value(); ((UnifiedGL*)v)->dirty_compute = true; ((UnifiedGL*)v)->redraw();
    }, gl);

    Fl_Check_Button* inv_btn = new Fl_Check_Button(950, 110, 200, 25, "Invert");
    inv_btn->callback([](Fl_Widget* w, void* v){
        ((UnifiedGL*)v)->show_inverse = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->dirty_compute = true; ((UnifiedGL*)v)->redraw();
    }, gl);

    Fl_Value_Slider* s1 = new Fl_Value_Slider(950, 145, 200, 20, "Deg/P");
    s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 20); s1->value(5);
    s1->callback([](Fl_Widget* w, void* v){
        UnifiedGL* gl = (UnifiedGL*)v; gl->max_deg = gl->p_prime = (int)((Fl_Value_Slider*)w)->value();
        gl->dirty_compute = true; gl->redraw();
    }, gl);

    Fl_Input* in = new Fl_Input(950, 180, 200, 25, "Poly"); in->value("1,0,1");
    Fl_Browser* br = new Fl_Browser(870, 240, 310, 150, "Roots");
    br->type(FL_HOLD_BROWSER);

    Context* ctx = new Context{gl, in, br};

    Fl_Button* b = new Fl_Button(870, 210, 310, 25, "Adjoin Root");
    b->callback([](Fl_Widget*, void* v){
        Context* c = (Context*)v; string s = c->in->value(), t; vector<int> co;
        for(auto &ch:s) if(ch==',') ch=' ';
        stringstream ss(s); int val; while(ss >> val) co.push_back(val);
        if (co.size() < 2) return;
        c->gl->adj_coeffs = co; vector<cd> cd_co; for(int k:co) cd_co.push_back((double)k);
        c->gl->adj_roots = dk_solve_roots(cd_co);
        c->br->clear();
        for(auto& r : c->gl->adj_roots) { ostringstream os; os << r; c->br->add(os.str().c_str()); }
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
