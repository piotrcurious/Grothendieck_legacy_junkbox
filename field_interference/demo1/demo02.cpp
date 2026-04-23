#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Input.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/gl.h>

#include <vector>
#include <string>
#include <complex>
#include <sstream>
#include <algorithm>
#include <iostream>

#include "../complex_math.h"
#include "../galois_math.h"

using namespace std;

class GrothendieckGL : public Fl_Gl_Window {
public:
    int p = 11;
    GFElement g = {2, 0, 1};
    GFElement A = {1, 2, 1};
    bool show_splitting = false;

    GrothendieckGL(int X, int Y, int W, int H) : Fl_Gl_Window(X, Y, W, H) {}

    void draw() override {
        if (!valid()) {
            valid(1);
            glEnable(GL_POINT_SMOOTH);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        int W = w(), H = h();
        if (!show_splitting) draw_field_view(W, H);
        else draw_complex_splitting(W, H);
    }

    void draw_field_view(int W, int H) {
        glViewport(0, 0, W/2, H);
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        glOrtho(-0.5, p - 0.5, -1, p, -1, 1);
        glColor4f(1, 1, 1, 0.05f);
        glBegin(GL_LINES);
        for(int i=0; i<p; ++i) { glVertex2f(i, -1); glVertex2f(i, p); glVertex2f(-1, i); glVertex2f(p, i); }
        glEnd();
        GFElement r = gf_poly_mod(A, g, p);
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        for (int x = 0; x < p; ++x) {
            int valA = eval_gf(A, x), valr = eval_gf(r, x);
            glColor4f(0.3f, 0.6f, 1.0f, 0.6f); glVertex2f(x, valA);
            glColor4f(1.0f, 0.3f, 0.3f, 0.6f); glVertex2f(x, valr);
            if (eval_gf(g, x) == 0) { glColor3f(1, 1, 0); glPointSize(12); glVertex2f(x, valA); glPointSize(8); }
        }
        glEnd();
        glBegin(GL_LINES);
        for (int x = 0; x < p; ++x) {
            int valA = eval_gf(A, x), valr = eval_gf(r, x);
            if (valA == valr) glColor4f(1, 1, 0, 0.2f); else glColor4f(1, 1, 1, 0.1f);
            glVertex2f(x, valA); glVertex2f(x, valr);
        }
        glEnd();
        glViewport(W/2, 0, W/2, H);
        glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1, 1, -1, 1, -1, 1);
        glBegin(GL_LINE_LOOP); glColor4f(0.5, 0.5, 0.5, 0.5);
        for(int i=0; i<64; ++i) { double th = 2*M_PI*i/64; glVertex2f(0.7*cos(th), 0.7*sin(th)); }
        glEnd();
        glPointSize(6.0f); glBegin(GL_POINTS);
        for(int i=0; i<p; ++i) {
            double th = 2*M_PI*i/p;
            if (eval_gf(g, i) == 0) glColor3f(1, 1, 0); else glColor3f(0.4, 0.4, 0.4);
            glVertex2f(0.7*cos(th), 0.7*sin(th));
        }
        glEnd();
    }

    void draw_complex_splitting(int W, int H) {
        glViewport(0, 0, W, H);
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        double aspect = (double)W/H; glOrtho(-2*aspect, 2*aspect, -2, 2, -1, 1);
        glColor4f(1, 1, 1, 0.2); glBegin(GL_LINES); glVertex2f(-10,0); glVertex2f(10,0); glVertex2f(0,-10); glVertex2f(0,10); glEnd();
        vector<cd> gc; for(int c : g) gc.push_back(cd(c, 0));
        auto roots = dk_solve_roots(gc);
        vector<cd> Ac; for(int c : A) Ac.push_back(cd(c, 0));
        glPointSize(10.0f);
        for (auto& alpha : roots) {
            cd valA = complex_eval_poly(Ac, alpha);
            glBegin(GL_POINTS); glColor3f(1, 1, 0); glVertex2f(alpha.real(), alpha.imag()); glColor3f(0, 1, 1); glVertex2f(valA.real(), valA.imag()); glEnd();
            glBegin(GL_LINES); glColor4f(1, 1, 1, 0.3); glVertex2f(alpha.real(), alpha.imag()); glVertex2f(valA.real(), valA.imag()); glEnd();
        }
    }

    int eval_gf(const GFElement& poly, int x) {
        int res = 0; long long xp = 1;
        for (int c : poly) { res = (int)((res + 1LL * c * xp) % p); xp = (xp * x) % p; }
        return (res + p) % p;
    }
};

struct UI { GrothendieckGL* gl; Fl_Value_Slider* sp; Fl_Input* ig; Fl_Input* iA; };

int main(int argc, char **argv) {
    Fl_Window *win = new Fl_Window(1200, 800, "Grothendieck Viewpoint");
    GrothendieckGL *gl = new GrothendieckGL(10, 10, 880, 780);
    Fl_Group *ctrl = new Fl_Group(900, 10, 290, 780); ctrl->box(FL_UP_BOX);
    Fl_Value_Slider *s_p = new Fl_Value_Slider(910, 40, 270, 25, "p"); s_p->type(FL_HOR_NICE_SLIDER); s_p->bounds(2, 53); s_p->value(11); s_p->step(1);
    Fl_Input *in_g = new Fl_Input(910, 100, 270, 25, "g(x)"); in_g->value("-1, 0, 1");
    Fl_Input *in_A = new Fl_Input(910, 160, 270, 25, "A(x)"); in_A->value("1, 2, 1");
    Fl_Check_Button *chk = new Fl_Check_Button(910, 210, 270, 25, "Split");
    UI *ui = new UI{gl, s_p, in_g, in_A};
    auto upd = [](Fl_Widget*, void* v) {
        UI* u = (UI*)v; u->gl->p = (int)u->sp->value();
        auto parse = [](string s) {
            GFElement res; for(auto &c:s) if(c==',') c=' ';
            stringstream ss(s); int val; while(ss >> val) res.push_back(val);
            return res;
        };
        u->gl->g = parse(u->ig->value()); u->gl->A = parse(u->iA->value()); u->gl->redraw();
    };
    s_p->callback(upd, ui); in_g->callback(upd, ui); in_A->callback(upd, ui);
    chk->callback([](Fl_Widget* w, void* v){ ((GrothendieckGL*)v)->show_splitting = ((Fl_Check_Button*)w)->value(); ((GrothendieckGL*)v)->redraw(); }, gl);
    ctrl->end(); win->end(); win->show(); return Fl::run();
}
