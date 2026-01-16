// galois_grothendieck_fltk_demo.cpp // Single-file FLTK + OpenGL demonstration for Linux // Shows "interference" between two Galois viewpoints: //  - Galois field of numbers (integers mod p) and //  - Galois field of polynomials (polynomials over GF(p) reduced mod a modulus g(x)). // Uses plain C types (int, float) and small combinatorial examples to make the // algebraic geometry intuition concrete (Spec, reduction, pullbacks, localizations). // Compile (example): // g++ -std=c++17 -O2 galois_grothendieck_fltk_demo.cpp -o galois_demo 
//    -lfltk -lfltk_gl -lGL -lGLU

#include <FL/Fl.H> #include <FL/Fl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Button.H> #include <FL/Fl_Choice.H> #include <FL/Fl_Slider.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Gl_Window.H> #include <FL/gl.h> #include <vector> #include <string> #include <cmath> #include <sstream> #include <algorithm> #include <iostream>

using namespace std;

// --- Simple integer-based polynomial type (coeffs in Z, but we'll reduce mod p when needed) struct IntPoly { // coeffs[i] * x^i vector<int> coeffs; IntPoly(int deg=0){ coeffs.assign(deg+1,0); } int deg() const { for(int i=(int)coeffs.size()-1;i>=0;--i) if(coeffs[i]!=0) return i; return 0; } // evaluate as float (real polynomial using float casts) float eval_float(float x) const { float res = 0.0f; for(int i=deg(); i>=0; --i) res = res * x + (float)coeffs[i]; return res; } // evaluate over GF(p) at an integer x in 0..p-1 using Horner (int arithmetic) int eval_mod_p(int x, int p) const { int res = 0; int xm = ((x%p)+p)%p; for(int i=deg(); i>=0; --i){ res = ( (long long)res * xm + (coeffs[i]%p + p) ) % p; } return (res%p + p) % p; } string to_string() const { ostringstream ss; for(int i=0;i<(int)coeffs.size();++i){ ss << coeffs[i] << (i+1<(int)coeffs.size()? ",":""); } return ss.str(); } };

// Polynomial long division over Z/pZ: divides A by B modulo p, returns remainder R with deg R < deg B IntPoly rem_mod_poly(const IntPoly &A, const IntPoly &B, int p){ IntPoly R = A; int bp = B.deg(); if(bp==0 && B.coeffs[0]%p==0) return R; // degenerate // Make a mutable copy of coeffs mod p vector<int> rc = R.coeffs; while((int)rc.size() < bp+1) rc.push_back(0); // ensure top degree auto normalize = [&](vector<int> &v){ while(v.size()>1 && v.back()==0) v.pop_back(); }; normalize(rc); vector<int> bc = B.coeffs; normalize(bc); if(bc.empty()) return R; // find inverse of leading coeff of B mod p auto inv_mod = [&](int a)->int{ a = (a%p+p)%p; for(int t=1;t<p;++t) if((at)%p==1) return t; return 1; // fallback (p should be prime in our UI) }; int bl = bc.back()%p; if(bl<0) bl+=p; int blinv = inv_mod(bl); while((int)rc.size()-1 >= (int)bc.size()-1){ int shift = (int)rc.size() - (int)bc.size(); int lead = rc.back() % p; if(lead<0) lead+=p; int factor = (int)((1LLlead * blinv) % p); // subtract factor * bc * x^shift from rc for(int i=0;i<(int)bc.size();++i){ int idx = i + shift; int val = rc[idx] - (long long)factor * (bc[i]%p) % p; val %= p; if(val<0) val+=p; rc[idx] = val; } // trim normalize(rc); } IntPoly S; S.coeffs = rc; if(S.coeffs.empty()) S.coeffs = {0}; return S; }

// --- Demo GL widget --- class DemoGL : public Fl_Gl_Window { public: IntPoly *poly; // pointer to the polynomial controlled by UI int prime_p; IntPoly modulus_g; // modulus polynomial for polynomial field Fl_Box info_box; DemoGL(int X,int Y,int W,int H,const charL=0):Fl_Gl_Window(X,Y,W,H,L){ poly = nullptr; prime_p = 7; modulus_g = IntPoly(3); // default modulus x^3 + x + 1 -> coeffs [1,1,0,1] modulus_g.coeffs = {1,1,0,1}; info_box = nullptr; end(); } void draw() override { if (!valid()) { valid(1); glEnable(GL_POINT_SMOOTH); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); glViewport(0,0,w(),h()); glMatrixMode(GL_PROJECTION); glLoadIdentity(); // We'll draw in normalized device coords and handle transforms manually } glClearColor(0.12f,0.12f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

// split viewport: left = real float graph, right = GF(p) discrete points
    int W = w(); int H = h();
    int mid = W/2;

    // Draw left: real polynomial graph (x in [-2,2])
    glViewport(0,0,mid,H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2,2,-10,10,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_axes();
    if(poly) draw_real_poly();

    // Draw right: discrete GF(p) visualization
    glViewport(mid,0,W-mid,H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // map integer domain 0..p-1 to x in [0,1], y in [0,p-1]
    glOrtho(-0.1,1.1,-0.5,prime_p-0.5,-1,1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    draw_gf_points();

    // update textual info box if attached
    if(info_box && poly){
        std::ostringstream ss;
        ss << "poly coeffs (Z): " << poly->to_string() << "  |  p = " << prime_p << "  |  modulus g(x) = ";
        for(int i=0;i<(int)modulus_g.coeffs.size();++i){ ss << modulus_g.coeffs[i] << (i+1<(int)modulus_g.coeffs.size()?",":""); }
        info_box->label(ss.str().c_str());
    }
}

void draw_axes(){
    // axes center
    glColor3f(0.6f,0.6f,0.6f);
    glBegin(GL_LINES);
    glVertex2f(-2.0f,0.0f); glVertex2f(2.0f,0.0f);
    glVertex2f(0.0f,-10.0f); glVertex2f(0.0f,10.0f);
    glEnd();
}

void draw_real_poly(){
    glLineWidth(2.0f);
    glColor3f(0.2f,0.8f,0.2f);
    glBegin(GL_LINE_STRIP);
    for(int i=0;i<=400;++i){
        float x = -2.0f + 4.0f * i / 400.0f;
        float y = poly->eval_float(x);
        // clamp y to viewport
        if(y<-100) y=-100; if(y>100) y=100;
        glVertex2f(x,y);
    }
    glEnd();
    // also draw small sample points showing integer evaluation mapped to float
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for(int a= -3; a<=3; ++a){
        float x = (float)a; float y = poly->eval_float(x);
        glVertex2f(x,y);
    }
    glEnd();
}

void draw_gf_points(){
    // We'll show three layers:
    // 1) evaluations of polynomial as integers mod p at x=0..p-1 (discrete points)
    // 2) evaluations after reducing polynomial coefficients modulo p (same as 1 mathematically, but visually separate)
    // 3) evaluations of representative of [poly] in (GF(p)[x]/(g)) reduced to degree < deg(g), then evaluated at 0..p-1

    int p = prime_p;

    // 1: raw eval of integer polynomial with integer coefficients reduced mod p
    glPointSize(8.0f);
    glBegin(GL_POINTS);
    for(int x=0;x<p;++x){
        int v = poly->eval_mod_p(x,p);
        float fx = (float)x / (float)(p-1);
        float fy = (float)v;
        // green: raw
        glColor3f(0.2f,0.9f,0.2f);
        glVertex2f(fx,fy);
    }
    glEnd();

    // 2: coefficients reduced modulo p first (visual redundancy shown as small blue squares)
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for(int x=0;x<p;++x){
        IntPoly A = *poly; // make copy
        // reduce coefficients mod p for display
        for(auto &c : A.coeffs) c = ((c%p)+p)%p;
        int v = A.eval_mod_p(x,p);
        float fx = (float)x / (float)(p-1);
        float fy = (float)v;
        glColor3f(0.2f,0.6f,0.9f);
        glVertex2f(fx,fy);
    }
    glEnd();

    // 3: reduce representative mod g(x) (i.e., take remainder r = poly mod g) and evaluate r over GF(p)
    IntPoly r = rem_mod_poly(*poly, modulus_g, p);
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    for(int x=0;x<p;++x){
        int v = r.eval_mod_p(x,p);
        float fx = (float)x / (float)(p-1);
        float fy = (float)v;
        // red: result after quotient-ring reduction (class representative)
        glColor3f(0.9f,0.2f,0.2f);
        glVertex2f(fx,fy);
    }
    glEnd();

    // also draw lines connecting raw->representative to emphasize "interference"
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for(int x=0;x<p;++x){
        int raw = poly->eval_mod_p(x,p);
        int repr = r.eval_mod_p(x,p);
        float fx = (float)x / (float)(p-1);
        glColor3f(0.8f,0.6f,0.1f);
        glVertex2f(fx,(float)raw);
        glVertex2f(fx,(float)repr);
    }
    glEnd();

    // annotate domain ticks
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for(int x=0;x<p;++x){
        float fx = (float)x / (float)(p-1);
        glColor3f(0.8f,0.8f,0.8f);
        glVertex2f(fx,-0.25f);
    }
    glEnd();
}

};

// --- UI and glue --- int main(int argc, char **argv){ Fl_Window *win = new Fl_Window(100,100,1100,600,"Galois & Grothendieck: Interference Demo");

// polynomial with degree up to 5 for interactive control
IntPoly mainPoly(5);
mainPoly.coeffs = {1, 2, -1, 0, 0, 0};

DemoGL *glv = new DemoGL(10,50,1080,540);
glv->poly = &mainPoly;

Fl_Box *info = new Fl_Box(10,10,1080,30);
info->box(FL_FLAT_BOX);
info->labelfont(FL_HELVETICA_BOLD);
info->labelsize(12);
glv->info_box = info;

// Prime choice
Fl_Choice *prime_choice = new Fl_Choice(10,560,200,26,"prime p");
prime_choice->add("2"); prime_choice->add("3"); prime_choice->add("5"); prime_choice->add("7"); prime_choice->add("11"); prime_choice->add("13");
prime_choice->value(3); // index into list, 0-based -> '7'
prime_choice->callback([](Fl_Widget*w, void* ud){
    Fl_Choice *c = (Fl_Choice*)w; DemoGL *g = (DemoGL*)ud;
    const char *s = c->text(); if(!s) return;
    g->prime_p = atoi(s);
    g->redraw();
}, glv);

// modulus choice: preset modulus polynomials (small degree)
Fl_Choice *mod_choice = new Fl_Choice(220,560,300,26,"modulus g(x)");
// store labels and associated coeff vectors in lambda captures when selected
vector<pair<string, vector<int>>> mods = {
    {"x^3 + x + 1", {1,1,0,1}},
    {"x^3 + 2*x + 2", {2,2,0,1}},
    {"x^2 + 1", {1,0,1}},
    {"x^4 + x + 1", {1,1,0,0,1}}
};
for(auto &m: mods) mod_choice->add(m.first.c_str());
mod_choice->value(0);
mod_choice->callback([&mods](Fl_Widget*w, void* ud){
    Fl_Choice*c=(Fl_Choice*)w; DemoGL* g=(DemoGL*)ud; int idx=c->value();
    if(idx<0) return; g->modulus_g.coeffs = mods[idx].second; g->redraw();
}, glv);

// Sliders for coefficients
vector<Fl_Value_Slider*> sliders;
for(int i=0;i<6;++i){
    Fl_Value_Slider *s = new Fl_Value_Slider(540 + i*85,560,80,26, (string("a_")+to_string(i)).c_str());
    s->type(FL_HORIZONTAL);
    s->bounds(-13,13);
    s->step(1);
    s->value(mainPoly.coeffs[i]);
    s->align(FL_ALIGN_TOP);
    s->callback([&mainPoly, i, glv](Fl_Widget*w, void*){
        Fl_Value_Slider *sv = (Fl_Value_Slider*)w;
        mainPoly.coeffs[i] = (int)round(sv->value());
        glv->redraw();
    });
    sliders.push_back(s);
}

// Quick reset button
Fl_Button *reset = new Fl_Button(880,560,80,26,"reset");
reset->callback([&mainPoly, &sliders, glv](Fl_Widget*, void*){
    mainPoly.coeffs = {1,2,-1,0,0,0};
    for(int i=0;i<6;++i) sliders[i]->value(mainPoly.coeffs[i]);
    glv->redraw();
});

// small explanatory box
Fl_Box *help = new Fl_Box(980,520,120,80);
help->label("Left: real polynomial (float)\nRight: GF(p) discrete evals\nGreen/raw, Blue/coefs mod p, Red/repr mod g(x)");
help->box(FL_UP_BOX);

win->end();
win->resizable(win);
win->show(argc,argv);
return Fl::run();

}
