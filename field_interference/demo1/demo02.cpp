// galois_grothendieck_fltk_demo.cpp // FLTK + OpenGL single-file demo (improved) // Demonstrates the logical derivation of "interference" between: //  - the Galois field of coefficients (scalars) used, and //  - the polynomial quotient ring (GF(p)[x]/(g(x))) seen as functions on the variety V(g). // // Key mathematical point made explicit in code and visualization: //  - Reduction map rho_p: Z[x] -> (Z/pZ)[x] (coefficients reduced mod p). //  - Quotient map pi_g: (Z/pZ)[x] -> (Z/pZ)[x]/(g) (classes modulo g). //  - Equality of classes pi_g(rho_p(A)) = pi_g(rho_p(B)) does NOT imply //    equality of evaluations A(a) and B(a) for arbitrary a in GF(p). //    Instead, A(a) = B(a) for all a in V(g) (the roots of g in GF(p)). //  - This is the geometric / Grothendieck viewpoint: functions in the quotient //    ring are well-defined on the variety V(g) (they descend to functions on V(g)). // // The visualization shows three columns: //  Left:    real/float interpretation of polynomial (int -> float evaluation) //  Middle:  evaluation as functions on GF(p) (x in GF(p)), raw vs coeff-reduced //  Right:   quotient-class representative evaluation; highlights points in V(g) //           where class-equality implies pointwise-equality. // // Build (Linux): // g++ -std=c++17 -O2 galois_grothendieck_fltk_demo.cpp -o galois_demo 
//    -lfltk -lfltk_gl -lGL -lGLU

#include <FL/Fl.H> #include <FL/Fl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Button.H> #include <FL/Fl_Choice.H> #include <FL/Fl_Slider.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Gl_Window.H> #include <FL/gl.h> #include <vector> #include <string> #include <cmath> #include <sstream> #include <algorithm> #include <iostream>

using namespace std;

// --- Lightweight integer polynomial representation --- struct IntPoly { vector<int> coeffs; // coeffs[i] * x^i IntPoly(int deg=0){ coeffs.assign(max(1,deg+1),0); } int deg() const { for(int i=(int)coeffs.size()-1;i>=0;--i) if(coeffs[i]!=0) return i; return 0; } // Real evaluation via Horner (float) float eval_float(float x) const { float res = 0.0f; for(int i=deg(); i>=0; --i) res = res * x + (float)coeffs[i]; return res; } // Evaluate with integer arithmetic then reduce mod p (Horner in Z then mod p) long long eval_integer(long long x) const { long long res = 0; for(int i=deg(); i>=0; --i) res = res * x + (long long)coeffs[i]; return res; } // Evaluate over GF(p) with x in 0..p-1 int eval_mod_p(int x, int p) const { int res = 0; int xm = ((x%p)+p)%p; for(int i=deg(); i>=0; --i){ res = (int)((1LL*res * xm + ((coeffs[i]%p)+p)%p) % p); } return (res%p + p) % p; } string to_string() const { ostringstream ss; for(int i=0;i<(int)coeffs.size();++i){ ss << coeffs[i] << (i+1<(int)coeffs.size()? ",":""); } return ss.str(); } };

// Reduce coefficients mod p: the reduction map rho_p : Z[x] -> (Z/pZ)[x] IntPoly reduce_coeffs_mod_p(const IntPoly &A, int p){ IntPoly R((int)A.coeffs.size()-1); R.coeffs.resize(A.coeffs.size()); for(size_t i=0;i<A.coeffs.size();++i) R.coeffs[i] = ((A.coeffs[i]%p)+p)%p; return R; }

// Polynomial long division in (Z/pZ)[x] : compute remainder of A by B modulo p IntPoly rem_mod_poly(const IntPoly &A_in, const IntPoly &B_in, int p){ // Work on copies with coefficients in 0..p-1 IntPoly A = reduce_coeffs_mod_p(A_in, p); IntPoly B = reduce_coeffs_mod_p(B_in, p); auto normalize = [&](vector<int> &v){ while(v.size()>1 && v.back()==0) v.pop_back(); }; normalize(A.coeffs); normalize(B.coeffs); if(B.coeffs.empty()) return A; int degA = (int)A.coeffs.size()-1; int degB = (int)B.coeffs.size()-1; if(degA < degB) return A; // inverse of leading coeff of B mod p (p assumed prime for UI) auto inv_mod = [&](int a)->int{ a = (a%p+p)%p; for(int t=1;t<p;++t) if((at)%p==1) return t; return 1; }; int bl = B.coeffs.back(); int blinv = inv_mod(bl); vector<int> rc = A.coeffs; while((int)rc.size()-1 >= degB){ int shift = (int)rc.size()-1 - degB; int lead = rc.back(); int factor = (int)((1LLlead * blinv) % p); // subtract factor * B * x^shift for(int i=0;i<=degB;++i){ int idx = i + shift; int val = rc[idx] - (int)((1LL * factor * B.coeffs[i]) % p); val %= p; if(val<0) val+=p; rc[idx] = val; } while(rc.size()>1 && rc.back()==0) rc.pop_back(); } IntPoly R; R.coeffs = rc; if(R.coeffs.empty()) R.coeffs = {0}; return R; }

// Find roots of a polynomial in GF(p) by brute force (returns set V(g) subset of GF(p)) vector<int> roots_in_GF(const IntPoly &G_in, int p){ IntPoly G = reduce_coeffs_mod_p(G_in, p); vector<int> roots; for(int a=0;a<p;++a) if(G.eval_mod_p(a,p)==0) roots.push_back(a); return roots; }

// --- GL widget --- class DemoGL : public Fl_Gl_Window { public: IntPoly *poly; // integer polynomial A(x) IntPoly modulus_g; // modulus polynomial g(x) int prime_p; Fl_Box info_box; DemoGL(int X,int Y,int W,int H,const charL=0):Fl_Gl_Window(X,Y,W,H,L){ poly = nullptr; prime_p = 7; modulus_g = IntPoly(3); modulus_g.coeffs = {1,1,0,1}; info_box=nullptr; end(); } void draw() override { if (!valid()) { valid(1); glEnable(GL_POINT_SMOOTH); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); } glClearColor(0.12f,0.12f,0.12f,1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); int W=w(), H=h(); int colW = W/3;

// Left: real/float graph
    glViewport(0,0,colW,H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-2,2,-10,10,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_axes(-2,2,-10,10);
    if(poly) draw_real_poly();

    // Middle: GF(p) as discrete domain points 0..p-1
    glViewport(colW,0,colW,H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-0.5, (double)prime_p-0.5, -0.5, prime_p-0.5, -1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_gf_middle();

    // Right: quotient-class viewpoint and V(g)
    glViewport(2*colW,0,colW,H);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-0.5, (double)prime_p-0.5, -0.5, prime_p-0.5, -1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_gf_right();

    if(info_box && poly){
        ostringstream ss; ss << "A(x) coeffs: " << poly->to_string() << "  |  p=" << prime_p << "  |  g(x)=";
        for(size_t i=0;i<modulus_g.coeffs.size();++i) ss << modulus_g.coeffs[i] << (i+1<modulus_g.coeffs.size()?",":"");
        info_box->label(ss.str().c_str());
    }
}

void draw_axes(double xl,double xr,double yl,double yr){
    glColor3f(0.6f,0.6f,0.6f); glBegin(GL_LINES);
    glVertex2f((float)xl,0.0f); glVertex2f((float)xr,0.0f);
    glVertex2f(0.0f,(float)yl); glVertex2f(0.0f,(float)yr);
    glEnd();
}

void draw_real_poly(){
    glLineWidth(2.0f); glColor3f(0.2f,0.8f,0.2f); glBegin(GL_LINE_STRIP);
    for(int i=0;i<=800;++i){ float x=-2.0f + 4.0f*i/800.0f; float y=poly->eval_float(x); if(y<-100) y=-100; if(y>100) y=100; glVertex2f(x,y);} glEnd();
    glPointSize(6.0f); glBegin(GL_POINTS); for(int a=-3;a<=3;++a){ float x=(float)a; float y=poly->eval_float(x); glVertex2f(x,y);} glEnd();
}

void draw_gf_middle(){
    int p = prime_p;
    // Raw evaluation A(a) reduced mod p (as integers then mod p) -> shown as green
    glPointSize(10.0f); glBegin(GL_POINTS);
    for(int a=0;a<p;++a){ int val = (int)((poly->eval_integer(a) % p + p) % p); glColor3f(0.2f,0.9f,0.2f); glVertex2f((float)a,(float)val);} glEnd();
    // Coeff-reduced evaluation rho_p(A)(a) -> blue squares
    IntPoly Ar = reduce_coeffs_mod_p(*poly,p);
    glPointSize(8.0f); glBegin(GL_POINTS);
    for(int a=0;a<p;++a){ int val = Ar.eval_mod_p(a,p); glColor3f(0.2f,0.6f,0.9f); glVertex2f((float)a,(float)val);} glEnd();
    // annotate equality places between raw and coeff-reduced (they should always be equal mathematically)
    glBegin(GL_LINES); glLineWidth(1.0f);
    for(int a=0;a<p;++a){ int raw = (int)((poly->eval_integer(a) % p + p)%p); int red = Ar.eval_mod_p(a,p); if(raw!=red){ glColor3f(0.9f,0.2f,0.2f);} else { glColor3f(0.8f,0.8f,0.1f);} glVertex2f((float)a,(float)raw); glVertex2f((float)a+0.1f,(float)red); }
    glEnd();
    // Draw axes ticks
    glColor3f(0.8f,0.8f,0.8f); glPointSize(2.0f); glBegin(GL_POINTS); for(int a=0;a<p;++a) glVertex2f((float)a,-0.25f); glEnd();
}

void draw_gf_right(){
    int p = prime_p;
    // Representative in quotient: r = A mod g (performed in (Z/pZ)[x])
    IntPoly Ar = reduce_coeffs_mod_p(*poly,p);
    IntPoly r = rem_mod_poly(Ar, modulus_g, p);
    // Evaluate r at each a in GF(p)
    glPointSize(10.0f); glBegin(GL_POINTS);
    for(int a=0;a<p;++a){ int val = r.eval_mod_p(a,p); glColor3f(0.9f,0.2f,0.2f); glVertex2f((float)a,(float)val);} glEnd();
    // Show set V(g): roots of g in GF(p)
    vector<int> V = roots_in_GF(modulus_g, p);
    // highlight these x positions and show that A(a) and r(a) agree on V
    glPointSize(14.0f); glBegin(GL_POINTS);
    for(int v : V){ // yellow marker at (v, A(v)) and green checked line
        int Av = (int)((poly->eval_integer(v)%p + p)%p);
        int rv = r.eval_mod_p(v,p);
        if(Av==rv) glColor3f(0.2f,0.9f,0.2f); else glColor3f(0.9f,0.6f,0.1f);
        glVertex2f((float)v,(float)rv);
    }
    glEnd();
    // Draw lines connecting A(a) (from middle) to r(a) (here) to show where they differ
    glLineWidth(1.0f); glBegin(GL_LINES);
    for(int a=0;a<p;++a){ int raw = (int)((poly->eval_integer(a)%p + p)%p); int rv = r.eval_mod_p(a,p); glColor3f(0.6f,0.6f,0.6f); glVertex2f((float)a,(float)raw); glVertex2f((float)a,(float)rv); }
    glEnd();
    // Annotate V(g) labels
    glColor3f(0.95f,0.95f,0.3f); for(int v : V){ draw_label((float)v, -0.35f, string("root:" + to_string(v))); }
}

void draw_label(float x, float y, const string &txt){ // tiny built-in bitmap label using raster
    // NOTE: FLTK/OpenGL text is platform dependent; this is a lightweight placeholder using glRasterPos
    glRasterPos2f(x,y);
    for(char c : txt) glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, c);
}

};

// --- UI glue --- int main(int argc, char **argv){ Fl_Window *win = new Fl_Window(50,50,1200,700,"Galois + Grothendieck: logical demo (improved)"); IntPoly mainPoly(6); mainPoly.coeffs = {3, -2, 5, 0, 1, 0, 0}; DemoGL *glv = new DemoGL(10,50,1180,580); glv->poly = &mainPoly; Fl_Box *info = new Fl_Box(10,10,1180,30); info->box(FL_FLAT_BOX); info->labelfont(FL_HELVETICA_BOLD); info->labelsize(12); glv->info_box = info;

Fl_Choice *prime_choice = new Fl_Choice(10,640,160,28,"prime p");
vector<int> primes = {2,3,5,7,11,13,17,19}; for(int pr : primes) prime_choice->add(to_string(pr).c_str()); prime_choice->value(3);
prime_choice->callback([](Fl_Widget*w, void*ud){ Fl_Choice*c=(Fl_Choice*)w; DemoGL*g=(DemoGL*)ud; const char*s=c->text(); if(!s) return; g->prime_p = atoi(s); g->redraw(); }, glv);

vector<pair<string, vector<int>>> mods = {{"x^3 + x + 1",{1,1,0,1}}, {"x^2 + 1",{1,0,1}}, {"x^3 + 2*x + 2",{2,2,0,1}}, {"x^4 + x + 1",{1,1,0,0,1}} };
Fl_Choice *mod_choice = new Fl_Choice(190,640,300,28,"modulus g(x)"); for(auto &m:mods) mod_choice->add(m.first.c_str()); mod_choice->value(0);
mod_choice->callback([&mods](Fl_Widget*w, void*ud){ Fl_Choice*c=(Fl_Choice*)w; DemoGL*g=(DemoGL*)ud; int idx=c->value(); if(idx<0) return; g->modulus_g.coeffs = mods[idx].second; g->redraw(); }, glv);

vector<Fl_Value_Slider*> sliders; for(int i=0;i<7;++i){ Fl_Value_Slider*s=new Fl_Value_Slider(520 + i*90,640,80,28, (string("a_")+to_string(i)).c_str()); s->type(FL_HORIZONTAL); s->bounds(-20,20); s->step(1); s->value(mainPoly.coeffs[i]); s->align(FL_ALIGN_TOP);
    s->callback([&mainPoly, i, glv](Fl_Widget*w, void*){ Fl_Value_Slider*sv=(Fl_Value_Slider*)w; mainPoly.coeffs[i]=(int)round(sv->value()); glv->redraw(); }); sliders.push_back(s); }

Fl_Button *reset = new Fl_Button(960,640,80,28,"reset"); reset->callback([&mainPoly, &sliders, glv](Fl_Widget*,void*){ mainPoly.coeffs = {3,-2,5,0,1,0,0}; for(int i=0;i<(int)sliders.size();++i) sliders[i]->value(mainPoly.coeffs[i]); glv->redraw(); });

Fl_Box *help = new Fl_Box(1050,590,140,110);
help->box(FL_UP_BOX);
help->label("Left: float eval

Middle: A(a) (integer) vs rho_p(A)(a) Right: class rep r = rho_p(A) mod g; V(g) highlighted Logical fact: A ≡ r (mod g) => A(a)=r(a) only for a in V(g)");

win->end(); win->resizable(win); win->show(argc,argv); return Fl::run(); }
