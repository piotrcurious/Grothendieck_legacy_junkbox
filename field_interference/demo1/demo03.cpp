// galois_grothendieck_fltk_demo.cpp // FLTK + OpenGL single-file demo (extended) // Now includes: //  - visualization of which operations produce irrational (algebraic/non-algebraic) numbers; //  - field-extension construction UI: adjoin sqrt(n) (quadratic) or root of arbitrary integer polynomial g(x); //  - companion matrix construction and numeric root approximation (Durand–Kerner) to provide concrete approximations for algebraic elements; //  - explicit "ring scope" diagram showing closure of operations: Z -> Q -> R -> C and finite-field / quotient-ring alternatives; //  - clear visual and textual mapping: operation -> required smallest ring/field (or extension) that contains the result. // // Design choices to stay within plain C types: integer coefficients (int), floating approximations (double), and simple linear algebra (vectors of double). // The Durand–Kerner solver provides complex approximations for algebraic roots; companion matrix is shown in textual form. // // Build (Linux): // g++ -std=c++17 -O2 galois_grothendieck_fltk_demo.cpp -o galois_demo 
//    -lfltk -lfltk_gl -lGL -lGLU -lm

#include <FL/Fl.H> #include <FL/Fl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Button.H> #include <FL/Fl_Choice.H> #include <FL/Fl_Slider.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Gl_Window.H> #include <FL/gl.h> #include <vector> #include <string> #include <cmath> #include <sstream> #include <algorithm> #include <complex> #include <random> #include <iostream>

using namespace std; using cd = complex<double>;

// --------------------------- Math helpers ----------------------------------

// Basic integer polynomial struct IntPoly { vector<int> coeffs; // coeffs[i]*x^i IntPoly(int deg=0){ coeffs.assign(max(1, deg+1),0); } int deg() const { for(int i=(int)coeffs.size()-1;i>=0;--i) if(coeffs[i]!=0) return i; return 0; } string to_string() const { ostringstream ss; for(size_t i=0;i<coeffs.size();++i) ss<<coeffs[i]<<(i+1<coeffs.size()?",":""); return ss.str(); } };

// Companion matrix as dense double matrix (row-major) for monic polynomial x^n + c_{n-1} x^{n-1} + ... + c0 vector<double> companion_matrix(const IntPoly &P){ int n = P.deg(); // Expect monic: ensure leading coeff is 1 after dividing if needed // We'll build companion for monic polynomial of degree n: x^n + a_{n-1} x^{n-1} + ... + a0 vector<int> c = P.coeffs; if((int)c.size()!=n+1){ /* normalize size / } vector<double> M(nn, 0.0); for(int i=1;i<n;i++) M[in + (i-1)] = 1.0; // subdiagonal ones // last column = -a0..-a_{n-1} for(int i=0;i<n;i++){ int idx = i; // coefficient a_i corresponds to x^i double a = 0.0; if(idx < (int)c.size()) a = (double)c[idx]; M[in + (n-1)] = -a; // note: assumes leading coeff is 1 } return M; }

// Durand-Kerner (Weierstrass) method to find complex roots of polynomial with double coefficients vector<cd> durand_kerner_roots(const vector<cd> &coeffs, int max_iters=200, double tol=1e-12){ // coeffs: a0 + a1 x + ... + an x^n (n>=1) int n = (int)coeffs.size() - 1; if(n<=0) return {}; // Initial guesses: roots of unity scaled vector<cd> roots(n); const double theta0 = 2.0M_PI / n; for(int i=0;i<n;i++) roots[i] = std::polar(1.0, theta0i) * 0.5; // radius 0.5 arbitrary for(int iter=0; iter<max_iters; ++iter){ double max_change = 0.0; for(int i=0;i<n;++i){ cd xi = roots[i]; // evaluate polynomial at xi cd p = coeffs[n]; for(int k=n-1;k>=0;--k) p = p*xi + coeffs[k]; // compute product (xi - xj) j!=i cd prod = 1.0; for(int j=0;j<n;++j) if(j!=i) prod *= (xi - roots[j]); if(abs(prod) < 1e-18) prod = 1e-18; // avoid divide by zero cd delta = p / prod; roots[i] -= delta; max_change = max(max_change, abs(delta)); } if(max_change < tol) break; } return roots; }

// Convert IntPoly to complex-coeff vector for Durand-Kerner vector<cd> poly_to_cd_coeffs(const IntPoly &P){ int n = P.deg(); vector<cd> c(n+1); for(int i=0;i<=n;++i) c[i] = cd((i<(int)P.coeffs.size()? P.coeffs[i] : 0), 0.0); return c; }

// Check if integer is perfect square bool is_perfect_square(int n){ if(n<0) return false; int r = (int)floor(sqrt((double)n)+0.5); return r*r==n; }

// Small utility: which minimal ring contains an operation result? // We'll classify as: Z, Q, R (includes square roots), C (complex roots), algebraic extension Q(alpha) with degree>1 string minimal_ring_for_operation(const string &op, const vector<int> &args, const IntPoly &g){ // op: "add", "mul", "div", "sqrt", "root_of_poly" if(op=="add"||op=="mul") return "Z (closed)"; // integer addition/multiplication stays in Z if(op=="div") { // integer division may leave Z: if denominator divides numerator then Z else Q if(args.size()>=2 && args[1]!=0 && (args[0] % args[1] == 0)) return "Z (still)"; else return "Q (rationals)"; } if(op=="sqrt"){ int a = args.size()? args[0]:0; if(a>=0 && is_perfect_square(a)) return "Z (perfect square)"; else return "R (quadratic/real extension)"; } if(op=="root_of_poly"){ int deg = g.deg(); if(deg==0) return "trivial"; if(deg==1) return "Q (root rational)"; // linear // higher degree: algebraic extension Q(alpha) ostringstream ss; ss << "Q(α) of degree " << deg; return ss.str(); } return "unknown"; }

// --------------------------- GL Visual Widget ------------------------------ class DemoGL : public Fl_Gl_Window { public: IntPoly *current_poly; // used for adjoined polynomial g(x) int a_int, b_int; // integer inputs int selected_prime; string last_operation; cd chosen_root_approx; // numeric approx for adjoined algebraic element vector<cd> computed_roots;

Fl_Box *info_box;
DemoGL(int X,int Y,int W,int H,const char*L=0):Fl_Gl_Window(X,Y,W,H,L){ current_poly=nullptr; a_int=2; b_int=3; selected_prime=7; last_operation=""; chosen_root_approx=cd(0,0); end(); }

void draw() override {
    if(!valid()) { valid(1); glEnable(GL_POINT_SMOOTH); glEnable(GL_LINE_SMOOTH); glHint(GL_LINE_SMOOTH_HINT, GL_NICEST); }
    glClearColor(0.1f,0.12f,0.14f,1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    int W=w(), H=h();

    // Left: Ring scope diagram (Z -> Q -> R -> C) and operation closure indicators
    glViewport(0, H/2, W/2, H/2);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,1,0,1,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_ring_scope();

    // Right-top: Operation panel results (textual)
    glViewport(W/2, H/2, W/2, H/2);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0,1,0,1,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_operation_info();

    // Bottom-left: Companion matrix & numeric roots visualization
    glViewport(0,0,W/2,H/2);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_companion_and_roots();

    // Bottom-right: Concrete numeric approximation vs symbolic label
    glViewport(W/2,0,W/2,H/2);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    draw_numeric_vs_symbolic();

    if(info_box){ ostringstream ss; ss<<"Inputs: a="<<a_int<<" b="<<b_int<<"  poly g(x)="<<(current_poly?current_poly->to_string():string("(none)"))<<"  last op="<<last_operation; info_box->label(ss.str().c_str()); }
}

void draw_ring_scope(){
    // Draw four boxes Z, Q, R, C and arrows; color them based on last operation required ring
    vector<string> rings = {"Z","Q","R","C"};
    vector<float> x = {0.1f, 0.35f, 0.6f, 0.85f};
    for(int i=0;i<4;++i){
        draw_rect(x[i]-0.08f, 0.55f, 0.16f, 0.3f, 0.12f, 0.12f, 0.12f);
        draw_text(x[i], 0.7f, rings[i]);
    }
    // arrows
    for(int i=0;i<3;++i){ draw_arrow(x[i]+0.06f, 0.7f, x[i+1]-0.06f, 0.7f); }

    // Determine minimal ring required
    string req = map_last_op_to_ring();
    draw_text(0.5f, 0.4f, string("Minimal ring/field required: ")+req);

    // Show operations closures: which ring closes under last operation? if last_operation empty, dim all
    for(int i=0;i<4;++i){ float cx=x[i]; float cy=0.3f; if(ring_contains(req, i)) draw_circle(cx, cy, 0.03f); else draw_circle_outline(cx, cy, 0.03f); draw_text(cx, cy-0.06f, string("closes?")); }

    // Simple legend
    draw_text(0.1f, 0.25f, "Operations: add, mul -> Z; div -> Q; sqrt -> R; polynomial root -> algebraic ext.");
}

void draw_operation_info(){
    draw_text(0.5f,0.85f, string("Last operation: ") + last_operation);
    // Show more info depending on op
    if(last_operation=="add"||last_operation=="mul") draw_text(0.5f,0.6f, "Result stays in Z (closed under +,*)");
    else if(last_operation=="div"){
        if(b_int!=0 && a_int % b_int == 0) draw_text(0.5f,0.6f, "Exact integer division -> stays in Z"); else draw_text(0.5f,0.6f, "Non-integral -> requires Q (rationals)");
    } else if(last_operation=="sqrt"){
        if(is_perfect_square(a_int)) draw_text(0.5f,0.6f, "Perfect square -> remains in Z"); else draw_text(0.5f,0.6f, "Not a perfect square -> adjoin sqrt -> quadratic extension (R)");
    } else if(last_operation=="root_of_poly"){
        if(current_poly){ ostringstream ss; ss<<"Adjoin root alpha of g(x) = "<<current_poly->to_string()<<" -> extension Q(alpha) degree="<<current_poly->deg(); draw_text(0.5f,0.6f, ss.str()); }
    }
    // show symbolic label
    draw_text(0.5f,0.4f, string("Symbolic representation: ")+symbolic_label());
}

void draw_companion_and_roots(){
    if(!current_poly){ draw_text(0.0f,0.8f,"No polynomial chosen"); return; }
    // draw companion matrix entries as small grid
    int n = current_poly->deg(); if(n<=0){ draw_text(0.0f,0.8f,"Polynomial degree <=0"); return; }
    vector<double> M = companion_matrix(*current_poly);
    // render grid centered
    float cx = 0.0f, cy = 0.5f; float size = 1.6f / n;
    for(int r=0;r<n;++r){ for(int c=0;c<n;++c){ double val = M[r*n+c]; float x = cx - 0.8f + c*size; float y = cy + 0.6f - r*size; draw_rect(x, y, size*0.9f, size*0.9f, 0.14f,0.14f,0.2f); ostringstream ss; ss<<fixed<<setprecision(2)<<val; draw_text(x+size*0.45f, y+size*0.45f, ss.str()); } }

    // compute numeric roots if not already
    if(computed_roots.empty()){
        vector<cd> coeffs = poly_to_cd_coeffs(*current_poly);
        computed_roots = durand_kerner_roots(coeffs);
        if(!computed_roots.empty()) chosen_root_approx = computed_roots[0];
    }
    // draw roots on complex plane
    draw_text(0.0f,-0.5f,"Numeric approximations of roots (complex plane)");
    for(size_t i=0;i<computed_roots.size();++i){ double rx = computed_roots[i].real(); double ry = computed_roots[i].imag(); draw_point_complex(rx, ry); draw_text(rx*0.2+0.4, ry*0.2+0.1, to_string(i)); }
}

void draw_numeric_vs_symbolic(){
    // Show symbolic label for adjoined element and numeric approx (chosen_root_approx)
    draw_text(-0.2f,0.6f, string("Symbolic: ")+symbolic_label());
    ostringstream ss; ss<<"Numeric approx: "<<fixed<<setprecision(8)<<chosen_root_approx.real(); if(abs(chosen_root_approx.imag())>1e-12) ss<<" + "<<chosen_root_approx.imag()<<"i";
    draw_text(-0.2f,0.4f, ss.str());
    // show difference between float/double operations vs symbolic
    draw_text(-0.2f,0.2f, "Note: numeric approximations are in R/C (floating) and may hide algebraic relations");
}

// Utilities for GL drawing primitives (text is simple placeholder using raster via Fl)
void draw_text(float ux, float uy, const string &txt){ // coords in 0..1 or -1..1 depending viewport
    glColor3f(0.95f,0.95f,0.85f);
    // Use FLTK draw for text to avoid GLUT dependency
    int vp[4]; glGetIntegerv(GL_VIEWPORT, vp);
    float sx = (ux+1.0f)/2.0f * vp[2]; float sy = (1.0f-uy)/2.0f * vp[3];
    glWindowPos2i((int)(sx), (int)(sy));
    fl_draw(txt.c_str(), (int)sx, (int)sy);
}
void draw_rect(float x, float y, float w, float h, float r, float g, float b){ glColor3f(r,g,b); glBegin(GL_QUADS); glVertex2f(x,y); glVertex2f(x+w,y); glVertex2f(x+w,y-h); glVertex2f(x,y-h); glEnd(); }
void draw_circle(float x, float y, float r){ glColor3f(0.2f,0.9f,0.2f); glBegin(GL_TRIANGLE_FAN); glVertex2f(x,y); for(int i=0;i<=24;i++){ float a = i*(2*M_PI/24); glVertex2f(x+cos(a)*r, y+sin(a)*r);} glEnd(); }
void draw_circle_outline(float x, float y, float r){ glColor3f(0.9f,0.9f,0.9f); glBegin(GL_LINE_LOOP); for(int i=0;i<32;i++){ float a=i*(2*M_PI/32); glVertex2f(x+cos(a)*r,y+sin(a)*r);} glEnd(); }
void draw_arrow(float x1,float y1,float x2,float y2){ glColor3f(0.9f,0.9f,0.5f); glBegin(GL_LINES); glVertex2f(x1,y1); glVertex2f(x2,y2); glEnd(); }
void draw_point_complex(double x,double y){ // scale small
    float sx = (float)(0.2*x); float sy = (float)(0.2*y); draw_circle(sx+0.4f, sy+0.0f, 0.02f); }

string map_last_op_to_ring(){ if(last_operation.empty()) return "(none)"; if(last_operation=="add"||last_operation=="mul") return "Z"; if(last_operation=="div") return (b_int!=0 && a_int % b_int == 0)? "Z" : "Q"; if(last_operation=="sqrt") return is_perfect_square(a_int)? "Z" : "R"; if(last_operation=="root_of_poly") return current_poly? string("Q(alpha) degree ")+to_string(current_poly->deg()) : "Q(alpha)"; return "?"; }
bool ring_contains(const string &req, int ring_index){ // ring_index: 0 Z,1 Q,2 R,3 C
    if(req=="(none)") return false; if(req=="Z") return ring_index==0; if(req=="Q") return ring_index>=1; if(req=="R") return ring_index>=2; if(req.find("Q(alpha)")!=string::npos) return false; return false; }
string symbolic_label(){ if(last_operation=="sqrt"){ if(is_perfect_square(a_int)) return to_string((int)floor(sqrt((double)a_int)+0.5)); else { ostringstream ss; ss<<"sqrt("<<a_int<<") (adjoin)"; return ss.str(); } } if(last_operation=="root_of_poly" && current_poly){ ostringstream ss; ss<<"alpha where g( alpha ) = 0 , deg="<<current_poly->deg(); return ss.str(); } if(last_operation=="div") { if(b_int!=0 && a_int%b_int==0) return to_string(a_int/b_int); else { ostringstream ss; ss<<a_int<<"/"<<b_int; return ss.str(); } } if(last_operation=="add") { ostringstream ss; ss<<a_int<<" + "<<b_int; return ss.str(); } if(last_operation=="mul") { ostringstream ss; ss<<a_int<<" * "<<b_int; return ss.str(); } return string("(none)"); }

};

// --------------------------- UI glue -------------------------------------- int main(int argc, char **argv){ Fl_Window *win = new Fl_Window(40,40,1300,800, "Field-extension & ring-scope visualization (Grothendieck-aware)"); DemoGL *gl = new DemoGL(10,60,1280,700);

Fl_Box *info = new Fl_Box(10,10,1280,40);
info->box(FL_FLAT_BOX); info->labelfont(FL_HELVETICA_BOLD); info->labelsize(12);
gl->info_box = info;

// integer inputs
Fl_Value_Slider *slA = new Fl_Value_Slider(10,760,180,28, "a"); slA->type(FL_HORIZONTAL); slA->bounds(-50,50); slA->step(1); slA->value(2);
slA->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider*s=(Fl_Value_Slider*)w; gl->a_int=(int)round(s->value()); gl->redraw(); });
Fl_Value_Slider *slB = new Fl_Value_Slider(200,760,180,28, "b"); slB->type(FL_HORIZONTAL); slB->bounds(-50,50); slB->step(1); slB->value(3);
slB->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider*s=(Fl_Value_Slider*)w; gl->b_int=(int)round(s->value()); gl->redraw(); });

// operation buttons
Fl_Button *btnAdd = new Fl_Button(400,760,80,28, "add"); btnAdd->callback([gl](Fl_Widget*,void*){ gl->last_operation = "add"; gl->computed_roots.clear(); gl->redraw(); });
Fl_Button *btnMul = new Fl_Button(490,760,80,28, "mul"); btnMul->callback([gl](Fl_Widget*,void*){ gl->last_operation = "mul"; gl->computed_roots.clear(); gl->redraw(); });
Fl_Button *btnDiv = new Fl_Button(580,760,80,28, "div"); btnDiv->callback([gl](Fl_Widget*,void*){ gl->last_operation = "div"; gl->computed_roots.clear(); gl->redraw(); });
Fl_Button *btnSqrt = new Fl_Button(670,760,100,28, "sqrt(a)"); btnSqrt->callback([gl](Fl_Widget*,void*){ gl->last_operation = "sqrt"; gl->computed_roots.clear(); gl->redraw(); });

// polynomial controls for field extension
Fl_Box *polyLabel = new Fl_Box(780,720,220,20, "Polynomial g(x) (coeffs comma-separated, low->high):");
Fl_Input *polyInput = new Fl_Input(780,740,220,28); polyInput->value("1,0,1"); // x^2 + 1
Fl_Button *btnAdjoin = new Fl_Button(1010,740,120,28, "Adjoin root");
btnAdjoin->callback([gl, polyInput](Fl_Widget*,void*){
    string s = polyInput->value(); // parse ints
    vector<int> coeffs; string tmp;
    for(char ch: s){ if(ch==','){ if(!tmp.empty()){ coeffs.push_back(stoi(tmp)); tmp.clear(); } } else if(!isspace(ch)) tmp.push_back(ch); }
    if(!tmp.empty()) coeffs.push_back(stoi(tmp));
    if(coeffs.empty()) return;
    gl->current_poly = new IntPoly((int)coeffs.size()-1);
    gl->current_poly->coeffs = coeffs;
    // compute numeric roots now
    vector<cd> coeffs_cd = poly_to_cd_coeffs(*gl->current_poly);
    gl->computed_roots = durand_kerner_roots(coeffs_cd);
    if(!gl->computed_roots.empty()) gl->chosen_root_approx = gl->computed_roots[0];
    gl->last_operation = "root_of_poly";
    gl->redraw();
});

// reset / clear
Fl_Button *btnClear = new Fl_Button(1140,740,80,28, "clear"); btnClear->callback([gl](Fl_Widget*,void*){ gl->last_operation.clear(); gl->current_poly=nullptr; gl->computed_roots.clear(); gl->redraw(); });

win->end(); win->resizable(win); win->show(argc,argv);
return Fl::run();

}
