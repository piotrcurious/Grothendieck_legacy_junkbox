// unified_galois_visual.cpp // Single-file FLTK + OpenGL demo inspired by the Python UnifiedFieldExplorer // - Two modes: Algebraic (plot density of algebraic roots in C) and Finite (visualize GF(p^n)) // - Interactive FLTK UI: sliders for degree, coefficient size, blur sigma; prime p and extension degree n // - Numeric Durand–Kerner root solver to find polynomial roots (complex approximations) // - Heatmap accumulation + separable Gaussian blur implemented in C++ and displayed via glDrawPixels // - Finite-field embedding visualized as complex points with optional lattice connections // // Build (Linux): // g++ -std=c++17 -O2 unified_galois_visual.cpp -o unified_galois_visual 
//    -lfltk -lfltk_gl -lGL -lGLU -lm // // Notes: // - The program uses only plain C++ types (int, double) for math; floating approximations are used //   for complex embeddings and root approximations. The visualization emphasizes where //   algebraic constructions force field-extensions, and how finite-field combinatorics differ.

#include <FL/Fl.H> #include <FL/Fl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Button.H> #include <FL/Fl_Choice.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Gl_Window.H> #include <FL/gl.h> #include <vector> #include <string> #include <cmath> #include <complex> #include <random> #include <sstream> #include <algorithm> #include <iostream>

using namespace std; using cd = complex<double>;

// ---------------- Math utilities ----------------- static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

// Durand-Kerner root solver (returns vector of complex roots) vector<cd> durand_kerner(const vector<cd>& coeffs, int max_iters=200, double tol=1e-12){ int n = (int)coeffs.size()-1; vector<cd> roots; if(n <= 0) return roots; roots.resize(n); // initial guess: roots of unity scaled double radius = 0.5 + 0.5 * pow(2.0, 1.0/n); for(int i=0;i<n;++i) roots[i] = polar(radius, 2.0M_PIi/n); for(int iter=0; iter<max_iters; ++iter){ double max_change = 0.0; for(int i=0;i<n;++i){ cd xi = roots[i]; // evaluate polynomial at xi using Horner, coeffs are a0..an cd p = coeffs[n]; for(int k=n-1;k>=0;--k) p = p*xi + coeffs[k]; // compute product of (xi - xj) j!=i cd prod = 1.0; for(int j=0;j<n;++j) if(j!=i) prod *= (xi - roots[j]); if(abs(prod) < 1e-18) prod = 1e-18; cd delta = p / prod; roots[i] -= delta; max_change = max(max_change, abs(delta)); } if(max_change < tol) break; } return roots; }

// Separable Gaussian blur kernel generator vector<double> gaussian_kernel(double sigma, int &ksize_out){ if(sigma <= 0.0) { ksize_out = 1; return {1.0}; } int radius = (int)ceil(3.0sigma); int ksize = 2radius + 1; vector<double> k(ksize); double s2 = sigmasigma; double sum = 0.0; for(int i=-radius;i<=radius;++i){ double v = exp(-(ii)/(2.0*s2)); k[i+radius] = v; sum += v; } for(auto &x : k) x /= sum; ksize_out = ksize; return k; }

// Apply separable Gaussian blur to a floating grid (in-place), grid is res x res void separable_blur(vector<double> &grid, int res, double sigma){ if(sigma <= 0.0) return; int ksize; vector<double> k = gaussian_kernel(sigma, ksize); int radius = (ksize-1)/2; // temp buffer vector<double> temp(resres, 0.0); // horizontal pass for(int y=0;y<res;++y){ for(int x=0;x<res;++x){ double s = 0.0; for(int j=-radius;j<=radius;++j){ int xx = x + j; if(xx<0) xx=0; if(xx>=res) xx=res-1; s += k[j+radius] * grid[yres + xx]; } temp[yres + x] = s; } } // vertical pass for(int x=0;x<res;++x){ for(int y=0;y<res;++y){ double s = 0.0; for(int j=-radius;j<=radius;++j){ int yy = y + j; if(yy<0) yy=0; if(yy>=res) yy=res-1; s += k[j+radius] * temp[yyres + x]; } grid[y*res + x] = s; } } }

// Colormap: simple magma-like approximation mapping [0,1] -> RGB void magma_colormap(double t, unsigned char &r, unsigned char &g, unsigned char &b){ t = clamp01(t); // warm -> cool mapping double rr = pow(t, 0.4) * 255.0; double gg = pow(t, 1.9) * 180.0; double bb = pow(1.0-t, 1.2) * 220.0; r = (unsigned char)clamp01(rr/255.0)*255; g = (unsigned char)clamp01(gg/255.0)*255; b = (unsigned char)clamp01(bb/255.0)*255; }

// ---------------- GL Visual widget ----------------- class UnifiedGL : public Fl_Gl_Window { public: // parameters (editable from UI) bool mode_algebraic = true; int max_degree = 3; int max_coeff = 5; double sigma = 1.0; int prime_p = 3; int ext_n = 2;

// rendering grid for algebraic heatmap
int grid_res = 512;
vector<double> heat; // doubles res*res
vector<unsigned char> image; // RGB image res*res*3

// RNG
std::mt19937 rng;
UnifiedGL(int X,int Y,int W,int H,const char *L=0):Fl_Gl_Window(X,Y,W,H,L){
    heat.assign(grid_res*grid_res, 0.0);
    image.assign(grid_res*grid_res*3, 16);
    rng.seed(1337);
    end();
}

void draw() override {
    if(!valid()){
        valid(1);
        glDisable(GL_DEPTH_TEST);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }
    glClearColor(0.06f,0.06f,0.06f,1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(mode_algebraic) render_algebraic(); else render_finite();
}

// Map complex plane [-2,2]x[-2,2] to grid index
inline bool map_to_grid(const cd &z, int &ix, int &iy){
    double re = z.real(), im = z.imag();
    double xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0;
    if(re < xmin || re > xmax || im < ymin || im > ymax) return false;
    ix = (int)floor((re - xmin) / (xmax - xmin) * (grid_res-1) + 0.5);
    iy = (int)floor((im - ymin) / (ymax - ymin) * (grid_res-1) + 0.5);
    if(ix<0) ix=0; if(ix>=grid_res) ix=grid_res-1; if(iy<0) iy=0; if(iy>=grid_res) iy=grid_res-1;
    return true;
}

void render_algebraic(){
    // reset heat
    std::fill(heat.begin(), heat.end(), 0.0);
    // sample polynomials randomly with degrees 1..max_degree
    std::uniform_int_distribution<int> coeff_dist(-max_coeff, max_coeff);
    int samples_per_degree = 4000 / max(1,max_degree);
    for(int d=1; d<=max_degree; ++d){
        for(int s=0; s<samples_per_degree; ++s){
            // construct polynomial coefficients a0..ad (a_d must be nonzero)
            vector<cd> coeffs(d+1);
            for(int k=0;k<=d;++k){ int c = coeff_dist(rng); if(k==d && c==0) c= (coeff_dist(rng)==0?1:coeff_dist(rng)); coeffs[k] = cd((double)c, 0.0); }
            // Durand-Kerner expects a0..an and returns n roots
            auto roots = durand_kerner(coeffs);
            for(auto &r : roots){ int ix, iy; if(map_to_grid(r, ix, iy)) heat[iy*grid_res + ix] += 1.0; }
        }
    }
    // blur and normalize
    separable_blur(heat, grid_res, sigma);
    double maxv = 0.0; for(double v : heat) if(v>maxv) maxv = v; if(maxv < 1e-12) maxv = 1.0;
    // generate image using log scale
    for(int y=0;y<grid_res;++y){ for(int x=0;x<grid_res;++x){ double v = heat[y*grid_res + x]; double t = log(1.0 + v)/log(1.0 + maxv); unsigned char r,g,b; magma_colormap(t, r,g,b); int idx = (y*grid_res + x)*3; image[idx+0]=r; image[idx+1]=g; image[idx+2]=b; }}
    // draw as pixels
    glRasterPos2f(-1.0f, -1.0f);
    glPixelZoom((float)w() / (float)grid_res, (float)h()/(float)grid_res);
    glDrawPixels(grid_res, grid_res, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    // overlay textual info
    draw_overlay_text("Algebraic mode: root-density in C (samples of random polynomials). Log-scaled heatmap.", 10, 10);
}

void render_finite(){
    // clear background
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glClear(GL_COLOR_BUFFER_BIT);
    // compute elements of GF(p^n) as vectors of coefficients 0..p-1
    int p = prime_p; int n = ext_n;
    int N = 1; for(int i=0;i<n;++i) N *= p;
    vector<cd> elems; elems.reserve(N);
    for(int i=0;i<N;++i){ int t=i; cd z(0.0,0.0); for(int j=0;j<n;++j){ int c = t % p; t /= p; // map coefficient c to complex basis
            double angle = 2.0*M_PI * (double)j / max(1,n);
            cd basis = polar(1.0, angle);
            z += basis * (double)c; }
        elems.push_back(z);
    }
    // normalize positions for display
    double maxr = 0.0; for(auto &z : elems) maxr = max(maxr, abs(z)); if(maxr==0) maxr = 1.0;
    // draw connections for near neighbors to reveal lattice
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for(size_t i=0;i<elems.size();++i){ double xr = elems[i].real()/maxr, yr = elems[i].imag()/maxr; float hue = (float)(abs(elems[i]) / maxr); glColor3f(0.2f + 0.7f*hue, 0.6f*(1.0f-hue), 0.9f*hue); glVertex2f((float)xr, (float)yr); }
    glEnd();
    // draw base field elements (the first p elements) larger and highlighted
    glPointSize(10.0f); glBegin(GL_POINTS);
    for(int i=0;i<min(p,(int)elems.size()); ++i){ double xr = elems[i].real()/maxr, yr = elems[i].imag()/maxr; glColor3f(1.0f, 0.85f, 0.15f); glVertex2f((float)xr, (float)yr); }
    glEnd();
    // optionally draw lattice connections for distance ~1
    glLineWidth(1.0f); glBegin(GL_LINES);
    for(size_t i=0;i<elems.size();++i){ for(size_t j=i+1;j<elems.size();++j){ double d = abs(elems[i]-elems[j]); if(fabs(d - 1.0) < 0.15){ double xi = elems[i].real()/maxr, yi = elems[i].imag()/maxr; double xj = elems[j].real()/maxr, yj = elems[j].imag()/maxr; glColor4f(0.0f, 0.8f, 1.0f, 0.12f); glVertex2f((float)xi, (float)yi); glVertex2f((float)xj, (float)yj); } }}
    glEnd();
    draw_overlay_text((string("Finite mode: GF(")+to_string(p)+"^"+to_string(n)+") embedding (complex)"), 10, 10);
}

void draw_overlay_text(const string &s, int px, int py){
    glPixelZoom(1.0f,1.0f);
    fl_color(0xeeeeee);
    fl_font(FL_HELVETICA, 14);
    fl_draw(s.c_str(), px, py + 14);
}

};

// ---------------- UI glue ----------------- int main(int argc, char **argv){ Fl_Window *win = new Fl_Window(100,100,1200,800, "Unified Field Explorer (C++ FLTK + OpenGL)"); UnifiedGL *gl = new UnifiedGL(10,10,960,780);

// controls area
Fl_Box *info = new Fl_Box(980,10,200,780);
info->box(FL_UP_BOX);

// Mode choice
Fl_Choice *mode_choice = new Fl_Choice(990,30,180,24, "Mode");
mode_choice->add("Algebraic"); mode_choice->add("Finite"); mode_choice->value(0);
mode_choice->callback([gl](Fl_Widget*w, void*){
    Fl_Choice *c = (Fl_Choice*)w; const char *t = c->text(); if(!t) return; string s(t);
    gl->mode_algebraic = (s=="Algebraic"); gl->redraw(); });

// Degree / Prime slider
Fl_Value_Slider *s1 = new Fl_Value_Slider(990,80,180,22, "Degree / p"); s1->type(FL_HORIZONTAL); s1->bounds(1,11); s1->step(1); s1->value(3);
s1->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider *sv=(Fl_Value_Slider*)w; int v=(int)round(sv->value()); if(gl->mode_algebraic) gl->max_degree=v; else gl->prime_p = max(2,v); gl->redraw(); });

// Coeff / n slider
Fl_Value_Slider *s2 = new Fl_Value_Slider(990,120,180,22, "Coeff / n"); s2->type(FL_HORIZONTAL); s2->bounds(1,30); s2->step(1); s2->value(5);
s2->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider *sv=(Fl_Value_Slider*)w; int v=(int)round(sv->value()); if(gl->mode_algebraic) gl->max_coeff=v; else gl->ext_n = max(1,v); gl->redraw(); });

// Sigma slider for blur
Fl_Value_Slider *s3 = new Fl_Value_Slider(990,160,180,22, "Blur sigma"); s3->type(FL_HORIZONTAL); s3->bounds(0.0,6.0); s3->step(0.1); s3->value(1.0);
s3->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider *sv=(Fl_Value_Slider*)w; gl->sigma = sv->value(); gl->redraw(); });

// Regenerate button
Fl_Button *regen = new Fl_Button(1020, 220, 140, 30, "Regenerate");
regen->callback([gl](Fl_Widget*, void*){ gl->redraw(); });

// Quick presets
Fl_Button *preset1 = new Fl_Button(990,270,180,28, "Preset: low-degree algebraic");
preset1->callback([gl, s1, s2, s3](Fl_Widget*, void*){ gl->mode_algebraic=true; s1->value(3); s2->value(5); s3->value(1.0); gl->max_degree=3; gl->max_coeff=5; gl->sigma=1.0; gl->redraw(); });
Fl_Button *preset2 = new Fl_Button(990,305,180,28, "Preset: dense algebraic");
preset2->callback([gl, s1, s2, s3](Fl_Widget*, void*){ gl->mode_algebraic=true; s1->value(8); s2->value(12); s3->value(1.6); gl->max_degree=8; gl->max_coeff=12; gl->sigma=1.6; gl->redraw(); });
Fl_Button *preset3 = new Fl_Button(990,340,180,28, "Preset: GF(5^2)");
preset3->callback([gl, s1, s2, s3, mode_choice](Fl_Widget*, void*){ mode_choice->value(1); s1->value(5); s2->value(2); s3->value(0.5); gl->mode_algebraic=false; gl->prime_p=5; gl->ext_n=2; gl->redraw(); });

win->end(); win->resizable(win); win->show(argc, argv);
return Fl::run();

}
