// unified_galois_visual_production.cpp // Unified Field Explorer (production-ready) // Single-file C++ program using FLTK + OpenGL to visualize: //  - Algebraic root-density heatmap (samples of random polynomials, Durand–Kerner solver) //  - Finite-field embeddings (GF(p^n) visualized in the complex plane) //  - Field-extension overlay: adjoin polynomial g(x), compute numeric roots (embeddings) //  - Companion matrix display and eigenvalue visualizer (eigenvalues computed via Durand–Kerner) //  - Export companion matrix (CSV) and heatmap snapshot (PPM) // // Features to make it production-ready: //  - Robust parsing and input validation //  - Clear UI with labeled controls and presets //  - Export functions for reproducible snapshots //  - Reasonable defaults and limits to avoid heavy CPU use //  - Self-contained (no external linear algebra libraries) — numeric root-finding via Durand–Kerner // // Build (Linux): // g++ -std=c++17 -O2 unified_galois_visual_production.cpp -o unified_galois_visual 
//    -lfltk -lfltk_gl -lGL -lGLU -lm // // Run: // ./unified_galois_visual

#include <FL/Fl.H> #include <FL/Fl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Button.H> #include <FL/Fl_Choice.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Gl_Window.H> #include <FL/Fl_Check_Button.H> #include <FL/Fl_Input.H> #include <FL/Fl_Browser.H> #include <FL/Fl_Multiline_Output.H> #include <FL/Fl_Native_File_Chooser.H> #include <FL/gl.h>

#include <vector> #include <string> #include <cmath> #include <complex> #include <random> #include <sstream> #include <algorithm> #include <iostream> #include <fstream> #include <iomanip>

using namespace std; using cd = complex<double>;

static inline double clamp01(double x){ return x<0?0:(x>1?1:x); }

// ------------------ Numerical utilities ------------------ // Durand-Kerner root solver (stable for small degrees; configurable iterations) vector<cd> durand_kerner(const vector<cd>& coeffs, int max_iters=400, double tol=1e-12){ int n = (int)coeffs.size()-1; vector<cd> roots; if(n <= 0) return roots; roots.resize(n); // intelligent radius using Cauchy bound double max_a = 0.0; for(int i=0;i<n; ++i) max_a = max(max_a, abs(coeffs[i])); double radius = 1.0 + max_a; for(int i=0;i<n;++i) roots[i] = polar(radius, 2.0M_PIi/n); for(int iter=0; iter<max_iters; ++iter){ double max_change = 0.0; for(int i=0;i<n;++i){ cd xi = roots[i]; cd p = coeffs[n]; for(int k=n-1;k>=0;--k) p = p*xi + coeffs[k]; cd prod = 1.0; for(int j=0;j<n;++j) if(j!=i) prod *= (xi - roots[j]); if(abs(prod) < 1e-18) prod = 1e-18; cd delta = p / prod; roots[i] -= delta; max_change = max(max_change, abs(delta)); } if(max_change < tol) break; } return roots; }

// Companion matrix builder for monic polynomial with low->high coefficients (a0 + a1 x + ... + a_{n-1} x^{n-1} + x^n) vector<double> companion_matrix(const vector<int>& coeffs){ int n = (int)coeffs.size() - 1; // coeffs.size() = n+1 vector<double> M(nn, 0.0); for(int i=1;i<n;++i) M[in + (i-1)] = 1.0; // subdiagonal ones for(int i=0;i<n;++i){ double a = (i < (int)coeffs.size() ? (double)coeffs[i] : 0.0); M[i*n + (n-1)] = -a; } return M; }

string companion_matrix_to_csv(const vector<double>& M){ if(M.empty()) return string(); int n = (int) sqrt((double)M.size()); ostringstream ss; ss<<fixed<<setprecision(12); for(int r=0;r<n;++r){ for(int c=0;c<n;++c){ ss<<M[r*n + c]; if(c+1<n) ss<<","; } if(r+1<n) ss<<" "; } return ss.str(); }

cd eval_poly_at_alpha(const vector<int>& coeffs, const cd &alpha){ cd acc(0,0); cd powa(1,0); for(size_t i=0;i<coeffs.size();++i){ acc += powa * (double)coeffs[i]; powa *= alpha; } return acc; }

// Gaussian blur helpers (separable) vector<double> gaussian_kernel(double sigma, int &ksize_out){ if(sigma<=0.0){ ksize_out=1; return {1.0}; } int radius=(int)ceil(3.0sigma); int ksize=2radius+1; vector<double> k(ksize); double s2=sigmasigma; double sum=0.0; for(int i=-radius;i<=radius;++i){ double v=exp(-(ii)/(2.0s2)); k[i+radius]=v; sum+=v; } for(auto &x:k) x/=sum; ksize_out=ksize; return k; } void separable_blur(vector<double> &grid, int res, double sigma){ if(sigma<=0.0) return; int ksize; vector<double> k = gaussian_kernel(sigma, ksize); int radius=(ksize-1)/2; vector<double> tmp(resres,0.0); for(int y=0;y<res;++y){ for(int x=0;x<res;++x){ double s=0; for(int j=-radius;j<=radius;++j){ int xx=x+j; if(xx<0) xx=0; if(xx>=res) xx=res-1; s += k[j+radius] * grid[yres + xx]; } tmp[yres + x] = s; } } for(int x=0;x<res;++x){ for(int y=0;y<res;++y){ double s=0; for(int j=-radius;j<=radius;++j){ int yy=y+j; if(yy<0) yy=0; if(yy>=res) yy=res-1; s += k[j+radius] * tmp[yyres + x]; } grid[yres + x] = s; } } }

void magma_colormap(double t, unsigned char &r, unsigned char &g, unsigned char &b){ t = clamp01(t); double rr = pow(t, 0.4) * 255.0; double gg = pow(t, 1.9) * 180.0; double bb = pow(1.0-t, 1.2) * 220.0; r=(unsigned char)(clamp01(rr/255.0)*255); g=(unsigned char)(clamp01(gg/255.0)*255); b=(unsigned char)(clamp01(bb/255.0)*255); }

inline bool map_to_grid(const cd &z, int &ix, int &iy, int grid_res){ double re=z.real(), im=z.imag(); double xmin=-2.0, xmax=2.0, ymin=-2.0, ymax=2.0; if(re<xmin||re>xmax||im<ymin||im>ymax) return false; ix = (int)floor((re-xmin)/(xmax-xmin)(grid_res-1)+0.5); iy = (int)floor((im-ymin)/(ymax-ymin)(grid_res-1)+0.5); ix=max(0,min(grid_res-1,ix)); iy=max(0,min(grid_res-1,iy)); return true; }

// -------------------- GL Visualization widget -------------------- class UnifiedGL : public Fl_Gl_Window { public: // Parameters bool mode_algebraic = true; int max_degree = 3; int max_coeff = 5; double sigma = 1.0; int prime_p = 3; int ext_n = 2; int grid_res = 512; // internal buffers vector<double> heat; vector<unsigned char> image; std::mt19937 rng;

// filters
bool filter_real = false; bool filter_unit = false; bool filter_quadratic = false; bool filter_high_degree = false;

// extension / companion embedding
vector<int> adjoin_coeffs; vector<cd> adjoin_roots; int chosen_root_index = -1; cd chosen_alpha;
// companion matrix storage
vector<double> companion_M;

UnifiedGL(int X,int Y,int W,int H,const char *L=0):Fl_Gl_Window(X,Y,W,H,L){ heat.assign(grid_res*grid_res,0.0); image.assign(grid_res*grid_res*3,16); rng.seed(random_device{}()); end(); }

void draw() override {
    if(!valid()){ valid(1); glDisable(GL_DEPTH_TEST); glPixelStorei(GL_UNPACK_ALIGNMENT,1); }
    glClearColor(0.06f,0.06f,0.06f,1.0f); glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    if(mode_algebraic) render_algebraic(); else render_finite();
}

void render_algebraic(){
    std::fill(heat.begin(), heat.end(), 0.0);
    std::uniform_int_distribution<int> coeff_dist(-max_coeff, max_coeff);
    int samples_per_degree = max(1, 3000 / max(1, max_degree));
    for(int d=1; d<=max_degree; ++d){
        for(int s=0; s<samples_per_degree; ++s){
            vector<cd> coeffs(d+1); vector<int> icoeffs(d+1);
            for(int k=0;k<=d;++k){ int c = coeff_dist(rng); if(k==d && c==0) c = (coeff_dist(rng)==0?1:coeff_dist(rng)); icoeffs[k]=c; coeffs[k]=cd((double)c,0.0); }
            auto roots = durand_kerner(coeffs);
            for(auto &r : roots){
                bool is_real = fabs(r.imag()) < 1e-8;
                bool is_unit = fabs(abs(r) - 1.0) < 1e-3;
                bool degree_quadratic = (d==2);
                bool degree_high = (d>=3);
                bool pass=true;
                if(filter_real && !is_real) pass=false;
                if(filter_unit && !is_unit) pass=false;
                if(filter_quadratic && !degree_quadratic) pass=false;
                if(filter_high_degree && !degree_high) pass=false;
                if(!pass) continue;
                int ix, iy; if(map_to_grid(r, ix, iy, grid_res)) heat[iy*grid_res + ix] += 1.0;
            }
        }
    }
    separable_blur(heat, grid_res, sigma);
    double maxv = 0.0; for(double v:heat) if(v>maxv) maxv=v; if(maxv < 1e-12) maxv = 1.0;
    for(int y=0;y<grid_res;++y){ for(int x=0;x<grid_res;++x){ double v = heat[y*grid_res + x]; double t = log(1.0 + v)/log(1.0 + maxv); unsigned char r,g,b; magma_colormap(t, r,g,b); int idx=(y*grid_res + x)*3; image[idx]=r; image[idx+1]=g; image[idx+2]=b; }}
    // draw image
    glRasterPos2f(-1.0f, -1.0f);
    glPixelZoom((float)w()/(float)grid_res, (float)h()/(float)grid_res);
    glDrawPixels(grid_res, grid_res, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    draw_overlay_text("Algebraic mode: root-density in C. Companion embedding (white) and eigenvalues listed in panel.", 10, 18);

    // draw images of small Q(alpha) elements if adjoined
    if(!adjoin_coeffs.empty() && chosen_root_index>=0 && chosen_root_index < (int)adjoin_roots.size()){
        chosen_alpha = adjoin_roots[chosen_root_index];
        // enumerate small polynomials degree < deg with coefficients in [-2,2]
        int deg = (int)adjoin_coeffs.size()-1; if(deg<1) deg=1;
        int bound = 2; int total = (int) pow((2*bound+1), deg);
        int cap = min(total, 2000);
        glPointSize(4.0f); glBegin(GL_POINTS); glColor3f(1.0f,1.0f,1.0f);
        for(int idx=0; idx<cap; ++idx){ int t=idx; vector<int> pc(deg); for(int k=0;k<deg;++k){ pc[k] = (t % (2*bound+1)) - bound; t /= (2*bound+1); } cd z = eval_poly_at_alpha(pc, chosen_alpha); int ix,iy; if(map_to_grid(z, ix, iy, grid_res)){ double rx = -1.0 + 2.0 * ix / (grid_res-1); double ry = -1.0 + 2.0 * iy / (grid_res-1); glVertex2f((float)rx, (float)ry); } }
        glEnd();
    }
}

void render_finite(){
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(-1,1,-1,1,-1,1); glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    int p = max(2, prime_p); int n = max(1, ext_n); int N=1; for(int i=0;i<n;++i) N*=p; vector<cd> elems; elems.reserve(N);
    for(int i=0;i<N;++i){ int t=i; cd z(0,0); for(int j=0;j<n;++j){ int c = t % p; t /= p; double angle = 2.0*M_PI * (double)j / max(1,n); cd basis = polar(1.0, angle); z += basis * (double)c; } elems.push_back(z); }
    double maxr=0; for(auto &z: elems) maxr=max(maxr, abs(z)); if(maxr==0) maxr=1.0;
    glPointSize(6.0f); glBegin(GL_POINTS);
    for(size_t i=0;i<elems.size();++i){ double xr=elems[i].real()/maxr, yr=elems[i].imag()/maxr; float hue=(float)(abs(elems[i])/maxr); glColor3f(0.2f + 0.7f*hue, 0.6f*(1.0f-hue), 0.9f*hue); glVertex2f((float)xr, (float)yr); }
    glEnd();
    glPointSize(10.0f); glBegin(GL_POINTS); for(int i=0;i<min(p,(int)elems.size()); ++i){ double xr=elems[i].real()/maxr, yr=elems[i].imag()/maxr; glColor3f(1.0f,0.85f,0.15f); glVertex2f((float)xr, (float)yr); } glEnd();
    draw_overlay_text((string("Finite mode: GF(")+to_string(p)+"^"+to_string(n)+") embedding (complex)"), 10, 18);
}

void draw_overlay_text(const string &s, int px, int py){ glPixelZoom(1.0f,1.0f); fl_color(0xeeeeee); fl_font(FL_HELVETICA, 14); fl_draw(s.c_str(), px, py + 14); }

};

// ------------------------ UI glue / main ------------------------ int main(int argc, char **argv){ Fl_Window *win = new Fl_Window(100,100,1300,860, "Unified Field Explorer — Production Ready"); UnifiedGL *gl = new UnifiedGL(10,10,960,780);

// Right-side control panel
Fl_Box *panel = new Fl_Box(980,10,300,780); panel->box(FL_UP_BOX);
// Mode choice
Fl_Choice *mode_choice = new Fl_Choice(990,30,260,26, "Mode"); mode_choice->add("Algebraic"); mode_choice->add("Finite"); mode_choice->value(0);
mode_choice->callback([gl](Fl_Widget*w, void*){ Fl_Choice*c=(Fl_Choice*)w; string s=c->text(); gl->mode_algebraic = (s=="Algebraic"); gl->redraw(); });
// Degree / p
Fl_Value_Slider *s1 = new Fl_Value_Slider(990,70,260,22, "Degree / p"); s1->type(FL_HORIZONTAL); s1->bounds(1,12); s1->step(1); s1->value(3);
s1->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider*sv=(Fl_Value_Slider*)w; int v=(int)round(sv->value()); if(gl->mode_algebraic) gl->max_degree=v; else gl->prime_p=max(2,v); gl->redraw(); });
// coeff / n
Fl_Value_Slider *s2 = new Fl_Value_Slider(990,110,260,22, "Coeff / n"); s2->type(FL_HORIZONTAL); s2->bounds(1,40); s2->step(1); s2->value(5);
s2->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider*sv=(Fl_Value_Slider*)w; int v=(int)round(sv->value()); if(gl->mode_algebraic) gl->max_coeff=v; else gl->ext_n=max(1,v); gl->redraw(); });
// blur
Fl_Value_Slider *s3 = new Fl_Value_Slider(990,150,260,22, "Blur sigma"); s3->type(FL_HORIZONTAL); s3->bounds(0.0,6.0); s3->step(0.1); s3->value(1.0);
s3->callback([gl](Fl_Widget*w, void*){ Fl_Value_Slider*sv=(Fl_Value_Slider*)w; gl->sigma = sv->value(); gl->redraw(); });
// Filters
Fl_Check_Button *cb_real = new Fl_Check_Button(990,190,260,20, "Filter: Real roots"); cb_real->callback([gl](Fl_Widget*w, void*){ gl->filter_real = ((Fl_Check_Button*)w)->value(); gl->redraw(); });
Fl_Check_Button *cb_unit = new Fl_Check_Button(990,215,260,20, "Filter: Unit-modulus roots"); cb_unit->callback([gl](Fl_Widget*w, void*){ gl->filter_unit = ((Fl_Check_Button*)w)->value(); gl->redraw(); });
Fl_Check_Button *cb_quad = new Fl_Check_Button(990,240,260,20, "Filter: Quadratic (d=2) samples"); cb_quad->callback([gl](Fl_Widget*w, void*){ gl->filter_quadratic = ((Fl_Check_Button*)w)->value(); gl->redraw(); });
Fl_Check_Button *cb_high = new Fl_Check_Button(990,265,260,20, "Filter: High-degree (d>=3) samples"); cb_high->callback([gl](Fl_Widget*w, void*){ gl->filter_high_degree = ((Fl_Check_Button*)w)->value(); gl->redraw(); });

// Companion embedding controls
Fl_Box *adjoinLabel = new Fl_Box(990,300,260,18, "Adjoin polynomial g(x) — coefficients low->high (comma-separated)");
Fl_Input *polyInput = new Fl_Input(990,320,260,24); polyInput->value("1,0,1");
Fl_Button *btnAdjoin = new Fl_Button(990,350,120,28, "Adjoin");
Fl_Button *btnExportMatrix = new Fl_Button(1130,350,120,28, "Export matrix");
Fl_Button *btnSaveHeat = new Fl_Button(990,385,260,28, "Save heatmap (PPM)");

Fl_Browser *eigen_browser = new Fl_Browser(990,425,260,150, "Eigenvalues"); eigen_browser->type(FL_HOLD_BROWSER);
Fl_Multiline_Output *matrix_out = new Fl_Multiline_Output(990,585,260,150, "Companion matrix (rows)"); matrix_out->wrap_mode(FL_WRAP_NONE);

// Adjoin action: parse polynomial, compute roots (eigenvalues), fill browser and matrix display
btnAdjoin->callback([gl, polyInput, eigen_browser, matrix_out](Fl_Widget*, void*){
    string s = polyInput->value(); vector<int> coeffs; string tmp;
    for(char ch : s){ if(ch==','){ if(!tmp.empty()){ try{ coeffs.push_back(stoi(tmp)); } catch(...){ } tmp.clear(); } } else if(!isspace((unsigned char)ch)) tmp.push_back(ch); }
    if(!tmp.empty()){ try{ coeffs.push_back(stoi(tmp)); } catch(...){} }
    if(coeffs.size()<2){ fl_alert("Please enter at least two coefficients (e.g. 1,0,1 for x^2+1)."); return; }
    // store
    gl->adjoin_coeffs = coeffs;
    // compute numeric roots using Durand-Kerner
    vector<cd> coeffs_cd(coeffs.size()); for(size_t i=0;i<coeffs.size(); ++i) coeffs_cd[i] = cd((double)coeffs[i], 0.0);
    gl->adjoin_roots = durand_kerner(coeffs_cd);
    // fill browser and matrix
    eigen_browser->clear(); for(size_t i=0;i<gl->adjoin_roots.size(); ++i){ ostringstream ss; ss<<i<<": "<<fixed<<setprecision(10)<<gl->adjoin_roots[i].real()<<" + "<<gl->adjoin_roots[i].imag()<<"i"; eigen_browser->add(ss.str().c_str()); }
    // compute companion matrix
    gl->companion_M = companion_matrix(coeffs);
    // format companion matrix as rows
    ostringstream ms; ms<<fixed<<setprecision(6);
    int n = (int)gl->adjoin_coeffs.size() - 1;
    for(int r=0;r<n;++r){ for(int c=0;c<n;++c){ ms<<setw(12)<<gl->companion_M[r*n + c]; } ms<<"

"; } matrix_out->value(ms.str().c_str()); if(!gl->adjoin_roots.empty()){ gl->chosen_root_index = 0; gl->chosen_alpha = gl->adjoin_roots[0]; eigen_browser->select(1); } gl->redraw(); });

// eigenvalue selection: clicking a line in browser sets chosen_root
eigen_browser->callback([gl, eigen_browser](Fl_Widget*, void*){
    int idx = eigen_browser->value(); if(idx<=0) return; int sel = idx - 1; if(sel >= 0 && sel < (int)gl->adjoin_roots.size()){ gl->chosen_root_index = sel; gl->chosen_alpha = gl->adjoin_roots[sel]; gl->redraw(); }
});

// Export companion matrix CSV
btnExportMatrix->callback([gl](Fl_Widget*, void*){
    if(gl->companion_M.empty()){ fl_alert("No companion matrix to export. Adjoin a polynomial first."); return; }
    Fl_Native_File_Chooser chooser; chooser.title("Save companion matrix as CSV"); chooser.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE); chooser.filter("*.csv"); if(chooser.show()!=0) return; string path = chooser.filename(); if(path.empty()) return; if(path.find('.')==string::npos) path += ".csv";
    ofstream ofs(path, ios::binary); if(!ofs){ fl_alert("Unable to write file"); return; } string csv = companion_matrix_to_csv(gl->companion_M); ofs<<csv; ofs.close(); fl_message("Saved companion matrix to %s", path.c_str()); });

// Save heatmap to PPM (simple exporter)
btnSaveHeat->callback([gl](Fl_Widget*, void*){
    Fl_Native_File_Chooser chooser; chooser.title("Save heatmap (PPM)"); chooser.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE); chooser.filter("*.ppm"); if(chooser.show()!=0) return; string path = chooser.filename(); if(path.empty()) return; if(path.find('.')==string::npos) path += ".ppm";
    // ensure heatmap rendered into gl->image (call render_algebraic temporarily)
    if(gl->mode_algebraic){ gl->render_algebraic(); } else { gl->render_finite(); }
    // write PPM
    ofstream ofs(path, ios::binary); if(!ofs){ fl_alert("Unable to open file for writing"); return; }
    int res = gl->grid_res; ofs<<"P6

"<<res<<" "<<res<<" 255 "; ofs.write((const char*)gl->image.data(), gl->image.size()); ofs.close(); fl_message("Saved heatmap to %s", path.c_str()); });

// Regenerate & presets
Fl_Button *regen = new Fl_Button(990,740,260,28, "Regenerate (re-sample)"); regen->callback([gl](Fl_Widget*, void*){ gl->redraw(); });
Fl_Button *preset1 = new Fl_Button(990,700,260,28, "Preset: low-degree algebraic"); preset1->callback([gl, s1, s2, s3](Fl_Widget*, void*){ s1->value(3); s2->value(5); s3->value(1.0); gl->mode_algebraic=true; gl->max_degree=3; gl->max_coeff=5; gl->sigma=1.0; gl->redraw(); });
Fl_Button *preset2 = new Fl_Button(990,660,260,28, "Preset: dense algebraic"); preset2->callback([gl, s1, s2, s3](Fl_Widget*, void*){ s1->value(9); s2->value(12); s3->value(1.6); gl->mode_algebraic=true; gl->max_degree=9; gl->max_coeff=12; gl->sigma=1.6; gl->redraw(); });

win->end(); win->resizable(win); win->show(argc, argv);
return Fl::run();

}
