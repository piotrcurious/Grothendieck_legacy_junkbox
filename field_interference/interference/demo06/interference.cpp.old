// unified_galois_visual.cpp
// Unified Field Explorer (Production-Ready)
//
// Description:
//   A high-performance C++ tool using FLTK and OpenGL to visualize:
//   1. Algebraic Numbers: Root density heatmaps of random polynomials (Littlewood polynomials).
//   2. Finite Fields: Visual representations of GF(p^n) extensions.
//   3. Field Extensions: Adjoining specific roots via companion matrices.
//
// Build Instructions (Linux):
//   g++ -std=c++17 -O3 unified_galois_visual.cpp -o unified_galois_visual \
//       -lfltk -lfltk_gl -lGL -lGLU -lm
//
// Usage:
//   ./unified_galois_visual

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
#include <FL/Fl_Native_File_Chooser.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <vector>
#include <string>
#include <cmath>
#include <complex>
#include <random>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <atomic>

using namespace std;
using cd = complex<double>;

// Constants
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ------------------ Mathematical Utilities ------------------ //

static inline double clamp01(double x) { 
    return x < 0 ? 0 : (x > 1 ? 1 : x); 
}

// Durand-Kerner method for finding all roots of a polynomial simultaneously.
// Coeffs are ordered: a_0 + a_1*x + ... + a_n*x^n
vector<cd> durand_kerner(const vector<cd>& input_coeffs, int max_iters = 400, double tol = 1e-12) {
    // Strip trailing zeros to find actual degree
    vector<cd> coeffs = input_coeffs;
    while (coeffs.size() > 1 && abs(coeffs.back()) < 1e-9) {
        coeffs.pop_back();
    }
    
    int n = (int)coeffs.size() - 1;
    vector<cd> roots;
    if (n <= 0) return roots;
    roots.resize(n);

    // Normalize coefficients so leading term is 1
    cd leading = coeffs[n];
    for (auto& c : coeffs) c /= leading;

    // Cauchy bound for initial radius
    double max_a = 0.0;
    for (int i = 0; i < n; ++i) max_a = max(max_a, abs(coeffs[i]));
    double radius = 1.0 + max_a;

    // Initialize roots on a circle with slight offset to avoid symmetry locks
    for (int i = 0; i < n; ++i) {
        roots[i] = polar(radius, 2.0 * M_PI * i / n + 0.1); 
    }

    for (int iter = 0; iter < max_iters; ++iter) {
        double max_change = 0.0;
        for (int i = 0; i < n; ++i) {
            cd xi = roots[i];
            
            // Horner's method for P(xi)
            cd p_val = coeffs[n]; 
            for (int k = n - 1; k >= 0; --k) p_val = p_val * xi + coeffs[k];

            cd prod = 1.0;
            for (int j = 0; j < n; ++j) {
                if (i != j) prod *= (xi - roots[j]);
            }

            if (abs(prod) < 1e-18) prod = 1e-18; // Prevent div by zero
            
            cd delta = p_val / prod;
            roots[i] -= delta;
            max_change = max(max_change, abs(delta));
        }
        if (max_change < tol) break;
    }
    return roots;
}

// Companion matrix for monic polynomial a_0 + ... + x^n
// Returns row-major flat vector
vector<double> companion_matrix(const vector<int>& coeffs) {
    int n = (int)coeffs.size() - 1; // Degree
    if (n < 1) return {};
    
    // Check if monic, if not, we effectively normalize (though integer math makes this tricky, 
    // we assume the user provides a monic or we treat it as characteristic poly logic)
    // Here we strictly follow the definition where the last column is -a_i
    
    vector<double> M(n * n, 0.0);
    
    // Subdiagonal 1s
    for (int i = 1; i < n; ++i) {
        M[i * n + (i - 1)] = 1.0;
    }
    
    // Last column: -a_0, -a_1 ... -a_{n-1}
    for (int i = 0; i < n; ++i) {
        double val = (i < (int)coeffs.size()) ? (double)coeffs[i] : 0.0;
        // In companion matrix for x^n + a_{n-1}x^{n-1} + ... + a_0,
        // The last column is typically -a_0, -a_1, ..., -a_{n-1}
        M[i * n + (n - 1)] = -val;
    }
    return M;
}

string companion_matrix_to_csv(const vector<double>& M) {
    if (M.empty()) return "";
    int n = (int)sqrt((double)M.size());
    ostringstream ss;
    ss << fixed << setprecision(12);
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            ss << M[r * n + c];
            if (c + 1 < n) ss << ",";
        }
        ss << "\n";
    }
    return ss.str();
}

cd eval_poly_at_alpha(const vector<int>& coeffs, const cd& alpha) {
    cd acc(0, 0);
    cd powa(1, 0);
    for (int c : coeffs) {
        acc += powa * (double)c;
        powa *= alpha;
    }
    return acc;
}

// ------------------ Image Processing ------------------ //

vector<double> gaussian_kernel(double sigma, int& ksize_out) {
    if (sigma <= 0.0) {
        ksize_out = 1;
        return {1.0};
    }
    int radius = (int)ceil(3.0 * sigma);
    int ksize = 2 * radius + 1;
    vector<double> k(ksize);
    double s2 = 2.0 * sigma * sigma;
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double v = exp(-(i * i) / s2);
        k[i + radius] = v;
        sum += v;
    }
    for (auto& x : k) x /= sum;
    ksize_out = ksize;
    return k;
}

void separable_blur(vector<double>& grid, int res, double sigma) {
    if (sigma <= 0.0) return;
    int ksize;
    vector<double> k = gaussian_kernel(sigma, ksize);
    int radius = (ksize - 1) / 2;
    vector<double> tmp(res * res, 0.0);

    // Horizontal pass
    for (int y = 0; y < res; ++y) {
        for (int x = 0; x < res; ++x) {
            double s = 0;
            for (int j = -radius; j <= radius; ++j) {
                int xx = clamp(x + j, 0, res - 1);
                s += k[j + radius] * grid[y * res + xx];
            }
            tmp[y * res + x] = s;
        }
    }

    // Vertical pass
    for (int x = 0; x < res; ++x) {
        for (int y = 0; y < res; ++y) {
            double s = 0;
            for (int j = -radius; j <= radius; ++j) {
                int yy = clamp(y + j, 0, res - 1);
                s += k[j + radius] * tmp[yy * res + x];
            }
            grid[y * res + x] = s;
        }
    }
}

void magma_colormap(double t, unsigned char& r, unsigned char& g, unsigned char& b) {
    t = clamp01(t);
    // Approximate Magma-like gradient
    double rr = pow(t, 0.4) * 255.0;
    double gg = pow(t, 1.9) * 180.0;
    double bb = pow(1.0 - t, 1.2) * 220.0;
    
    // Boost contrast for visibility
    if (t > 0.95) { rr = 255; gg = 255; bb = 220; } // White/Yellow hot tip
    
    r = (unsigned char)(clamp01(rr / 255.0) * 255);
    g = (unsigned char)(clamp01(gg / 255.0) * 255);
    b = (unsigned char)(clamp01(bb / 255.0) * 255);
}

// ------------------ Visualization Widget ------------------ //

class UnifiedGL : public Fl_Gl_Window {
public:
    // Core Parameters
    bool mode_algebraic = true;
    int max_degree = 3;
    int max_coeff = 5;
    double sigma = 1.0;
    int prime_p = 3;
    int ext_n = 2;
    
    // Internal resolution
    const int grid_res = 512;
    
    // Data Buffers
    vector<double> heat;
    vector<unsigned char> image;
    std::mt19937 rng;
    bool dirty_compute = true; // Flag to trigger re-computation

    // Filters
    bool filter_real = false;
    bool filter_unit = false;
    bool filter_quadratic = false;
    bool filter_high_degree = false;

    // Field Extension State
    vector<int> adjoin_coeffs; // e.g., 1, 0, 1 for x^2 + 1
    vector<cd> adjoin_roots;
    int chosen_root_index = -1;
    cd chosen_alpha;
    vector<double> companion_M;

    UnifiedGL(int X, int Y, int W, int H, const char* L = 0) 
        : Fl_Gl_Window(X, Y, W, H, L) {
        heat.resize(grid_res * grid_res);
        image.resize(grid_res * grid_res * 3);
        rng.seed(std::random_device{}());
        end();
    }

    // Helper to map complex plane to pixel grid [-2, 2]
    bool map_to_grid(const cd& z, int& ix, int& iy) {
        double re = z.real();
        double im = z.imag();
        double limit = 2.0;
        if (re < -limit || re > limit || im < -limit || im > limit) return false;
        
        // Map [-2, 2] to [0, grid_res]
        double norm_x = (re + limit) / (2.0 * limit);
        double norm_y = (im + limit) / (2.0 * limit);
        
        ix = (int)(norm_x * (grid_res - 1));
        iy = (int)(norm_y * (grid_res - 1));
        return true;
    }

    void compute_algebraic() {
        std::fill(heat.begin(), heat.end(), 0.0);
        std::uniform_int_distribution<int> coeff_dist(-max_coeff, max_coeff);
        
        // Adaptive sampling count based on complexity
        int samples = 50000;
        if (max_degree > 6) samples = 20000;
        
        for (int i = 0; i < samples; ++i) {
            // Pick a random degree [1, max_degree]
            std::uniform_int_distribution<int> deg_dist(1, max_degree);
            int d = deg_dist(rng);
            
            // Build polynomial
            vector<cd> coeffs(d + 1);
            for (int k = 0; k <= d; ++k) {
                int c = coeff_dist(rng);
                // Ensure monic-ish or at least non-zero leading term for full degree
                if (k == d && c == 0) c = 1; 
                coeffs[k] = cd((double)c, 0.0);
            }

            // Solve
            auto roots = durand_kerner(coeffs);

            // Filter and Accumulate
            for (const auto& r : roots) {
                bool is_real = abs(r.imag()) < 1e-4;
                bool is_unit = abs(abs(r) - 1.0) < 1e-2;
                bool is_quad = (d == 2);
                bool is_high = (d >= 3);

                bool pass = true;
                if (filter_real && !is_real) pass = false;
                if (filter_unit && !is_unit) pass = false;
                if (filter_quadratic && !is_quad) pass = false;
                if (filter_high_degree && !is_high) pass = false;

                if (pass) {
                    int ix, iy;
                    if (map_to_grid(r, ix, iy)) {
                        heat[iy * grid_res + ix] += 1.0;
                    }
                }
            }
        }

        // Post-process: Blur and Color Mapping
        separable_blur(heat, grid_res, sigma);
        
        double maxv = 0.0;
        for (double v : heat) if (v > maxv) maxv = v;
        if (maxv < 1e-9) maxv = 1.0;

        for (int i = 0; i < grid_res * grid_res; ++i) {
            double v = heat[i];
            // Logarithmic scaling for better dynamic range
            double t = log(1.0 + v * 10.0) / log(1.0 + maxv * 10.0);
            unsigned char r, g, b;
            magma_colormap(t, r, g, b);
            image[i * 3 + 0] = r;
            image[i * 3 + 1] = g;
            image[i * 3 + 2] = b;
        }
    }

    void draw() override {
        if (!valid()) {
            valid(1);
            glDisable(GL_DEPTH_TEST);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        }

        // Recompute if parameters changed
        if (dirty_compute && mode_algebraic) {
            compute_algebraic();
            dirty_compute = false;
        }

        glClearColor(0.06f, 0.06f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        if (mode_algebraic) {
            render_algebraic_view();
        } else {
            render_finite_view();
        }
        
        draw_overlays();
    }

    void render_algebraic_view() {
        // Draw the pre-computed heatmap
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        
        glRasterPos2f(-1.0f, -1.0f);
        glPixelZoom((float)w() / grid_res, (float)h() / grid_res);
        glDrawPixels(grid_res, grid_res, GL_RGB, GL_UNSIGNED_BYTE, image.data());
    }

    void render_finite_view() {
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        glOrtho(-2.5, 2.5, -2.5, 2.5, -1, 1); // Broader view for finite fields
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        int p = max(2, prime_p);
        int n = max(1, ext_n);
        int N = (int)pow(p, n);
        
        // Safety cap for rendering
        if (N > 100000) N = 100000; 

        vector<cd> elems;
        elems.reserve(N);

        // Generate elements of GF(p^n) visualized via basis
        // This is a visual embedding (Eisenstein-like integers), not canonical
        for (int i = 0; i < N; ++i) {
            int t = i;
            cd z(0, 0);
            for (int j = 0; j < n; ++j) {
                int coeff = t % p;
                t /= p;
                // Basis vectors spread on unit circle
                double angle = 2.0 * M_PI * j / n;
                cd basis = polar(1.0, angle);
                z += basis * (double)coeff;
            }
            elems.push_back(z);
        }

        // Auto-scale
        double maxr = 0;
        for (auto& z : elems) maxr = max(maxr, abs(z));
        if (maxr == 0) maxr = 1.0;
        
        // Draw points
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for (const auto& z : elems) {
            // Normalized color based on magnitude/angle
            float hue = (float)(arg(z) / (2.0*M_PI) + 0.5);
            float mag = (float)(abs(z) / maxr);
            glColor3f(0.2f + 0.8f * mag, 0.6f * hue, 1.0f - 0.5f * mag);
            glVertex2f((float)z.real(), (float)z.imag());
        }
        glEnd();
    }

    void draw_overlays() {
        // Reset coordinate system for overlays to [-1, 1] standard GL
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        glOrtho(-1, 1, -1, 1, -1, 1);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();

        // 1. Draw Axes
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        glVertex2f(-1.0f, 0.0f); glVertex2f(1.0f, 0.0f); // X-axis
        glVertex2f(0.0f, -1.0f); glVertex2f(0.0f, 1.0f); // Y-axis
        glEnd();

        // 2. Draw Unit Circle
        glBegin(GL_LINE_LOOP);
        glColor4f(1.0f, 1.0f, 1.0f, 0.2f);
        for(int i=0; i<64; ++i){
            double theta = 2.0 * M_PI * i / 64.0;
            // Since our view is [-2, 2], radius 1 is at 0.5 in Normalized Device Coords
            glVertex2f((float)(0.5 * cos(theta)), (float)(0.5 * sin(theta)));
        }
        glEnd();

        // 3. Draw Adjoined Root Extensions (if active)
        if (!adjoin_coeffs.empty() && chosen_root_index >= 0 && 
            chosen_root_index < (int)adjoin_roots.size()) {
            
            // Re-setup view to match data [-2,2]
            glMatrixMode(GL_PROJECTION); glLoadIdentity();
            glOrtho(-2.0, 2.0, -2.0, 2.0, -1, 1);

            // Draw the generated lattice Q(alpha)
            int deg = (int)adjoin_coeffs.size() - 1;
            if (deg < 1) deg = 1;
            
            int bound = 1; // Small coefficients [-1, 1]
            int total = (int)pow((2 * bound + 1), deg);
            int cap = min(total, 5000);

            glPointSize(4.0f);
            glBegin(GL_POINTS);
            glColor3f(0.2f, 1.0f, 0.2f); // Green for extension lattice
            
            for (int idx = 0; idx < cap; ++idx) {
                int t = idx;
                vector<int> pc(deg);
                for (int k = 0; k < deg; ++k) {
                    pc[k] = (t % (2 * bound + 1)) - bound;
                    t /= (2 * bound + 1);
                }
                cd z = eval_poly_at_alpha(pc, chosen_alpha);
                glVertex2f((float)z.real(), (float)z.imag());
            }
            glEnd();

            // Highlight the chosen alpha root
            glPointSize(8.0f);
            glBegin(GL_POINTS);
            glColor3f(1.0f, 0.0f, 0.0f); // Red
            glVertex2f((float)chosen_alpha.real(), (float)chosen_alpha.imag());
            glEnd();
        }

        // 4. Text Overlay (using FLTK over GL)
        glPixelZoom(1.0f, 1.0f); // Reset zoom
        // Note: Actual text drawing happens in standard FLTK draw() sequence *after* GL flush,
        // but here we use a helper triggered by the main loop or widget overlay.
        // For simplicity in this widget, we rely on the parent window descriptions.
    }

    void trigger_regen() {
        dirty_compute = true;
        redraw();
    }
};

// ------------------------ UI & Main ------------------------ //

int main(int argc, char** argv) {
    Fl_Window* win = new Fl_Window(1300, 860, "Unified Field Explorer");
    
    // Left: Visualization Area
    UnifiedGL* gl = new UnifiedGL(10, 10, 900, 840);
    gl->mode_algebraic = true;

    // Right: Control Panel
    Fl_Group* panel = new Fl_Group(920, 10, 370, 840);
    
    // --- Section 1: Visualization Mode ---
    Fl_Box* lbl_mode = new Fl_Box(920, 20, 370, 25, "--- Visualization Mode ---");
    lbl_mode->labelfont(FL_BOLD);
    
    Fl_Choice* mode_choice = new Fl_Choice(1020, 50, 200, 25, "Mode:");
    mode_choice->add("Algebraic (Root Density)");
    mode_choice->add("Finite Field Embedding");
    mode_choice->value(0);
    mode_choice->callback([](Fl_Widget* w, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        Fl_Choice* c = (Fl_Choice*)w;
        gl->mode_algebraic = (c->value() == 0);
        gl->trigger_regen();
    }, gl);

    // --- Section 2: Parameters ---
    Fl_Value_Slider* s1 = new Fl_Value_Slider(1020, 90, 260, 20, "Deg / P");
    s1->type(FL_HOR_NICE_SLIDER); s1->bounds(1, 15); s1->step(1); s1->value(5);
    s1->align(FL_ALIGN_LEFT);
    s1->callback([](Fl_Widget* w, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        int val = (int)((Fl_Value_Slider*)w)->value();
        gl->max_degree = val;
        gl->prime_p = max(2, val);
        gl->trigger_regen();
    }, gl);

    Fl_Value_Slider* s2 = new Fl_Value_Slider(1020, 120, 260, 20, "Coeff / N");
    s2->type(FL_HOR_NICE_SLIDER); s2->bounds(1, 20); s2->step(1); s2->value(5);
    s2->align(FL_ALIGN_LEFT);
    s2->callback([](Fl_Widget* w, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        int val = (int)((Fl_Value_Slider*)w)->value();
        gl->max_coeff = val;
        gl->ext_n = max(1, val);
        gl->trigger_regen();
    }, gl);

    Fl_Value_Slider* s3 = new Fl_Value_Slider(1020, 150, 260, 20, "Blur");
    s3->type(FL_HOR_NICE_SLIDER); s3->bounds(0.0, 5.0); s3->step(0.1); s3->value(1.0);
    s3->align(FL_ALIGN_LEFT);
    s3->callback([](Fl_Widget* w, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        gl->sigma = ((Fl_Value_Slider*)w)->value();
        gl->trigger_regen();
    }, gl);

    // Filters
    Fl_Check_Button* cb_real = new Fl_Check_Button(930, 180, 150, 20, "Only Real");
    cb_real->callback([](Fl_Widget* w, void* v) { ((UnifiedGL*)v)->filter_real = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->trigger_regen(); }, gl);
    
    Fl_Check_Button* cb_unit = new Fl_Check_Button(1100, 180, 150, 20, "Only Unit Mod");
    cb_unit->callback([](Fl_Widget* w, void* v) { ((UnifiedGL*)v)->filter_unit = ((Fl_Check_Button*)w)->value(); ((UnifiedGL*)v)->trigger_regen(); }, gl);

    Fl_Button* btn_regen = new Fl_Button(930, 220, 350, 30, "Regenerate Samples");
    btn_regen->callback([](Fl_Widget*, void* v) { ((UnifiedGL*)v)->trigger_regen(); }, gl);

    // --- Section 3: Field Extensions ---
    Fl_Box* lbl_ext = new Fl_Box(920, 270, 370, 25, "--- Companion & Extensions ---");
    lbl_ext->labelfont(FL_BOLD);

    Fl_Box* lbl_info = new Fl_Box(930, 295, 350, 40, "Enter poly coeff (low->high).\nEx: 1,0,1 for 1 + x^2");
    lbl_info->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE);
    
    Fl_Input* inp_poly = new Fl_Input(1000, 340, 280, 25, "Poly:");
    inp_poly->value("1,0,1");

    Fl_Browser* br_roots = new Fl_Browser(930, 420, 350, 120, "Roots (Eigenvalues)");
    br_roots->type(FL_HOLD_BROWSER);
    
    Fl_Multiline_Output* out_matrix = new Fl_Multiline_Output(930, 570, 350, 200, "Companion Matrix (M)");
    out_matrix->textfont(FL_COURIER);
    out_matrix->textsize(12);

    Fl_Button* btn_adjoin = new Fl_Button(930, 380, 170, 25, "Adjoin Poly");
    
    // Lambda for Adjoining Logic
// 1. Define a struct to hold the UI pointers we need inside the callback
    struct AdjoinContext {
        UnifiedGL* gl;
        Fl_Input* inp;
        Fl_Browser* list;
        Fl_Multiline_Output* mat;
    };

    // 2. Allocate this context on the heap (lives for the duration of the app)
    AdjoinContext* ctx = new AdjoinContext{gl, inp_poly, br_roots, out_matrix};

    // 3. Register the callback using a CAPTURELESS lambda, passing 'ctx' as user_data
    btn_adjoin->callback([](Fl_Widget*, void* v) {
        // Cast void* back to our context
        AdjoinContext* c = (AdjoinContext*)v;
        
        // --- Original Logic (Updated to use 'c->') ---
        string s = c->inp->value();
        vector<int> coeffs;
        string tmp;
        for (char ch : s) {
            if (ch == ',' || ch == ' ') {
                if (!tmp.empty()) { try { coeffs.push_back(stoi(tmp)); } catch (...) {} tmp.clear(); }
            } else {
                tmp.push_back(ch);
            }
        }
        if (!tmp.empty()) try { coeffs.push_back(stoi(tmp)); } catch (...) {}

        if (coeffs.size() < 2) {
            fl_alert("Polynomial degree must be at least 1.");
            return;
        }

        c->gl->adjoin_coeffs = coeffs;
        
        // Compute roots numerically
        vector<cd> c_cd;
        for (int val : coeffs) c_cd.push_back(cd((double)val, 0.0));
        c->gl->adjoin_roots = durand_kerner(c_cd);

        // Update Browser
        c->list->clear();
        for (size_t i = 0; i < c->gl->adjoin_roots.size(); ++i) {
            ostringstream ss;
            ss << i << ": " << fixed << setprecision(5) << c->gl->adjoin_roots[i].real() 
               << (c->gl->adjoin_roots[i].imag() >= 0 ? " + " : " - ") 
               << abs(c->gl->adjoin_roots[i].imag()) << "i";
            c->list->add(ss.str().c_str());
        }
        if (!c->gl->adjoin_roots.empty()) {
            c->list->select(1);
            c->gl->chosen_root_index = 0;
            c->gl->chosen_alpha = c->gl->adjoin_roots[0];
        }

        // Compute Matrix
        c->gl->companion_M = companion_matrix(coeffs);
        c->mat->value(companion_matrix_to_csv(c->gl->companion_M).c_str());
        
        c->gl->redraw();
        // --- End Logic ---

    }, ctx); // <--- Pass the context here as the second argument
    
    
    // Callback when selecting a root from the list
    br_roots->callback([](Fl_Widget* w, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        Fl_Browser* b = (Fl_Browser*)w;
        int sel = b->value();
        if (sel > 0 && sel <= (int)gl->adjoin_roots.size()) {
            gl->chosen_root_index = sel - 1;
            gl->chosen_alpha = gl->adjoin_roots[sel - 1];
            gl->redraw();
        }
    }, gl);

    Fl_Button* btn_export = new Fl_Button(1110, 380, 170, 25, "Export CSV");
    btn_export->callback([](Fl_Widget*, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        if (gl->companion_M.empty()) { fl_alert("No matrix to export."); return; }
        Fl_Native_File_Chooser fc;
        fc.title("Save Matrix CSV");
        fc.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE);
        fc.filter("*.csv");
        if (fc.show() == 0) {
            ofstream ofs(fc.filename());
            ofs << companion_matrix_to_csv(gl->companion_M);
        }
    }, gl);

    // --- Footer ---
    Fl_Button* btn_save_img = new Fl_Button(930, 800, 350, 30, "Save Heatmap (PPM)");
    btn_save_img->callback([](Fl_Widget*, void* v) {
        UnifiedGL* gl = (UnifiedGL*)v;
        Fl_Native_File_Chooser fc;
        fc.title("Save Heatmap");
        fc.type(Fl_Native_File_Chooser::BROWSE_SAVE_FILE);
        fc.filter("*.ppm");
        if (fc.show() == 0) {
            ofstream ofs(fc.filename(), ios::binary);
            ofs << "P6\n" << gl->grid_res << " " << gl->grid_res << "\n255\n";
            ofs.write((char*)gl->image.data(), gl->image.size());
        }
    }, gl);

    panel->end();

    win->end();
    win->resizable(gl);
    win->show(argc, argv);

    // Initial compute
    gl->trigger_regen();

    return Fl::run();
}
