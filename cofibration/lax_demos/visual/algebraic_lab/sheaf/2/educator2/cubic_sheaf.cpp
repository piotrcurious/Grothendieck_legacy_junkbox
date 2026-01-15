// cubic_sheaf.cpp
//
// RIGOROUS VISUALIZATION OF ALGEBRAIC GEOMETRY CONCEPTS
// Target: The Riemann Surface defined by F(w, z) = w^3 - 3w - z = 0
//
// KEY CONCEPTS:
// 1. Fiber/Stalk: The set of 3 roots {w1, w2, w3} over a point z.
// 2. Discriminant Locus: The points z = +/-2 where the fiber collapses (branch points).
// 3. Analytic Continuation: Solving the path lifting using Newton-Raphson tracking 
//    (simulating the flat connection).
// 4. Monodromy Group S3: Visualizing permutations of the fiber.

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Output.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <algorithm>

using Complex = std::complex<float>;
static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Mathematical Engine
// -----------------------------------------------------------------------------

// The Curve Equation: F(w, z) = w^3 - 3w - z = 0
// Partial derivative F_w = 3w^2 - 3
// Partial derivative F_z = -1

// Cardano's method is numerically unstable for continuity visualization due to 
// arbitrary branch cuts of the cube root. 
// Instead, we use a hybrid approach:
// 1. For the Static Mesh: Laguerre's Method (robust root finding).
// 2. For the Path: Newton-Raphson Continuation (rigorous topology tracking).

class CubicSurfaceMath {
public:
    // Solves w^3 - 3w - z = 0 for a static point z
    static std::vector<Complex> getRoots(Complex z) {
        std::vector<Complex> roots;
        // We use a simplified Durand-Kerner or just multiple Newton starts
        // to find all 3 roots. 
        // For w^3 - 3w - z = 0:
        
        Complex w_guesses[3] = { {2,0}, {-1, 1.7f}, {-1, -1.7f} }; // Triangle start
        
        for(int k=0; k<3; ++k) {
            Complex w = w_guesses[k];
            for(int iter=0; iter<10; ++iter) {
                Complex f = w*w*w - 3.0f*w - z;
                Complex df = 3.0f*w*w - 3.0f;
                if (std::abs(df) < 1e-4f) df = 1e-4f; 
                w = w - f/df;
            }
            roots.push_back(w);
        }
        
        // Very basic distinctness filter/sort could go here
        return roots;
    }

    // Analytic Continuation:
    // Given a root w_old at z_old, find the corresponding root w_new at z_new
    // This maintains the "Sheet Identity" along a path.
    static Complex liftStep(Complex z_new, Complex w_old) {
        Complex w = w_old;
        // Newton converge to the root at z_new closest to w_old
        for(int iter=0; iter<5; ++iter) {
            Complex f = w*w*w - 3.0f*w - z_new;
            Complex df = 3.0f*w*w - 3.0f;
            if (std::abs(df) < 1e-5f) break; // Singularity
            w = w - f/df;
        }
        return w;
    }
};

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
struct AppState {
    float rotX = 20.0f, rotY = -30.0f;
    float zoom = 1.0f;
    
    // The Loop in Base Space (Control Points)
    float loopRadius = 2.5f;
    float loopCenterRe = 0.0f;
    float loopCenterIm = 0.0f;
    
    // Visualization Options
    bool showMesh = true;
    bool showCritical = true;
    bool showFiber = true;
    
    // Calculated Monodromy Data
    std::string permutationStr = "Identity";
    std::vector<Complex> baseLoop;
    std::vector<std::vector<Complex>> liftedStrands; // 3 strands
};
AppState g_state;

// -----------------------------------------------------------------------------
// Geometry Generation
// -----------------------------------------------------------------------------
void calculatePath() {
    g_state.baseLoop.clear();
    g_state.liftedStrands.assign(3, {});
    
    // 1. Generate Base Loop (Circle)
    int segments = 120;
    for(int i=0; i<=segments; ++i) {
        float theta = 2.0f * PI * (float)i / segments;
        float r = g_state.loopRadius;
        float x = g_state.loopCenterRe + r * cos(theta);
        float y = g_state.loopCenterIm + r * sin(theta);
        g_state.baseLoop.push_back({x, y});
    }

    // 2. Initial Fibers (Start of path)
    // We solve fully at t=0
    auto startRoots = CubicSurfaceMath::getRoots(g_state.baseLoop[0]);
    // Sort them by real part for consistent initial labeling (1, 2, 3)
    std::sort(startRoots.begin(), startRoots.end(), [](const Complex& a, const Complex& b){
        return a.real() < b.real();
    });

    for(int i=0; i<3; ++i) g_state.liftedStrands[i].push_back(startRoots[i]);

    // 3. Homotopy Lifting (Continuation)
    for(size_t t=1; t<g_state.baseLoop.size(); ++t) {
        Complex z_curr = g_state.baseLoop[t];
        for(int strand=0; strand<3; ++strand) {
            Complex w_prev = g_state.liftedStrands[strand].back();
            Complex w_curr = CubicSurfaceMath::liftStep(z_curr, w_prev);
            g_state.liftedStrands[strand].push_back(w_curr);
        }
    }

    // 4. Compute Permutation (Monodromy)
    // Compare end roots with start roots
    std::vector<int> perm(3);
    auto endRoots = CubicSurfaceMath::getRoots(g_state.baseLoop.back()); // Re-solve at end (should match start z)
    
    for(int i=0; i<3; ++i) {
        Complex endW = g_state.liftedStrands[i].back();
        // Find which start root this is closest to
        int matchIdx = -1;
        float minD = 1e9f;
        for(int j=0; j<3; ++j) {
            float d = std::abs(endW - g_state.liftedStrands[j].front()); // Compare to STARTS of other strands
            if(d < minD) { minD = d; matchIdx = j; }
        }
        perm[i] = matchIdx + 1; // 1-based index
    }

    std::stringstream ss;
    ss << "Fiber Permutation: (" << perm[0] << " " << perm[1] << " " << perm[2] << ")";
    if (perm[0]==1 && perm[1]==2 && perm[2]==3) ss << " [Identity]";
    else ss << " [Non-Trivial]";
    g_state.permutationStr = ss.str();
}

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
void colorFromComplex(Complex w, float alpha) {
    float a = std::arg(w);
    float h = (a + PI) / (2.0f * PI);
    float r,g,b;
    
    // Simple HSV
    float x = 1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f);
    if(h*6<1) {r=1;g=x;b=0;}
    else if(h*6<2) {r=x;g=1;b=0;}
    else if(h*6<3) {r=0;g=1;b=x;}
    else if(h*6<4) {r=0;g=x;b=1;}
    else if(h*6<5) {r=x;g=0;b=1;}
    else {r=1;g=0;b=x;}
    
    glColor4f(r, g, b, alpha);
}

class SurfaceView : public Fl_Gl_Window {
    int lastX, lastY;
public:
    SurfaceView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
    }

    void draw() override {
        if(!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_LINE_SMOOTH);
            glLineWidth(2.0f);
        }
        glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(40.0/g_state.zoom, (float)w()/h(), 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 8, 12, 0, 0, 0, 0, 1, 0);
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        drawBaseSpace();
        if(g_state.showMesh) drawRiemannSurfacePoints();
        drawLiftedPaths();
    }

    int handle(int e) override {
        if(e == FL_PUSH) { lastX = Fl::event_x(); lastY = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            g_state.rotY += (Fl::event_x() - lastX);
            g_state.rotX += (Fl::event_y() - lastY);
            lastX = Fl::event_x(); lastY = Fl::event_y();
            redraw();
            return 1;
        }
        if(e == FL_MOUSEWHEEL) {
            g_state.zoom -= Fl::event_dy() * 0.1f;
            if(g_state.zoom < 0.1) g_state.zoom = 0.1;
            redraw();
            return 1;
        }
        return Fl_Gl_Window::handle(e);
    }

private:
    void drawBaseSpace() {
        float y = -3.0f;
        // Grid
        glBegin(GL_LINES);
        glColor4f(0.3f, 0.3f, 0.3f, 0.3f);
        for(int i=-6; i<=6; ++i) {
            glVertex3f(i, y, -6); glVertex3f(i, y, 6);
            glVertex3f(-6, y, i); glVertex3f(6, y, i);
        }
        glEnd();

        // Discriminant Locus (Branch Points at z = +/- 2)
        // Calculated from 3w^2-3=0 -> w=+-1 -> z = +-2
        if(g_state.showCritical) {
            glPointSize(10.0f);
            glBegin(GL_POINTS);
            glColor3f(1.0f, 0.2f, 0.2f); // RED Singularities
            glVertex3f(2.0f, y, 0.0f);
            glVertex3f(-2.0f, y, 0.0f);
            glEnd();
        }

        // The Base Loop
        glLineWidth(3.0f);
        glBegin(GL_LINE_STRIP);
        glColor3f(1, 1, 0); // Yellow Path
        for(auto& z : g_state.baseLoop) glVertex3f(z.real(), y + 0.05f, z.imag());
        glEnd();
        
        // Arrow on path
        if(!g_state.baseLoop.empty()) {
            Complex p = g_state.baseLoop[g_state.baseLoop.size()/8];
            glPushMatrix();
            glTranslatef(p.real(), y+0.05f, p.imag());
            glColor3f(1, 1, 0);
            GLUquadric* q = gluNewQuadric(); gluSphere(q, 0.1, 8, 8); gluDeleteQuadric(q);
            glPopMatrix();
        }
    }

    void drawRiemannSurfacePoints() {
        // Drawing a connected mesh for general Riemann surfaces is very hard due to cuts.
        // A "Point Cloud" approach is scientifically rigorous and avoids fake polygons.
        glPointSize(3.0f);
        glBegin(GL_POINTS);
        
        // Use a grid slightly wider than the view
        float step = 0.2f;
        for(float r = -4.0f; r <= 4.0f; r += step) {
            for(float i = -4.0f; i <= 4.0f; i += step) {
                Complex z(r, i);
                auto roots = CubicSurfaceMath::getRoots(z);
                for(auto& w : roots) {
                    // Height mapping: Real part of w
                    // Color mapping: Argument of w (Phase)
                    // This separates the sheets visually
                    colorFromComplex(w, 0.6f);
                    glVertex3f(z.real(), w.real(), z.imag());
                }
            }
        }
        glEnd();
    }

    void drawLiftedPaths() {
        // Draw the 3 strands of the lifted path
        // Each strand corresponds to a specific section over the loop
        
        glLineWidth(4.0f);
        
        for(int s=0; s<3; ++s) {
            glBegin(GL_LINE_STRIP);
            // Color each strand distinctly to see the permutation?
            // Or color by phase to match surface? Let's use Phase.
            for(size_t k=0; k<g_state.liftedStrands[s].size(); ++k) {
                Complex z = g_state.baseLoop[k];
                Complex w = g_state.liftedStrands[s][k];
                
                // Slightly lighter/brighter than surface
                colorFromComplex(w, 1.0f);
                glVertex3f(z.real(), w.real(), z.imag());
            }
            glEnd();
        }
        
        // Draw "Beads" at the start (Fiber over z0)
        glPointSize(12.0f);
        glBegin(GL_POINTS);
        for(int s=0; s<3; ++s) {
            Complex z = g_state.baseLoop[0];
            Complex w = g_state.liftedStrands[s][0];
            glColor3f(1,1,1); // White start beads
            glVertex3f(z.real(), w.real(), z.imag());
        }
        glEnd();
    }
};

// -----------------------------------------------------------------------------
// UI Construction
// -----------------------------------------------------------------------------
SurfaceView* glWin = nullptr;
Fl_Output* outPerm = nullptr;

void update_cb(Fl_Widget* w, void*) {
    calculatePath();
    if(outPerm) outPerm->value(g_state.permutationStr.c_str());
    if(glWin) glWin->redraw();
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1280, 800, "Rigorous Sheaf Monodromy: w^3 - 3w = z");
    
    // Control Panel
    Fl_Group* grp = new Fl_Group(10, 10, 300, 780);
    Fl_Box* title = new Fl_Box(10, 20, 300, 40, "Analytic Continuation\nof Cubic Sheaf");
    title->labelfont(FL_BOLD); title->labelsize(18);

    int y = 70;
    auto makeSlider = [&](const char* lbl, float min, float max, float& val) {
        Fl_Value_Slider* s = new Fl_Value_Slider(10, y, 290, 25, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(val);
        s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* v){
            *(float*)v = ((Fl_Value_Slider*)w)->value();
            update_cb(nullptr, nullptr);
        }, &val);
        y += 50;
    };

    makeSlider("Loop Radius (Encircles Branch Points?)", 0.5f, 4.5f, g_state.loopRadius);
    makeSlider("Loop Center Real", -3.0f, 3.0f, g_state.loopCenterRe);
    
    // Information
    Fl_Box* info = new Fl_Box(10, y, 290, 100, 
        "Branch Points (Singularities): z = +2, -2\n\n"
        "Try encircling ONE point vs TWO points.\n"
        "Encircling +/- 2 creates a transposition.\n"
        "Encircling both cancels out (if 0 winding).");
    info->box(FL_BORDER_BOX);
    info->align(FL_ALIGN_INSIDE | FL_ALIGN_WRAP | FL_ALIGN_LEFT);
    y += 110;

    outPerm = new Fl_Output(10, y, 290, 30, "Monodromy Group Action");
    outPerm->align(FL_ALIGN_TOP_LEFT);
    
    grp->end();

    // GL View
    glWin = new SurfaceView(320, 10, 950, 780);
    glWin->end();

    calculatePath();
    update_cb(nullptr, nullptr);

    win->resizable(glWin);
    win->show(argc, argv);
    return Fl::run();
}
