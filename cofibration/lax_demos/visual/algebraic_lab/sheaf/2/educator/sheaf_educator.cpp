// sheaf_educator.cpp
//
// A Visualization of the Etale Space of the Sheaf of Holomorphic Functions
// for the multi-valued function w = sqrt(z - A).
//
// CONCEPTS DEMONSTRATED:
// 1. Base Space (X): The Complex Plane C (bottom grid).
// 2. Etale Space (E): The Riemann Surface (the colored mesh).
// 3. Projection (pi): The vertical mapping from E to X.
// 4. Stalk (Fx): The set of germs at a point x (the vertical fiber).
// 5. Germ: A specific point on the surface representing a function element.
// 6. Section: A continuous choice of germs over an open set U.
// 7. Monodromy: The permutation of the stalk when traversing a loop.

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Pack.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Output.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <complex>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

// -----------------------------------------------------------------------------
// Constants & Math Helpers
// -----------------------------------------------------------------------------
static constexpr float PI = 3.14159265359f;

struct Complex3 { float x, y, z; }; // 3D Coordinate for GL

// Hsv to Rgb for Domain Coloring
static void getDomainColor(const std::complex<float>& w, float alpha, float outCol[4]) {
    float angle = std::arg(w);
    float h = (angle + PI) / (2.0f * PI); // 0.0 to 1.0
    float s = 0.8f;
    float v = 0.5f + 0.5f * (1.0f - 1.0f / (1.0f + 2.0f * std::abs(w))); // Magnitude brightness

    // HSV to RGB
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float r = 0, g = 0, b = 0;

    float h6 = h * 6.0f;
    if      (h6 < 1.0f) { r = c; g = x; b = 0; }
    else if (h6 < 2.0f) { r = x; g = c; b = 0; }
    else if (h6 < 3.0f) { r = 0; g = c; b = x; }
    else if (h6 < 4.0f) { r = 0; g = x; b = c; }
    else if (h6 < 5.0f) { r = x; g = 0; b = c; }
    else                { r = c; g = 0; b = x; }

    outCol[0] = r + m; outCol[1] = g + m; outCol[2] = b + m; outCol[3] = alpha;
}

// -----------------------------------------------------------------------------
// Application State
// -----------------------------------------------------------------------------
struct AppState {
    // Parameters
    float branchRe = 0.0f;
    float branchIm = 0.0f;
    float germRadius = 0.5f;     // Size of the local section visualization
    float surfaceAlpha = 0.7f;
    
    // Animation / Interaction
    bool  isAnimating = false;
    float animTime = 0.0f;       // 0.0 to 1.0 along path
    float animSpeed = 0.01f;

    // The Path (in Base Space)
    std::vector<std::complex<float>> pathZ; 
    
    // The Lifted Path (in Etale Space - one for each sheet initially)
    // In a sheaf context, we track a specific germ moving.
    std::vector<std::complex<float>> liftedPath; 

    // Probe (The specific germ being examined)
    std::complex<float> probeZ = {1.5f, 0.5f};
    int activeSheet = 0; // Which sheet is the probe currently on?

    // View Options
    bool showSheets = true;
    bool showStalk = true;
    bool showGerm = true;
    bool showBaseGrid = true;
    float zoom = 1.0f;
    float rotX = 30.0f, rotY = -45.0f;
};

AppState g_state;

// -----------------------------------------------------------------------------
// Sheaf Logic: Analytic Continuation
// -----------------------------------------------------------------------------

// The covering map logic: w^2 = z - A
// Returns the two values in the stalk over z.
std::pair<std::complex<float>, std::complex<float>> getStalkValues(std::complex<float> z) {
    std::complex<float> delta = z - std::complex<float>(g_state.branchRe, g_state.branchIm);
    std::complex<float> w1 = std::sqrt(delta);
    return {w1, -w1};
}

// Lift a path from base space to the surface starting from a specific germ
void liftPath() {
    g_state.liftedPath.clear();
    if (g_state.pathZ.empty()) return;

    // Start with the germ closest to the previous probe state to maintain continuity context
    // or reset if starting fresh.
    auto startVals = getStalkValues(g_state.pathZ[0]);
    
    // Pick the value closest to the current probe "sheet" idea, 
    // effectively continuing the section.
    // For simplicity in this demo, we default to the positive root unless specified.
    std::complex<float> currentW = (g_state.activeSheet == 0) ? startVals.first : startVals.second;
    
    g_state.liftedPath.push_back(currentW);

    for (size_t i = 0; i < g_state.pathZ.size() - 1; ++i) {
        // Analytic continuation: small steps
        // To find the next w, we pick the root of (z_next - A) closest to current w.
        std::complex<float> zNext = g_state.pathZ[i+1];
        auto nextVals = getStalkValues(zNext);
        
        float d1 = std::abs(nextVals.first - currentW);
        float d2 = std::abs(nextVals.second - currentW);
        
        currentW = (d1 < d2) ? nextVals.first : nextVals.second;
        g_state.liftedPath.push_back(currentW);
    }
}

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
class GeometryView : public Fl_Gl_Window {
    int lastX, lastY;
    bool dragging = false;

public:
    GeometryView(int x, int y, int w, int h) : Fl_Gl_Window(x, y, w, h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
        // Default Path
        g_state.pathZ.push_back({-2.0f, -1.0f});
        g_state.pathZ.push_back({2.0f, -1.0f});
        liftPath();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_LINE_SMOOTH);
            glLineWidth(1.5f);
        }

        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float ratio = (float)w() / h();
        gluPerspective(45.0 / g_state.zoom, ratio, 0.1, 100.0);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0, 6, 10, 0, 0, 0, 0, 1, 0);
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        drawBaseGrid();
        if (g_state.showSheets) drawRiemannSurface();
        drawPathAndLift();
        drawStalkAndGerm();
    }

    int handle(int event) override {
        switch(event) {
            case FL_PUSH:
                lastX = Fl::event_x(); lastY = Fl::event_y();
                dragging = true;
                return 1;
            case FL_DRAG:
                if (dragging) {
                    g_state.rotY += (Fl::event_x() - lastX);
                    g_state.rotX += (Fl::event_y() - lastY);
                    lastX = Fl::event_x(); lastY = Fl::event_y();
                    redraw();
                }
                return 1;
            case FL_RELEASE:
                dragging = false; return 1;
            case FL_MOUSEWHEEL:
                g_state.zoom -= Fl::event_dy() * 0.1f;
                if (g_state.zoom < 0.1f) g_state.zoom = 0.1f;
                redraw();
                return 1;
            default: return Fl_Gl_Window::handle(event);
        }
    }

private:
    void drawBaseGrid() {
        if (!g_state.showBaseGrid) return;
        
        float y = -2.5f; // Base space is "below"
        
        // Grid
        glLineWidth(1.0f);
        glBegin(GL_LINES);
        glColor4f(0.3f, 0.3f, 0.3f, 0.5f);
        for (int i = -5; i <= 5; ++i) {
            glVertex3f(i, y, -5); glVertex3f(i, y, 5);
            glVertex3f(-5, y, i); glVertex3f(5, y, i);
        }
        glEnd();

        // Axes
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor3f(0.8, 0.3, 0.3); glVertex3f(-5, y, 0); glVertex3f(5, y, 0); // Real
        glColor3f(0.3, 0.8, 0.3); glVertex3f(0, y, -5); glVertex3f(0, y, 5); // Imag
        glEnd();

        // Branch Point on Base
        glPushMatrix();
        glTranslatef(g_state.branchRe, y, g_state.branchIm);
        glColor3f(1.0f, 0.2f, 0.2f);
        GLUquadric* q = gluNewQuadric();
        gluSphere(q, 0.15, 16, 16);
        gluDeleteQuadric(q);
        glPopMatrix();

        // Branch Cut (Convention: Negative Real Axis relative to branch point)
        glBegin(GL_LINES);
        glColor4f(1.0f, 0.5f, 0.0f, 0.8f);
        glVertex3f(g_state.branchRe, y, g_state.branchIm);
        glVertex3f(g_state.branchRe - 10.0f, y, g_state.branchIm);
        glEnd();
        
        // Label Base Space
        // (GL text is complex, using simple geometry indicator or rely on UI)
    }

    void drawRiemannSurface() {
        // We draw the two sheets of w = sqrt(z-A)
        // Height = Real(w), but we can compress it.
        // Let's use Height = Real(w) for 3D visualization.
        
        int grid = 60;
        float range = 4.0f;
        
        // We need to be careful with the branch cut when drawing triangles
        // to avoid ugly jagged edges connecting the two sheets incorrectly.
        // We iterate and color.
        
        for (int sheet = 0; sheet < 2; ++sheet) {
            for (int i = 0; i < grid; ++i) {
                float u0 = -range + (2.0f * range * i) / grid;
                float u1 = -range + (2.0f * range * (i + 1)) / grid;

                glBegin(GL_TRIANGLE_STRIP);
                for (int j = 0; j <= grid; ++j) {
                    float v = -range + (2.0f * range * j) / grid;
                    
                    auto drawVert = [&](float u, float v) {
                        std::complex<float> z(u, v);
                        auto vals = getStalkValues(z);
                        std::complex<float> w = (sheet == 0) ? vals.first : vals.second;
                        
                        // Check for branch cut proximity to avoid drawing across it?
                        // Actually, standard GL will just draw a steep slope, which is fine 
                        // as it represents the "gluing" visually if we don't separate them.
                        // But strictly, we want to see the topology.
                        
                        float col[4];
                        getDomainColor(w, g_state.surfaceAlpha, col);
                        glColor4fv(col);
                        glVertex3f(u, w.real(), v); 
                    };

                    drawVert(u0, v);
                    drawVert(u1, v);
                }
                glEnd();
            }
        }
    }

    void drawPathAndLift() {
        float baseY = -2.48f; // Slightly above grid
        
        // Base Path
        if (g_state.pathZ.size() > 1) {
            glLineWidth(3.0f);
            glColor4f(1.0f, 1.0f, 0.0f, 0.6f);
            glBegin(GL_LINE_STRIP);
            for (auto& z : g_state.pathZ) {
                glVertex3f(z.real(), baseY, z.imag());
            }
            glEnd();
        }

        // Lifted Path (The Section)
        if (g_state.liftedPath.size() > 1) {
            glLineWidth(4.0f);
            glColor4f(0.0f, 1.0f, 1.0f, 1.0f); // Cyan for the active section
            glBegin(GL_LINE_STRIP);
            for (size_t i = 0; i < g_state.liftedPath.size(); ++i) {
                // Need corresponding Z
                float t = (float)i / (g_state.liftedPath.size()-1);
                // Simple approx if indices match
                std::complex<float> z = g_state.pathZ[std::min(i, g_state.pathZ.size()-1)]; 
                // We re-calculate exact z if pathZ is sampled differently, but here indices align 1:1 roughly
                // actually simpler:
                if (i < g_state.pathZ.size()) {
                    z = g_state.pathZ[i];
                    glVertex3f(z.real(), g_state.liftedPath[i].real() + 0.02f, z.imag());
                }
            }
            glEnd();
        }
    }

    void drawStalkAndGerm() {
        // Calculate current probe position (interpolated if animating)
        std::complex<float> zCurr = g_state.probeZ;
        std::complex<float> wCurr;
        
        if (g_state.isAnimating && !g_state.pathZ.empty()) {
            size_t n = g_state.pathZ.size();
            float idxF = g_state.animTime * (n - 1);
            size_t i = (size_t)idxF;
            size_t next = std::min(i + 1, n - 1);
            float t = idxF - i;
            
            // Linear interp Z
            std::complex<float> z1 = g_state.pathZ[i];
            std::complex<float> z2 = g_state.pathZ[next];
            zCurr = z1 + t * (z2 - z1);

            // Linear interp W (Lifted)
            std::complex<float> w1 = g_state.liftedPath[i];
            std::complex<float> w2 = g_state.liftedPath[next];
            wCurr = w1 + t * (w2 - w1); // Note: this is visually correct, technically w follows sqrt
            
            // For exact math, we should re-calculate w from zCurr based on branch continuity
            // But w1->w2 interp is safe if steps are small.
        } else {
             // Just find the w on the active sheet
             // This logic is simplified; in a real tool we track the sheet ID.
             auto vals = getStalkValues(zCurr);
             // Find closest to last known lifted point? 
             // For static probe, just use sheet 0 or 1.
             wCurr = (g_state.activeSheet == 0) ? vals.first : vals.second;
        }

        float baseY = -2.5f;

        // 1. Draw The Stalk (Fiber)
        if (g_state.showStalk) {
            glLineWidth(1.0f);
            glColor4f(1.0f, 1.0f, 1.0f, 0.4f);
            
            auto vals = getStalkValues(zCurr);
            
            // Line from base to max height
            glBegin(GL_LINES);
            glVertex3f(zCurr.real(), baseY, zCurr.imag());
            glVertex3f(zCurr.real(), std::max(vals.first.real(), vals.second.real()) + 1.0f, zCurr.imag());
            glEnd();

            // Draw points on the fiber (Germs at this stalk)
            glPointSize(8.0f);
            glBegin(GL_POINTS);
            glColor3f(1, 0, 1); // Magenta points
            glVertex3f(zCurr.real(), vals.first.real(), zCurr.imag());
            glVertex3f(zCurr.real(), vals.second.real(), zCurr.imag());
            glEnd();
        }

        // 2. Draw The Active Germ (Section Neighborhood)
        if (g_state.showGerm) {
            // Draw a small disk on the surface around wCurr
            // This represents s(U) where U is a neighborhood of zCurr
            int segments = 20;
            float r = g_state.germRadius;
            
            glColor4f(0.0f, 1.0f, 1.0f, 0.8f); // Cyan patch
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(zCurr.real(), wCurr.real() + 0.01f, zCurr.imag()); // Center slightly offset up
            
            // We need to approximate the surface tangent or just simple flat disk
            // A flat disk is a good "linear approximation" (tangent space) visualization!
            for (int i = 0; i <= segments; ++i) {
                float theta = 2.0f * PI * i / segments;
                float dx = r * cos(theta);
                float dy = r * sin(theta);
                
                // We want the disk to lie "on" the surface.
                // Slope in Re(w) direction is roughly d(Re(sqrt))/dx... 
                // Let's just draw flat horizontal for clarity, or map exactly.
                // Mapping exactly is better:
                std::complex<float> zNeigh = zCurr + std::complex<float>(dx, dy);
                
                // We need the w value continuous to wCurr
                std::complex<float> wNeighVals = std::sqrt(zNeigh - std::complex<float>(g_state.branchRe, g_state.branchIm));
                std::complex<float> wNeigh = (std::abs(wNeighVals - wCurr) < std::abs(-wNeighVals - wCurr)) ? wNeighVals : -wNeighVals;

                glVertex3f(zNeigh.real(), wNeigh.real() + 0.01f, zNeigh.imag());
            }
            glEnd();

            // Draw shadow of Germ on Base (The Open Set U)
            glColor4f(0.0f, 1.0f, 1.0f, 0.2f);
            glBegin(GL_TRIANGLE_FAN);
            glVertex3f(zCurr.real(), baseY + 0.01f, zCurr.imag());
            for (int i = 0; i <= segments; ++i) {
                float theta = 2.0f * PI * i / segments;
                glVertex3f(zCurr.real() + r*cos(theta), baseY + 0.01f, zCurr.imag() + r*sin(theta));
            }
            glEnd();
        }
    }
};

// -----------------------------------------------------------------------------
// UI & Logic
// -----------------------------------------------------------------------------
GeometryView* g_glWin = nullptr;
Fl_Output* g_statusOut = nullptr;

void updatePath() {
    // Generate a circular path based on params
    // Center (0,0), radius 2?
    // Let's make a path that can encircle the branch point.
    
    // Simple parametric path for demo: Circle centered at (0,0) with radius 2.5
    // User can just drag the "Probe" sliders if they want manual control,
    // but the animation uses a fixed loop for clarity.
    
    g_state.pathZ.clear();
    int samples = 200;
    for (int i=0; i<=samples; ++i) {
        float t = (float)i/samples * 2.0f * PI;
        // A path that circles the origin. If branch point moves, it might be inside or outside.
        g_state.pathZ.push_back({2.5f * cos(t), 2.5f * sin(t)});
    }
    liftPath();
    if (g_glWin) g_glWin->redraw();
}

void anim_cb(void*) {
    if (!g_state.isAnimating) return;
    
    g_state.animTime += g_state.animSpeed;
    if (g_state.animTime > 1.0f) {
        g_state.animTime = 0.0f;
        // Check for monodromy: did we switch sheets?
        // Compare end of lifted path with start.
        std::complex<float> start = g_state.liftedPath.front();
        std::complex<float> end = g_state.liftedPath.back();
        
        if (std::abs(start - end) > 0.1f) {
            // Swapped sheets!
            g_state.activeSheet = 1 - g_state.activeSheet; // Toggle logic for simple 2-sheet
            
            // For visual continuity in loop, we might want to reset the lifted path
            // to start from the *new* sheet.
            // But since pathZ is a loop, startZ == endZ. 
            // We just re-lift based on the new "start" W.
            liftPath();
        }
    }
    
    // Update status text
    if (g_statusOut) {
        std::stringstream ss;
        ss << "T=" << std::fixed << std::setprecision(2) << g_state.animTime 
           << " | Sheet: " << g_state.activeSheet;
        g_statusOut->value(ss.str().c_str());
    }

    g_glWin->redraw();
    Fl::repeat_timeout(1.0/60.0, anim_cb);
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1200, 800, "Sheaf & Monodromy Visualizer");
    
    // Left: Controls
    Fl_Group* controls = new Fl_Group(10, 10, 300, 780);
    controls->box(FL_FLAT_BOX);
    
    Fl_Pack* pack = new Fl_Pack(10, 10, 300, 780);
    pack->spacing(10);
    
    // Title Box
    Fl_Box* title = new Fl_Box(0, 0, 300, 40, "GROTHENDIECK'S\nETALE SPACE");
    title->labelfont(FL_BOLD);
    title->labelsize(16);
    title->align(FL_ALIGN_CENTER);
    
    // Description
    Fl_Box* desc = new Fl_Box(0, 0, 300, 80, 
        "Visualizing the sheaf of germs\n"
        "for w = sqrt(z - A).\n"
        "Red Dot: Branch Point (Singularity)\n"
        "Cyan Disk: Local Section (Germ)");
    desc->align(FL_ALIGN_INSIDE | FL_ALIGN_WRAP | FL_ALIGN_LEFT);
    desc->box(FL_BORDER_BOX);
    
    // Branch Point Control
    Fl_Box* lbl1 = new Fl_Box(0,0,300,20,"Branch Point Position (A)");
    lbl1->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    
    Fl_Value_Slider* sldRe = new Fl_Value_Slider(0,0,300,25, "Re(A)");
    sldRe->type(FL_HOR_NICE_SLIDER); sldRe->bounds(-3,3); sldRe->value(0);
    sldRe->callback([](Fl_Widget* w, void*){ 
        g_state.branchRe = ((Fl_Value_Slider*)w)->value(); 
        liftPath(); 
        g_glWin->redraw(); 
    });

    Fl_Value_Slider* sldIm = new Fl_Value_Slider(0,0,300,25, "Im(A)");
    sldIm->type(FL_HOR_NICE_SLIDER); sldIm->bounds(-3,3); sldIm->value(0);
    sldIm->callback([](Fl_Widget* w, void*){ 
        g_state.branchIm = ((Fl_Value_Slider*)w)->value(); 
        liftPath();
        g_glWin->redraw(); 
    });

    // Germ Radius
    Fl_Value_Slider* sldGerm = new Fl_Value_Slider(0,0,300,25, "Germ Radius (Open Set U)");
    sldGerm->type(FL_HOR_NICE_SLIDER); sldGerm->bounds(0.1, 1.5); sldGerm->value(0.5);
    sldGerm->callback([](Fl_Widget* w, void*){ 
        g_state.germRadius = ((Fl_Value_Slider*)w)->value(); 
        g_glWin->redraw(); 
    });

    // Toggles
    Fl_Check_Button* chkStalk = new Fl_Check_Button(0,0,300,25, "Show Stalk (Fiber)");
    chkStalk->value(1);
    chkStalk->callback([](Fl_Widget* w, void*){ g_state.showStalk = ((Fl_Check_Button*)w)->value(); g_glWin->redraw(); });

    Fl_Check_Button* chkGerm = new Fl_Check_Button(0,0,300,25, "Show Germ (Section)");
    chkGerm->value(1);
    chkGerm->callback([](Fl_Widget* w, void*){ g_state.showGerm = ((Fl_Check_Button*)w)->value(); g_glWin->redraw(); });

    // Animation Controls
    Fl_Box* sep = new Fl_Box(0,0,300,20,"--- Monodromy Simulation ---");
    
    Fl_Button* btnAnim = new Fl_Button(0,0,300,30, "Start/Stop Animation");
    btnAnim->callback([](Fl_Widget*, void*){
        g_state.isAnimating = !g_state.isAnimating;
        if(g_state.isAnimating) Fl::add_timeout(0, anim_cb);
    });

    g_statusOut = new Fl_Output(0,0,300,25, "Status");
    
    pack->end();
    controls->end();

    // Right: GL View
    g_glWin = new GeometryView(320, 10, 870, 780);
    g_glWin->end();

    win->end();
    win->resizable(g_glWin);
    
    // Initialize Path
    updatePath();
    
    win->show(argc, argv);
    return Fl::run();
}
