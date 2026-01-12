// Compile with: g++ -o analytical_sheaf analytical_sheaf.cpp -lfltk -lfltk_gl -lGL -lGLU
//
// CONTEXT: GROTHENDIECK ANALYTICAL GEOMETRY
// This simulation visualizes a "Covering Space" or "Etale Space".
// We visualize the analytical function w = sqrt(z - A) not as a graph,
// but as a Bundle over the base space C.
// 
// Concepts Demonstrated:
// 1. The Base Space S (The flat grid below).
// 2. The Total Space X (The Riemann Surface above).
// 3. The Projection pi: X -> S (The vertical relationship).
// 4. The Fiber pi^{-1}(z) (The yellow vertical stalk).
// 5. Ramification: The point where the fiber size changes.

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Box.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <complex>
#include <cmath>
#include <vector>
#include <iostream>

// Constants
const float PI = 3.14159265359f;

// -------------------------------------------------------------------------
// Global State representing the Morphism
// -------------------------------------------------------------------------
struct MorphismState {
    float branchPointRe = 1.0f; // Real part of A in w^2 = z - A
    float transparency = 0.6f;  // To see the structure
    
    // Interaction state
    float mouseRe = 0.0f;
    float mouseIm = 0.0f;
    bool showFiber = true;
};

MorphismState g_state;

// -------------------------------------------------------------------------
// Math Helpers: Complex HSV Coloring
// -------------------------------------------------------------------------
void setComplexColor(float r, float g, float b, float alpha) {
    glColor4f(r, g, b, alpha);
}

// Maps the argument (phase) of w to a hue
void colorByPhase(std::complex<float> w, float alpha) {
    float angle = std::arg(w);
    float h = angle / (2.0f * PI); 
    if(h < 0) h += 1.0f;
    
    float s = 1.0f;
    float v = 1.0f;
    
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    float r=0, g=0, b=0;
    if(h < 1/6.0f) { r=c; g=x; }
    else if(h < 2/6.0f) { r=x; g=c; }
    else if(h < 3/6.0f) { g=c; b=x; }
    else if(h < 4/6.0f) { g=x; b=c; }
    else if(h < 5/6.0f) { r=x; b=c; }
    else { r=c; b=x; }
    
    glColor4f(r+m, g+m, b+m, alpha);
}

// -------------------------------------------------------------------------
// OpenGL Renderer
// -------------------------------------------------------------------------
class SheafWindow : public Fl_Gl_Window {
    float rotX, rotY;
    int lastX, lastY;
    bool dragging;

public:
    SheafWindow(int x, int y, int w, int h) 
        : Fl_Gl_Window(x, y, w, h), rotX(30), rotY(-45), dragging(false) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_MULTISAMPLE | FL_ALPHA);
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(45.0, (float)w()/(float)h(), 0.1, 100.0);
            glMatrixMode(GL_MODELVIEW);
        }

        // Dark Analytical Background
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f); 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        
        // Camera setup
        gluLookAt(0, 4, 10, 0, 0, 0, 0, 1, 0);
        glRotatef(rotX, 1, 0, 0);
        glRotatef(rotY, 0, 1, 0);

        // 1. Draw the Base Space (Complex Plane C)
        drawBaseGrid();

        // 2. Draw the Total Space (Riemann Surface X)
        drawRiemannSurface();

        // 3. Draw the Fiber (Stalk) interaction
        if(g_state.showFiber) {
            drawFiberInteraction();
        }
    }

    int handle(int event) override {
        switch(event) {
            case FL_PUSH:
                lastX = Fl::event_x();
                lastY = Fl::event_y();
                dragging = true;
                return 1;
            case FL_DRAG:
                if (dragging) {
                    rotY += (Fl::event_x() - lastX);
                    rotX += (Fl::event_y() - lastY);
                    lastX = Fl::event_x();
                    lastY = Fl::event_y();
                    redraw();
                }
                return 1;
            case FL_RELEASE:
                dragging = false;
                return 1;
            default:
                return Fl_Gl_Window::handle(event);
        }
    }

private:
    // Helper to map 3D world coords to Screen Space is omitted for brevity,
    // so we control the "Probe" point via the UI slider or assumptions.
    // For this educational viz, we assume the probe is set via the UI state.

    void drawBaseGrid() {
        // The Base Space S = C (z-plane)
        // We draw it at y = -2.0 to physically separate Base from Bundle
        float y_base = -2.0f;
        float size = 4.0f;
        int divs = 10;

        glLineWidth(1.0f);
        glColor4f(0.5f, 0.5f, 0.5f, 0.3f);
        
        glBegin(GL_LINES);
        for(int i=-divs; i<=divs; ++i) {
            float v = (float)i/divs * size;
            glVertex3f(-size, y_base, v);
            glVertex3f(size,  y_base, v);
            glVertex3f(v, y_base, -size);
            glVertex3f(v, y_base, size);
        }
        glEnd();

        // Mark the Branch Point on the Base
        glPushMatrix();
        glTranslatef(g_state.branchPointRe, y_base, 0);
        glColor4f(1.0f, 0.2f, 0.2f, 1.0f); // Red indicate singularity
        GLUquadric* q = gluNewQuadric();
        gluSphere(q, 0.1f, 10, 10);
        gluDeleteQuadric(q);
        glPopMatrix();
    }

    void drawRiemannSurface() {
        // We visualize the multivalued function w = sqrt(z - A)
        // This is a manifold X over C.
        // We plot Real(w) as vertical height, but we must handle the two sheets.
        
        int res = 60; // Grid resolution
        float limit = 3.5f;
        
        // We draw both branches.
        // Branch 1: Positive root
        // Branch 2: Negative root
        
        for(int k=0; k<2; ++k) {
            int sign = (k==0) ? 1 : -1;
            
            glBegin(GL_QUADS);
            for(int i = 0; i < res; ++i) {
                for(int j = 0; j < res; ++j) {
                    // Map i,j to Complex Plane (z) coordinates
                    float u = (float)i/res * 2*limit - limit;
                    float v = (float)j/res * 2*limit - limit;
                    float step = (2*limit)/res;

                    auto eval = [&](float r, float im) {
                        std::complex<float> z(r, im);
                        std::complex<float> a(g_state.branchPointRe, 0);
                        // w = sqrt(z - A)
                        std::complex<float> w = std::sqrt(z - a);
                        if(sign == -1) w = -w;

                        // Visual Mapping:
                        // x = Re(z)
                        // z = Im(z)
                        // y = Re(w) (Height represents real part of value)
                        // Color = Arg(w)
                        
                        colorByPhase(w, g_state.transparency);
                        // Flatten near zero to avoid visual spikes
                        glVertex3f(r, w.real(), im); 
                    };

                    eval(u, v);
                    eval(u+step, v);
                    eval(u+step, v+step);
                    eval(u, v+step);
                }
            }
            glEnd();
        }
    }

    void drawFiberInteraction() {
        // Represents the fiber pi^{-1}(z)
        // z is the point (mouseRe, mouseIm) in the Base
        
        float z_re = g_state.mouseRe;
        float z_im = g_state.mouseIm;
        float y_base = -2.0f;

        // 1. Draw the Base Point
        glPushMatrix();
        glTranslatef(z_re, y_base, z_im);
        glColor4f(1.0f, 1.0f, 0.0f, 1.0f); // Yellow probe
        GLUquadric* q = gluNewQuadric();
        gluSphere(q, 0.1f, 10, 10);
        glPopMatrix();

        // 2. Calculate Fiber Points in X (the Surface)
        std::complex<float> z(z_re, z_im);
        std::complex<float> a(g_state.branchPointRe, 0);
        std::complex<float> w1 = std::sqrt(z - a);
        std::complex<float> w2 = -w1;

        // 3. Draw "Stalk" (Line from base to surface points)
        glLineWidth(2.0f);
        glBegin(GL_LINES);
        
        // Stalk 1
        glColor4f(1.0f, 1.0f, 0.0f, 0.5f);
        glVertex3f(z_re, y_base, z_im);
        glVertex3f(z_re, w1.real(), z_im);
        
        // Stalk 2
        glVertex3f(z_re, y_base, z_im);
        glVertex3f(z_re, w2.real(), z_im);
        glEnd();

        // 4. Draw Points on the Surface (The Geometric Fiber)
        // Point 1
        glPushMatrix();
        glTranslatef(z_re, w1.real(), z_im);
        glColor4f(0.2f, 1.0f, 0.2f, 1.0f); // Green dot on surface
        gluSphere(q, 0.1f, 10, 10);
        glPopMatrix();

        // Point 2
        glPushMatrix();
        glTranslatef(z_re, w2.real(), z_im);
        glColor4f(0.2f, 1.0f, 0.2f, 1.0f); 
        gluSphere(q, 0.1f, 10, 10);
        glPopMatrix();

        gluDeleteQuadric(q);
    }
};

// -------------------------------------------------------------------------
// UI Callbacks
// -------------------------------------------------------------------------
SheafWindow* g_glWin = nullptr;

void update_cb(Fl_Widget*, void*) {
    if(g_glWin) g_glWin->redraw();
}

void slider_branch_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.branchPointRe = s->value();
    g_glWin->redraw();
}

void slider_alpha_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.transparency = s->value();
    g_glWin->redraw();
}

// Controls the "Probe" z position
void slider_zx_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.mouseRe = s->value();
    g_glWin->redraw();
}

void slider_zy_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.mouseIm = s->value();
    g_glWin->redraw();
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char **argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1000, 600, "Grothendieck Analytical Bundle Viz");
    
    g_glWin = new SheafWindow(10, 10, 700, 580);
    
    Fl_Group* ui = new Fl_Group(720, 10, 270, 580);
    
    // Educational Text
    Fl_Box* txt = new Fl_Box(720, 20, 270, 100, 
        "Visualizing the Etale Space\n"
        "of the Sheaf defined by:\n"
        "w^2 = z - A\n\n"
        "Surface = Total Space X\n"
        "Grid = Base Space S");
    txt->align(FL_ALIGN_INSIDE | FL_ALIGN_TOP | FL_ALIGN_WRAP);
    txt->labelfont(FL_HELVETICA_BOLD);
    
    int y = 140;
    
    Fl_Value_Slider* s1 = new Fl_Value_Slider(730, y, 250, 25, "Branch Point A (Real Part)");
    s1->bounds(-2.0, 2.0);
    s1->value(g_state.branchPointRe);
    s1->callback(slider_branch_cb);
    
    y += 50;
    Fl_Value_Slider* s2 = new Fl_Value_Slider(730, y, 250, 25, "Surface Transparency");
    s2->bounds(0.1, 1.0);
    s2->value(g_state.transparency);
    s2->callback(slider_alpha_cb);
    
    y += 60;
    Fl_Box* sep = new Fl_Box(720, y, 270, 30, "--- Explore Fibers ---");
    sep->labelfont(FL_HELVETICA_BOLD);
    
    y += 40;
    Fl_Value_Slider* sx = new Fl_Value_Slider(730, y, 250, 25, "Base Point z (Real)");
    sx->bounds(-3.0, 3.0);
    sx->value(0.0);
    sx->callback(slider_zx_cb);
    
    y += 50;
    Fl_Value_Slider* sy = new Fl_Value_Slider(730, y, 250, 25, "Base Point z (Imag)");
    sy->bounds(-3.0, 3.0);
    sy->value(0.0);
    sy->callback(slider_zy_cb);

    ui->end();
    
    win->end();
    win->resizable(g_glWin);
    win->show(argc, argv);
    return Fl::run();
}
