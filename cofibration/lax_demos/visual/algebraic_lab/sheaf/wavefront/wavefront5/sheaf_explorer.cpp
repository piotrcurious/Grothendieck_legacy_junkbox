#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <cmath>
#include <vector>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// High-Dimensional Hyperparameter Set for Microlocal Analysis
// -----------------------------------------------------------------------------
struct WavefrontParams {
    // Spatial Parameters
    float defocus = 0.0f;      // Second order (focusing)
    float spherical = 0.0f;    // Fourth order
    float coma = 0.0f;         // Third order asymmetry
    float astigmatism = 0.0f;  // Cylindrical curvature
    
    // Dynamical Parameters
    float propagation = 0.0f;  // The 'time' or 'z' evolution
    float non_linearity = 0.0f;// Phase-dependent index shift
    float aperture = 1.0f;     // Spatial clipping
    
    // Visualization
    float morph_t = 0.0f;      // Dynamic evolution time
};

struct AppState {
    float rotX = 20.0f, rotY = -35.0f;
    float section_p = 0.0f;    // The 'momentum' we are slicing
    WavefrontParams p;
    bool showEtale = true;
    bool animate = false;
    float anim_count = 0.0f;
};

// -----------------------------------------------------------------------------
// The Microlocal Sheaf Engine
// -----------------------------------------------------------------------------
// We model the 1-jet space (x, p, S) where:
// x = spatial coordinate
// p = local momentum (dS/dx)
// S = Action / Phase
// -----------------------------------------------------------------------------
class MicrolocalEngine {
public:
    // Calculates the coordinate in the 5D phase space and projects to 3D for visualization
    void getJet(float x_in, float p_in, float out[3], const WavefrontParams& wp) {
        // Base coordinate x
        float x = x_in * wp.aperture;
        
        // Generating function (Phase Surface) S(x)
        // Includes Defocus, Coma, and Spherical Abberation
        float S_base = (wp.defocus * 0.5f * x * x) + 
                       (wp.coma * (1.0f/3.0f) * x * x * x) + 
                       (wp.spherical * 0.25f * x * x * x * x);
        
        // Local momentum p = dS/dx
        float p_local = (wp.defocus * x) + 
                        (wp.coma * x * x) + 
                        (wp.spherical * x * x * x);
        
        // Propagation evolution via Hamilton-Jacobi (Free space + Non-linearity)
        // New x' = x + p*z
        float z = wp.propagation;
        float x_prop = x + p_local * z + wp.astigmatism * sinf(x*2.0f);
        
        // Phase accumulation along the ray
        float S_prop = S_base + (0.5f * p_local * p_local * z) + wp.non_linearity * cosf(x + wp.morph_t);

        // Map to GL space
        out[0] = x_prop * 4.0f;   // Visual X: Evolved Position
        out[1] = S_prop * 2.0f;   // Visual Y: Action (Sheaf Value)
        out[2] = p_local * 6.0f;  // Visual Z: Momentum (Stalk parameter)
    }
};

// -----------------------------------------------------------------------------
// Visualization & Windowing
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    AppState* state;
    MicrolocalEngine engine;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.012f, 0.014f, 0.018f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(40, (float)w()/h(), 0.1, 300);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(30, 30, 50, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Draw Reference Grid (Optical Axis)
        glBegin(GL_LINES);
        glColor4f(1, 1, 1, 0.1f);
        for(int i=-10; i<=10; i++) {
            glVertex3f(i*4.0f, -10, -20); glVertex3f(i*4.0f, -10, 20);
            glVertex3f(-40, -10, i*2.0f); glVertex3f(40, -10, i*2.0f);
        }
        glEnd();

        if (state->showEtale) {
            int x_steps = 100, p_steps = 40;
            glBegin(GL_QUADS);
            for(int i=0; i<x_steps; i++) {
                float x1 = -PI + (2*PI*i/x_steps);
                float x2 = -PI + (2*PI*(i+1)/x_steps);
                for(int j=0; j<p_steps; j++) {
                    float p1 = -1.0f + (2.0f*j/p_steps);
                    float p2 = -1.0f + (2.0f*(j+1)/p_steps);
                    float v1[3], v2[3], v3[3], v4[3];
                    engine.getJet(x1, p1, v1, state->p);
                    engine.getJet(x2, p1, v2, state->p);
                    engine.getJet(x2, p2, v3, state->p);
                    engine.getJet(x1, p2, v4, state->p);
                    
                    // Color based on local momentum density
                    float hue = 0.5f + 0.4f * p1;
                    glColor4f(0.2f, hue, 1.0f, 0.25f);
                    glVertex3fv(v1); glVertex3fv(v2); glVertex3fv(v3); glVertex3fv(v4);
                }
            }
            glEnd();
        }

        // Section (Specific Ray path)
        glColor3f(1.0f, 0.4f, 0.0f); glLineWidth(4.0f);
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=200; i++) {
            float x = -PI + (2*PI*i/200.0f);
            float v[3]; engine.getJet(x, state->section_p, v, state->p);
            glVertex3fv(v);
        }
        glEnd(); glLineWidth(1.0f);
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            state->rotY += (float)(Fl::event_x()-lx) * 0.5f;
            state->rotX += (float)(Fl::event_y()-ly) * 0.5f;
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Hamilton-Jacobi Sheaf Explorer");
    
    int sw = 350;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(15, 17, 22));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "WAVEFRONT ETALE");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(22);

    int y = 80;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 65; return s;
    };

    create_s("Propagation Distance (z)", -5.0f, 5.0f, 0.0f, &g_state.p.propagation);
    create_s("Defocus (Second Order)", -2.0f, 2.0f, 0.5f, &g_state.p.defocus);
    create_s("Coma (Third Order)", -1.0f, 1.0f, 0.0f, &g_state.p.coma);
    create_s("Spherical (Fourth Order)", -1.0f, 1.0f, 0.1f, &g_state.p.spherical);
    create_s("Astigmatism Twist", -1.0f, 1.0f, 0.0f, &g_state.p.astigmatism);
    create_s("Non-linear Kerr Index", 0.0f, 2.0f, 0.0f, &g_state.p.non_linearity);
    create_s("Aperture Numerical Scale", 0.1f, 3.0f, 1.0f, &g_state.p.aperture);
    
    y += 10;
    Fl_Box* sep = new Fl_Box(20, y, sw-40, 1); sep->box(FL_FLAT_BOX); sep->color(FL_DARK3);
    y += 20;

    create_s("Fiber Selection (p-slice)", -1.0f, 1.0f, 0.0f, &g_state.section_p);

    auto b_mesh = new Fl_Check_Button(20, y, 140, 25, "Render Jet Surface");
    b_mesh->labelcolor(FL_WHITE); b_mesh->value(1);
    b_mesh->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showEtale = ((Fl_Check_Button*)w)->value(); }, &g_state);

    auto b_anim = new Fl_Check_Button(180, y, 140, 25, "Live Dynamics");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();

    SheafView* view = new SheafView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.anim_count += 0.02f;
            g_state.p.morph_t = g_state.anim_count;
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
