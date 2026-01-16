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
// Symplectic & Microlocal Parameters
// -----------------------------------------------------------------------------
struct MicrolocalParams {
    // Hamiltonian Phase-Space Coefficients
    float p_defocus = 0.5f;     // Quadratic (Focusing)
    float p_coma = 0.0f;        // Cubic (Asymmetry)
    float p_spherical = 0.1f;   // Quartic (Spherical)
    float p_trefoil = 0.0f;     // Angular 3-fold symmetry
    
    // External Field / Flow
    float z_propagation = 0.0f; // Time-like evolution in phase space
    float symplectic_twist = 0.0f; // Berry phase / Geometric rotation
    float potential_well = 0.0f; // Non-linear refractive index (Kerr)
    
    // Geometry of the Base Manifold
    float aperture = 1.0f;      
    float fiber_scale = 1.0f;   
};

struct AppState {
    float rotX = 25.0f, rotY = -40.0f;
    MicrolocalParams p;
    bool showLagrangian = true;
    bool showCaustics = true;
    bool animate = false;
    float time = 0.0f;
};

// -----------------------------------------------------------------------------
// Hamiltonian Dynamics Engine
// -----------------------------------------------------------------------------
class SymplecticEngine {
public:
    struct PhasePoint {
        float x, p, S; // Position, Momentum, Action (Phase)
    };

    // Computes the evolution of a point in the 1-jet space J^1(R)
    PhasePoint evolve(float x0, float p0, const MicrolocalParams& params) {
        // Initial Action based on generating function S(x)
        // This represents the initial section of our sheaf
        float S0 = (params.p_defocus * 0.5f * x0 * x0) + 
                   (params.p_coma * (1.0f/3.0f) * powf(x0, 3.0f)) + 
                   (params.p_spherical * 0.25f * powf(x0, 4.0f));

        // Initial Momentum p = dS/dx
        float p_start = (params.p_defocus * x0) + 
                        (params.p_coma * x0 * x0) + 
                        (params.p_spherical * powf(x0, 3.0f)) + p0;

        // Hamiltonian Flow (Free space propagation)
        // dx/dz = p, dp/dz = -dV/dx (where V is the potential_well)
        float z = params.z_propagation;
        
        // Potential gradient (simplified Kerr effect)
        float dVdx = params.potential_well * sinf(x0 + params.symplectic_twist);
        
        // Symplectic mapping (Hamilton-Jacobi)
        float x_final = x0 + p_start * z;
        float p_final = p_start - dVdx * z;
        
        // Action evolution (Integral of L dt)
        // S = Integral(p dx - H dt)
        float S_final = S0 + (0.5f * p_start * p_start * z) - (params.potential_well * z);

        return {x_final, p_final, S_final};
    }
};

// -----------------------------------------------------------------------------
// Visualization
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    AppState* state;
    SymplecticEngine engine;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.02f, 0.02f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 0.1, 500);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(40, 40, 60, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Draw Phase Space Axes
        glBegin(GL_LINES);
        glColor4f(0.5, 0.5, 0.5, 0.3f);
        glVertex3f(-50, 0, 0); glVertex3f(50, 0, 0); // X axis
        glVertex3f(0, -30, 0); glVertex3f(0, 30, 0); // S axis (Action)
        glVertex3f(0, 0, -50); glVertex3f(0, 0, 50); // P axis (Momentum)
        glEnd();

        if (state->showLagrangian) {
            int x_res = 120, p_res = 40;
            glBegin(GL_TRIANGLES);
            for(int i=0; i<x_res; i++) {
                float u1 = -PI + (2*PI*i/x_res);
                float u2 = -PI + (2*PI*(i+1)/x_res);
                for(int j=0; j<p_res; j++) {
                    float v1 = -1.5f + (3.0f*j/p_res);
                    float v2 = -1.5f + (3.0f*(j+1)/p_res);
                    
                    auto p1 = engine.evolve(u1, v1, state->p);
                    auto p2 = engine.evolve(u2, v1, state->p);
                    auto p3 = engine.evolve(u2, v2, state->p);
                    auto p4 = engine.evolve(u1, v2, state->p);
                    
                    // The "Etale Space" intensity depends on the Jacobian of the mapping
                    // which indicates how sections are "glued"
                    float det = 1.0f / (1.0f + state->p.z_propagation * state->p.p_defocus);
                    glColor4f(0.3f, 0.5f, 1.0f, 0.15f * fabsf(det));
                    
                    glVertex3f(p1.x*5, p1.S*3, p1.p*8); 
                    glVertex3f(p2.x*5, p2.S*3, p2.p*8);
                    glVertex3f(p3.x*5, p3.S*3, p3.p*8);

                    glVertex3f(p1.x*5, p1.S*3, p1.p*8);
                    glVertex3f(p3.x*5, p3.S*3, p3.p*8);
                    glVertex3f(p4.x*5, p4.S*3, p4.p*8);
                }
            }
            glEnd();
        }

        if (state->showCaustics) {
            // Draw the Fold lines (where the sheaf projection is singular)
            glColor3f(1.0f, 0.2f, 0.2f); glLineWidth(2.0f);
            glBegin(GL_LINE_STRIP);
            for(float x=-PI; x<=PI; x+=0.05f) {
                auto pt = engine.evolve(x, 0.0f, state->p);
                glVertex3f(pt.x*5.1f, pt.S*3.1f, pt.p*8.1f);
            }
            glEnd(); glLineWidth(1.0f);
        }
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
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Symplectic Microlocal Sheaf Engine");
    
    int sw = 360;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(18, 20, 26));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "MICROLOCAL JET SPACE");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(20);

    int y = 80;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 65; return s;
    };

    create_s("Symplectic Flow (z)", -5.0f, 5.0f, 0.0f, &g_state.p.z_propagation);
    create_s("Quadratic Focus", -2.0f, 2.0f, 0.5f, &g_state.p.p_defocus);
    create_s("Cubic Asymmetry (Coma)", -1.0f, 1.0f, 0.0f, &g_state.p.p_coma);
    create_s("Quartic Spherical", -0.5f, 0.5f, 0.1f, &g_state.p.p_spherical);
    create_s("Geometric Twist", -PI, PI, 0.0f, &g_state.p.symplectic_twist);
    create_s("Potential Well (Kerr)", 0.0f, 2.0f, 0.0f, &g_state.p.potential_well);
    
    y += 20;
    auto b_lagrange = new Fl_Check_Button(20, y, 160, 25, "Lagrangian Manifold");
    b_lagrange->labelcolor(FL_WHITE); b_lagrange->value(1);
    b_lagrange->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showLagrangian = ((Fl_Check_Button*)w)->value(); }, &g_state);

    auto b_caustic = new Fl_Check_Button(180, y, 160, 25, "Caustic Locus");
    b_caustic->labelcolor(FL_WHITE); b_caustic->value(1);
    b_caustic->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showCaustics = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto b_anim = new Fl_Check_Button(20, y, 160, 25, "Animate Flow");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();

    SheafView* view = new SheafView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.time += 0.01f;
            g_state.p.z_propagation = 2.0f * sinf(g_state.time);
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
