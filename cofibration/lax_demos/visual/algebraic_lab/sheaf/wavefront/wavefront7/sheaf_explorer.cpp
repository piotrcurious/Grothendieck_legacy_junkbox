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
// Rigorous Symplectic / Jet-Space Definitions
// -----------------------------------------------------------------------------
struct MicrolocalParams {
    // Initial Section Parameters (Phase-front at z=0)
    float c_defocus = 0.5f;     // Zernike Z4
    float c_astig   = 0.0f;     // Zernike Z5/6
    float c_coma    = 0.0f;     // Zernike Z7/8
    float c_sphere  = 0.1f;     // Zernike Z11
    float c_trefoil = 0.0f;     // Zernike Z9/10

    // Symplectic Flow Parameters (The "Z-Evolution")
    float prop_z = 0.0f;        // Evolution time
    float kerr_gamma = 0.0f;    // Non-linear potential depth
    float lattice_k = 1.0f;     // Spatial frequency of the potential
    float phase_shift = 0.0f;   // Global phase / Berry phase shift

    // Manifold Boundaries
    float aperture_width = 2.5f;
    float stalk_resolution = 1.0f;
};

struct AppState {
    float rotX = 25.0f, rotY = -40.0f;
    MicrolocalParams p;
    bool showLagrangian = true;
    bool showPhaseSpace = true;
    bool animate = false;
    float time_accumulator = 0.0f;
};

// -----------------------------------------------------------------------------
// Symplectic Integrator (Hamiltonian Flow)
// -----------------------------------------------------------------------------
class HamiltonianIntegrator {
public:
    struct StateVector {
        float x, p, S; // Position, Momentum, Action (Phase)
    };

    // Computes the initial 1-jet of the sheaf at z=0
    StateVector get_initial_jet(float x0, const MicrolocalParams& p) {
        // We define the initial phase S(x) as a Zernike-like polynomial
        // S(x) = sum c_i * P_i(x)
        float S = (p.c_defocus * 0.5f * x0 * x0) + 
                  (p.c_astig   * 0.5f * x0 * x0 * sinf(x0)) + 
                  (p.c_coma    * (1.0f/3.0f) * powf(x0, 3.0f)) + 
                  (p.c_sphere  * 0.25f * powf(x0, 4.0f)) +
                  (p.c_trefoil * powf(x0, 3.0f) * cosf(3.0f * x0));

        // The momentum p is the exact differential p = dS/dx
        // dS/dx = c_defocus*x + ...
        float p_init = (p.c_defocus * x0) + 
                       (p.c_astig   * (x0 * sinf(x0) + 0.5f * x0 * x0 * cosf(x0))) +
                       (p.c_coma    * x0 * x0) + 
                       (p.c_sphere  * powf(x0, 3.0f)) +
                       (p.c_trefoil * (3.0f * x0 * x0 * cosf(3.0f * x0) - 3.0f * powf(x0, 3.0f) * sinf(3.0f * x0)));

        return {x0, p_init, S + p.phase_shift};
    }

    // Evolves the jet through the Hamilton-Jacobi flow using Verlet Integration
    StateVector evolve(float x_start, float p_offset, const MicrolocalParams& p) {
        StateVector sv = get_initial_jet(x_start, p);
        sv.p += p_offset; // Adding fiber offset (scanning the stalk)

        float dt = 0.05f;
        int steps = std::abs((int)(p.prop_z / dt));
        if (steps == 0) return sv;
        
        float current_z = 0;
        float direction = (p.prop_z > 0) ? 1.0f : -1.0f;
        float step_size = dt * direction;

        for(int i = 0; i < steps; ++i) {
            // Hamiltonian: H(x, p) = p^2/2 + V(x)
            // Potential: V(x) = -gamma * cos(k * x)
            // Equations: dx/dz = p, dp/dz = -dV/dx = -gamma * k * sin(k * x)

            // Half-step momentum
            float dVdx = p.kerr_gamma * p.lattice_k * sinf(p.lattice_k * sv.x);
            sv.p -= 0.5f * dVdx * step_size;
            
            // Full-step position
            sv.x += sv.p * step_size;
            
            // Re-calculate force for second half
            dVdx = p.kerr_gamma * p.lattice_k * sinf(p.lattice_k * sv.x);
            sv.p -= 0.5f * dVdx * step_size;

            // Update Action: dS/dz = L = p*dx/dz - H = p^2 - (p^2/2 + V) = p^2/2 - V
            float V = -p.kerr_gamma * cosf(p.lattice_k * sv.x);
            sv.S += (0.5f * sv.p * sv.p - V) * step_size;
        }

        return sv;
    }
};

// -----------------------------------------------------------------------------
// Visualization Engine
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    AppState* state;
    HamiltonianIntegrator integrator;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        }
        glClearColor(0.015f, 0.015f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(40, (float)w()/h(), 0.1, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(60, 50, 80, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Reference Grid (Symplectic Phase Plane)
        glBegin(GL_LINES);
        glColor4f(0.4f, 0.4f, 0.6f, 0.2f);
        for(int i=-15; i<=15; i++) {
            glVertex3f(i*4.0f, -20, -50); glVertex3f(i*4.0f, -20, 50);
            glVertex3f(-60, -20, i*3.3f); glVertex3f(60, -20, i*3.3f);
        }
        glEnd();

        if (state->showLagrangian) {
            int x_res = 120, p_res = 30;
            float ap = state->p.aperture_width;

            glBegin(GL_TRIANGLES);
            for(int i=0; i<x_res; i++) {
                float x1 = -ap + (2.0f*ap*i/x_res);
                float x2 = -ap + (2.0f*ap*(i+1)/x_res);
                for(int j=0; j<p_res; j++) {
                    float f1 = -1.2f + (2.4f*j/p_res);
                    float f2 = -1.2f + (2.4f*(j+1)/p_res);
                    
                    auto pt1 = integrator.evolve(x1, f1, state->p);
                    auto pt2 = integrator.evolve(x2, f1, state->p);
                    auto pt3 = integrator.evolve(x2, f2, state->p);
                    auto pt4 = integrator.evolve(x1, f2, state->p);
                    
                    // The "Intensity" or "Sheaf Density" is inversely proportional 
                    // to the divergence of the trajectories (Caustics)
                    float dx = fabsf(pt2.x - pt1.x) + 1e-4f;
                    float weight = 0.4f / dx;
                    if(weight > 1.0f) weight = 1.0f;

                    glColor4f(0.2f, 0.6f, 1.0f, 0.2f * weight);
                    
                    glVertex3f(pt1.x*6, pt1.S*4, pt1.p*10); 
                    glVertex3f(pt2.x*6, pt2.S*4, pt2.p*10);
                    glVertex3f(pt3.x*6, pt3.S*4, pt3.p*10);

                    glVertex3f(pt1.x*6, pt1.S*4, pt1.p*10);
                    glVertex3f(pt3.x*6, pt3.S*4, pt3.p*10);
                    glVertex3f(pt4.x*6, pt4.S*4, pt4.p*10);
                }
            }
            glEnd();
        }

        if (state->showPhaseSpace) {
            // Visualize the "Caustic Fold" - the critical locus of the projection
            glColor3f(1.0f, 0.3f, 0.1f); glLineWidth(3.0f);
            glBegin(GL_LINE_STRIP);
            float ap = state->p.aperture_width;
            for(float x=-ap; x<=ap; x+=0.02f) {
                auto pt = integrator.evolve(x, 0.0f, state->p);
                glVertex3f(pt.x*6.02f, pt.S*4.02f, pt.p*10.02f);
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

// -----------------------------------------------------------------------------
// Application Entry and UI Setup
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Rigorous Microlocal Symplectic Explorer");
    
    int sw = 380;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(22, 24, 30));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "SYMPLECTIC INTEGRATOR");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(18);

    int y = 70;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 60; return s;
    };

    create_s("Hamiltonian Flow (z-prop)", -8.0f, 8.0f, 0.0f, &g_state.p.prop_z);
    create_s("Z4: Defocus / Curvature", -2.0f, 2.0f, 0.5f, &g_state.p.c_defocus);
    create_s("Z5: Primary Astigmatism", -1.5f, 1.5f, 0.0f, &g_state.p.c_astig);
    create_s("Z7: Coma Abberation", -1.0f, 1.0f, 0.0f, &g_state.p.c_coma);
    create_s("Z9: Trefoil Topology", -1.0f, 1.0f, 0.0f, &g_state.p.c_trefoil);
    create_s("Z11: Spherical Order", -0.5f, 0.5f, 0.1f, &g_state.p.c_sphere);
    
    y += 10;
    Fl_Box* sep1 = new Fl_Box(20, y, sw-40, 2); sep1->box(FL_FLAT_BOX); sep1->color(FL_DARK3);
    y += 20;

    create_s("Kerr Non-linearity (Gamma)", 0.0f, 3.0f, 0.0f, &g_state.p.kerr_gamma);
    create_s("Potential Spatial Frequency", 0.1f, 5.0f, 1.0f, &g_state.p.lattice_k);
    create_s("Aperture (Manifold Width)", 0.5f, 5.0f, 2.5f, &g_state.p.aperture_width);
    create_s("Berry Phase Offset", -PI, PI, 0.0f, &g_state.p.phase_shift);

    auto b_mesh = new Fl_Check_Button(20, y, 160, 25, "Render Etale Mesh");
    b_mesh->labelcolor(FL_WHITE); b_mesh->value(1);
    b_mesh->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showLagrangian = ((Fl_Check_Button*)w)->value(); }, &g_state);

    auto b_fold = new Fl_Check_Button(180, y, 160, 25, "Caustic Fold Line");
    b_fold->labelcolor(FL_WHITE); b_fold->value(1);
    b_fold->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showPhaseSpace = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Live Hamiltonian Flow");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();

    SheafView* view = new SheafView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.time_accumulator += 0.02f;
            g_state.p.prop_z = 6.0f * sinf(g_state.time_accumulator * 0.5f);
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
