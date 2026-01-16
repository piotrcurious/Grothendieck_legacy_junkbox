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
#include <complex>

static constexpr float PI = 3.14159265359f;
using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Spinor & Interface Physics Structs
// -----------------------------------------------------------------------------
struct PhysicsParams {
    // Initial Wavefront (The "Section")
    float defocus = 0.5f;     
    float coma = 0.0f;        
    float spherical = 0.1f;   

    // Pauli Exclusion / Spinor Layer
    float spin_orbit_coupling = 0.0f; // Mixes the p and S components via Pauli matrices
    float pauli_repulsion = 0.0f;     // Self-energy barrier in the etale space

    // Interface & Boundary
    float interface_pos = 0.0f;       // X-location of the dielectric boundary
    float refractive_index_n2 = 1.5f; // "Gluing" constant between sheaf segments
    
    // Global Dynamics
    float prop_z = 0.0f;
    float aperture = 3.5f;
    bool compute_interference = false;
};

struct AppState {
    float rotX = 20.0f, rotY = -35.0f;
    PhysicsParams p;
    bool animate = false;
    float anim_t = 0.0f;
};

// -----------------------------------------------------------------------------
// The Pauli-Fresnel Sheaf Integrator
// -----------------------------------------------------------------------------
class QuantumSheafEngine {
public:
    struct QuantumState {
        float x, p, S;
        float intensity;
        Complex spinor_up, spinor_down; // 2-component wavefunction
    };

    // Calculate the initial state with Pauli Spinor initialization
    QuantumState get_initial(float x0, const PhysicsParams& p) {
        float S = (p.defocus * 0.5f * x0 * x0) + (p.coma * 0.33f * x0 * x0 * x0);
        float p_val = (p.defocus * x0) + (p.coma * x0 * x0);
        
        // Spinor initialization (Pauli Exclusion Sheaf)
        // Represents the local "orientation" of the state in SU(2)
        float theta = x0 * p.spin_orbit_coupling;
        Complex up = Complex(cos(theta/2.0), 0);
        Complex down = Complex(sin(theta/2.0), 0);

        return {x0, p_val, S, 1.0f, up, down};
    }

    // Evolution through Interface and Hamiltonian Flow
    QuantumState evolve(float x0, float p_offset, const PhysicsParams& params) {
        QuantumState qs = get_initial(x0, params);
        qs.p += p_offset;

        float z = params.prop_z;
        float dt = 0.1f;
        int steps = std::abs((int)(z / dt));
        float direction = (z > 0) ? 1.0f : -1.0f;

        for (int i = 0; i < steps; ++i) {
            // 1. Check for Interface "Gluing" (Fresnel boundary)
            // If cross interface, change p (refraction) and adjust S
            float x_prev = qs.x;
            qs.x += qs.p * dt * direction;

            if ((x_prev < params.interface_pos && qs.x >= params.interface_pos) ||
                (x_prev > params.interface_pos && qs.x <= params.interface_pos)) {
                // Snell's Law in Phase Space: p_out = sqrt(n^2 - (1 - p_in^2))
                float p_in = qs.p;
                float sign = (p_in > 0) ? 1.0f : -1.0f;
                qs.p = sign * sqrtf(std::max(0.01f, params.refractive_index_n2 * params.refractive_index_n2 - (1.0f - p_in * p_in)));
                
                // Pauli Repulsion barrier: Add phase jump if "Spin" is aligned
                qs.S += params.pauli_repulsion * std::abs(qs.spinor_up.real());
            }

            // 2. Hamiltonian Flow
            // dS = L dt = (p*x_dot - H) dt = (p^2/2) dt
            qs.S += (0.5f * qs.p * qs.p) * dt * direction;

            // 3. Spinor Rotation (Spin-Orbit Coupling)
            float d_theta = qs.p * params.spin_orbit_coupling * dt;
            Complex old_up = qs.spinor_up;
            qs.spinor_up = cos(d_theta)*qs.spinor_up - sin(d_theta)*qs.spinor_down;
            qs.spinor_down = sin(d_theta)*old_up + cos(d_theta)*qs.spinor_down;
        }

        return qs;
    }
};

// -----------------------------------------------------------------------------
// Visualizer
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    AppState* state;
    QuantumSheafEngine engine;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(80, 60, 100, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Draw Interface Plane
        glBegin(GL_QUADS);
        glColor4f(1, 1, 1, 0.05f);
        float ix = state->p.interface_pos * 6.0f;
        glVertex3f(ix, -50, -50); glVertex3f(ix, 50, -50);
        glVertex3f(ix, 50, 50);   glVertex3f(ix, -50, 50);
        glEnd();

        // Render the Etale Space
        int x_res = 80, p_res = 20;
        float ap = state->p.aperture;
        
        glBegin(GL_TRIANGLES);
        for(int i=0; i<x_res; i++) {
            float x1 = -ap + (2.0f*ap*i/x_res);
            float x2 = -ap + (2.0f*ap*(i+1)/x_res);
            for(int j=0; j<p_res; j++) {
                float po1 = -1.5f + (3.0f*j/p_res);
                float po2 = -1.5f + (3.0f*(j+1)/p_res);

                auto s1 = engine.evolve(x1, po1, state->p);
                auto s2 = engine.evolve(x2, po1, state->p);
                auto s3 = engine.evolve(x2, po2, state->p);
                auto s4 = engine.evolve(x1, po2, state->p);

                // Color by Wavefunction Component (Spin-Up = Blue, Spin-Down = Green)
                float up_mag = std::abs(s1.spinor_up);
                float dn_mag = std::abs(s1.spinor_down);
                
                if (state->p.compute_interference) {
                    float phase = sinf(s1.S * 4.0f);
                    glColor4f(up_mag, 0.5f + 0.5f*phase, dn_mag, 0.25f);
                } else {
                    glColor4f(0.2f + up_mag*0.5f, 0.4f, 0.2f + dn_mag*0.8f, 0.2f);
                }

                glVertex3f(s1.x*6, s1.S*4, s1.p*12); 
                glVertex3f(s2.x*6, s2.S*4, s2.p*12);
                glVertex3f(s3.x*6, s3.S*4, s3.p*12);
                
                glVertex3f(s1.x*6, s1.S*4, s1.p*12);
                glVertex3f(s3.x*6, s3.S*4, s3.p*12);
                glVertex3f(s4.x*6, s4.S*4, s4.p*12);
            }
        }
        glEnd();
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
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Pauli-Fresnel Sheaf Engine");
    
    int sw = 380;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(15, 15, 20));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "SPINOR SHEAF MODEL");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(20);

    int y = 70;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 60; return s;
    };

    create_s("Propagate (z)", -10.0f, 10.0f, 0.0f, &g_state.p.prop_z);
    create_s("Spin-Orbit Coupling", 0.0f, 2.0f, 0.0f, &g_state.p.spin_orbit_coupling);
    create_s("Pauli Repulsion (S-Shift)", 0.0f, 5.0f, 0.0f, &g_state.p.pauli_repulsion);
    
    y += 10;
    Fl_Box* sep = new Fl_Box(20, y, sw-40, 1); sep->box(FL_FLAT_BOX); sep->color(FL_DARK3);
    y += 20;

    create_s("Interface Position (X)", -3.0f, 3.0f, 0.0f, &g_state.p.interface_pos);
    create_s("Dielectric index (n2)", 1.0f, 2.5f, 1.5f, &g_state.p.refractive_index_n2);
    create_s("Z4: Defocus", -2.0f, 2.0f, 0.5f, &g_state.p.defocus);
    create_s("Aperture", 1.0f, 5.0f, 3.5f, &g_state.p.aperture);

    auto b_int = new Fl_Check_Button(20, y, 200, 25, "Synthesize Wavefunction");
    b_int->labelcolor(FL_WHITE); b_int->value(0);
    b_int->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->p.compute_interference = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Animate Field");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();
    SheafView* view = new SheafView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.anim_t += 0.02f;
            g_state.p.prop_z = 8.0f * sinf(g_state.anim_t * 0.3f);
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
