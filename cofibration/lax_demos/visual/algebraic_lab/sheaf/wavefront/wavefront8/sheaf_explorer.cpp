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
#include <algorithm>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Mathematical Constants & Structs
// -----------------------------------------------------------------------------
struct MicrolocalParams {
    // Initial Section (Wavefront)
    float c_defocus = 0.5f;     
    float c_astig   = 0.0f;     
    float c_coma    = 0.0f;     
    float c_sphere  = 0.1f;     
    float c_trefoil = 0.0f;     

    // Dynamics (Hamiltonian)
    float prop_z = 0.0f;        
    float kerr_gamma = 0.0f;    
    float lattice_k = 1.0f;     
    float maslov_bias = 1.0f;   // Phase-sensitive weighting

    // Global Geometry
    float aperture = 3.0f;
    float fiber_width = 1.5f;   // Momentum range
    float lattice_phase = 0.0f;
};

struct AppState {
    float rotX = 22.0f, rotY = -35.0f;
    MicrolocalParams p;
    bool showEtale = true;
    bool showCaustics = true;
    bool showInterference = false; // New: Wigner-like density
    bool animate = false;
    float anim_t = 0.0f;
};

// -----------------------------------------------------------------------------
// Symplectic Integrator with Phase Tracking
// -----------------------------------------------------------------------------
class SheafIntegrator {
public:
    struct State {
        float x, p, S; 
        int maslov_index; // Counts caustic crossings
    };

    // Initial 1-jet + Initial Maslov state
    State get_start(float x0, float p_offset, const MicrolocalParams& p) {
        float S = (p.c_defocus * 0.5f * x0 * x0) + 
                  (p.c_astig   * 0.5f * x0 * x0 * sinf(x0)) + 
                  (p.c_coma    * (1.0f/3.0f) * powf(x0, 3.0f)) + 
                  (p.c_sphere  * 0.25f * powf(x0, 4.0f)) +
                  (p.c_trefoil * powf(x0, 3.0f) * cosf(3.0f * x0));

        float p_init = (p.c_defocus * x0) + 
                       (p.c_astig   * (x0 * sinf(x0) + 0.5f * x0 * x0 * cosf(x0))) +
                       (p.c_coma    * x0 * x0) + 
                       (p.c_sphere  * powf(x0, 3.0f)) +
                       (p.c_trefoil * (3.0f * x0 * x0 * cosf(3.0f * x0) - 3.0f * powf(x0, 3.0f) * sinf(3.0f * x0)));

        return {x0, p_init + p_offset, S, 0};
    }

    // Step-wise Hamiltonian Evolution
    State evolve(float x0, float p_off, const MicrolocalParams& params) {
        State s = get_start(x0, p_off, params);
        if (params.prop_z == 0) return s;

        float dt = 0.05f;
        float total_z = params.prop_z;
        int steps = std::abs((int)(total_z / dt));
        float step = (total_z > 0) ? dt : -dt;

        for(int i = 0; i < steps; ++i) {
            float old_p = s.p;
            
            // 1. Half-step Momentum (dV/dx)
            float force = params.kerr_gamma * params.lattice_k * sinf(params.lattice_k * s.x + params.lattice_phase);
            s.p -= 0.5f * force * step;
            
            // 2. Full-step Position
            s.x += s.p * step;
            
            // 3. Second half-step Momentum
            force = params.kerr_gamma * params.lattice_k * sinf(params.lattice_k * s.x + params.lattice_phase);
            s.p -= 0.5f * force * step;

            // 4. Action Update (S = integral of Lagrangian)
            float V = -params.kerr_gamma * cosf(params.lattice_k * s.x + params.lattice_phase);
            s.S += (0.5f * s.p * s.p - V) * step;

            // 5. Track Maslov Index (Sign changes in p-divergence/focusing)
            // A simple proxy: check if momentum flips relative to expansion
            if (old_p * s.p < 0 && std::abs(s.p) < 0.01f) {
                s.maslov_index++;
            }
        }
        return s;
    }
};

// -----------------------------------------------------------------------------
// Visualizer
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    AppState* state;
    SheafIntegrator engine;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw_axis() {
        glBegin(GL_LINES);
        glColor4f(1, 1, 1, 0.2f);
        glVertex3f(-100, 0, 0); glVertex3f(100, 0, 0);
        glVertex3f(0, -100, 0); glVertex3f(0, 100, 0);
        glVertex3f(0, 0, -100); glVertex3f(0, 0, 100);
        glEnd();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.01f, 0.012f, 0.015f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(70, 60, 90, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        draw_axis();

        float ap = state->p.aperture;
        float fw = state->p.fiber_width;

        if (state->showEtale) {
            int x_res = 100, p_res = 25;
            glBegin(GL_QUADS);
            for(int i=0; i<x_res; i++) {
                float x1 = -ap + (2.0f*ap*i/x_res);
                float x2 = -ap + (2.0f*ap*(i+1)/x_res);
                for(int j=0; j<p_res; j++) {
                    float p1 = -fw + (2.0f*fw*j/p_res);
                    float p2 = -fw + (2.0f*fw*(j+1)/p_res);
                    
                    auto pt1 = engine.evolve(x1, p1, state->p);
                    auto pt2 = engine.evolve(x2, p1, state->p);
                    auto pt3 = engine.evolve(x2, p2, state->p);
                    auto pt4 = engine.evolve(x1, p2, state->p);
                    
                    // Coloring based on Maslov Index and Action phase
                    float phase = sinf(pt1.S * 2.0f);
                    if (state->showInterference) {
                        glColor4f(0.4f + 0.3f*phase, 0.6f, 1.0f, 0.15f);
                    } else {
                        float col = (pt1.maslov_index % 2 == 0) ? 0.8f : 0.4f;
                        glColor4f(0.2f, 0.5f*col, 1.0f, 0.2f);
                    }

                    glVertex3f(pt1.x*6, pt1.S*4, pt1.p*12);
                    glVertex3f(pt2.x*6, pt2.S*4, pt2.p*12);
                    glVertex3f(pt3.x*6, pt3.S*4, pt3.p*12);
                    glVertex3f(pt4.x*6, pt4.S*4, pt4.p*12);
                }
            }
            glEnd();
        }

        if (state->showCaustics) {
            // Visualize the "Skeleton" of the Sheaf (Zero-momentum seed)
            glColor3f(1.0f, 0.2f, 0.1f); glLineWidth(3.0f);
            glBegin(GL_LINE_STRIP);
            for(float x=-ap; x<=ap; x+=0.02f) {
                auto pt = engine.evolve(x, 0.0f, state->p);
                glVertex3f(pt.x*6, pt.S*4, pt.p*12);
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
// UI Construction
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Advanced Microlocal Sheaf Engine");
    
    int sw = 380;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(20, 22, 28));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "WIGNER JET-SPACE DYNAMICS");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(18);

    int y = 70;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 60; return s;
    };

    create_s("Symplectic Flow (z-prop)", -10.0f, 10.0f, 0.0f, &g_state.p.prop_z);
    create_s("Initial Defocus (Z4)", -3.0f, 3.0f, 0.5f, &g_state.p.c_defocus);
    create_s("Initial Coma (Z7)", -1.0f, 1.0f, 0.0f, &g_state.p.c_coma);
    create_s("High-Order Spherical (Z11)", -0.5f, 0.5f, 0.1f, &g_state.p.c_sphere);
    create_s("Trefoil Topology (Z9)", -1.0f, 1.0f, 0.0f, &g_state.p.c_trefoil);
    
    y += 15;
    Fl_Box* sep = new Fl_Box(20, y, sw-40, 1); sep->box(FL_FLAT_BOX); sep->color(FL_DARK3);
    y += 20;

    create_s("Hamiltonian Potential Intensity", 0.0f, 5.0f, 0.0f, &g_state.p.kerr_gamma);
    create_s("Potential Lattice Frequency", 0.1f, 10.0f, 1.0f, &g_state.p.lattice_k);
    create_s("Phase-Space Fiber Width", 0.1f, 4.0f, 1.5f, &g_state.p.fiber_width);
    create_s("Aperture Selection", 0.5f, 6.0f, 3.0f, &g_state.p.aperture);

    auto b_etale = new Fl_Check_Button(20, y, 160, 25, "Etale Manifold");
    b_etale->labelcolor(FL_WHITE); b_etale->value(1);
    b_etale->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showEtale = ((Fl_Check_Button*)w)->value(); }, &g_state);

    auto b_dens = new Fl_Check_Button(180, y, 160, 25, "Density Interference");
    b_dens->labelcolor(FL_WHITE); b_dens->value(0);
    b_dens->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showInterference = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Animate Propagation");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();

    SheafView* view = new SheafView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.anim_t += 0.02f;
            g_state.p.prop_z = 8.0f * sinf(g_state.anim_t * 0.4f);
            g_state.p.lattice_phase = g_state.anim_t;
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
