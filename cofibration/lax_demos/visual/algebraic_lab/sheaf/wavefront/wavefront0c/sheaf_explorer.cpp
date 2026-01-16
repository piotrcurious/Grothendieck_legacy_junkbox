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
#include <iostream>

static constexpr float PI = 3.14159265359f;
using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Rygorystyczny Model Hamiltonianu Sieci NaCl (Baza FCC)
// -----------------------------------------------------------------------------
struct CrystalLattice {
    float a = 5.64f;            // Stała sieci [Angstrom]
    float V_Na = 3.2f;          // Potencjał jądra Na+
    float V_Cl = 2.1f;          // Potencjał jądra Cl-
    float mass_eff = 1.0f;      // Masa efektywna (m*)
};

struct QuantumSource {
    float lambda = 0.45f;       // Długość fali de Broglie'a
    float beam_width = 1.2f;    // Szerokość pakietu falowego
    float coherence = 1.0f;     // Stopień spójności sekcji
};

struct AppState {
    float rotX = 22.0f, rotY = -38.0f;
    CrystalLattice lattice;
    QuantumSource source;
    
    float evolution_z = 0.0f;   // Parametr ewolucji unitarnej
    float berry_phase_scale = 0.5f; // Skala fazy Berry'ego
    bool draw_points = true;
    bool simulation_active = false;
    float time_step = 0.0f;
};

// -----------------------------------------------------------------------------
// Silnik Mikrolokalny: Integrator Symplektyczny i Transport Berry'ego
// -----------------------------------------------------------------------------
class RigorousQuantumEngine {
public:
    struct PhaseSpacePoint {
        float x, p;             // Współrzędne w T*X (wiązka kstyczna)
        double S;               // Akcja klasyczna (Faza eikonalna)
        Complex spinor[2];      // Sekcja wiązki spinorowej SU(2)
        double amplitude;       // Wyznacznik Van Vlecka (Gęstość snopa)
    };

    // Obliczenie potencjału V(x) jako sumy wkładów od podsieci Na i Cl
    // Modeluje strukturę typu NaCl (dwie sieci FCC)
    double get_potential(double x, const CrystalLattice& c) {
        double k_recip = 2.0 * PI / c.a;
        // Superpozycja potencjałów kationów i anionów przesuniętych o a/2
        double v_na = c.V_Na * cos(k_recip * x);
        double v_cl = c.V_Cl * cos(k_recip * (x - c.a * 0.5));
        return v_na + v_cl;
    }

    // Siła krystaliczna: F = -nabla(V)
    double get_force(double x, const CrystalLattice& c) {
        double k_recip = 2.0 * PI / c.a;
        double f_na = k_recip * c.V_Na * sin(k_recip * x);
        double f_cl = k_recip * c.V_Cl * sin(k_recip * (x - c.a * 0.5));
        return f_na + f_cl;
    }

    // Symplektyczny integrator 4-tego rzędu (Forest-Ruth / Suzuki-Trotter)
    // Zapewnia zachowanie formy symplektycznej dx ^ dp
    void step_symplectic(PhaseSpacePoint& pt, double dz, const CrystalLattice& c) {
        static const double w1 = 1.0 / (2.0 - pow(2.0, 1.0/3.0));
        static const double w0 = 1.0 - 2.0 * w1;
        static const double d[3] = { w1, w0, w1 };
        static const double c_step[4] = { w1/2.0, (w1+w0)/2.0, (w1+w0)/2.0, w1/2.0 };

        for(int i=0; i<3; ++i) {
            // Krok Położenia
            pt.x += c_step[i] * (pt.p / c.mass_eff) * dz;
            // Krok Pędu
            double f = get_force(pt.x, c);
            pt.p += d[i] * f * dz;
            // Aktualizacja Akcji S (Integral Lagranżjanu L = p*v - H)
            double V = get_potential(pt.x, c);
            double L = (0.5 * pt.p * pt.p / c.mass_eff) - V;
            pt.S += L * d[i] * dz;
        }
        pt.x += c_step[3] * (pt.p / c.mass_eff) * dz;
    }

    PhaseSpacePoint evolve(float x_init, float p_init, const AppState& st) {
        PhaseSpacePoint pt;
        pt.x = x_init;
        pt.p = p_init;
        pt.S = 0.0;
        pt.amplitude = exp(-(x_init * x_init) / (st.source.beam_width * st.source.beam_width));
        
        // Inicjalizacja spinora w stanie mieszanym
        pt.spinor[0] = Complex(pt.amplitude, 0);
        pt.spinor[1] = Complex(0, 0);

        double z_total = st.evolution_z;
        double dz = 0.05;
        int steps = std::abs((int)(z_total / dz));
        double actual_dz = (z_total >= 0) ? dz : -dz;

        for(int i=0; i<steps; ++i) {
            step_symplectic(pt, actual_dz, st.lattice);

            // Transport Berry'ego: Ewolucja spinora wzdłuż krzywizny potencjału
            // dPsi = -i (H_spin . dz) Psi
            double b_field = pt.p * st.berry_phase_scale;
            Complex u = pt.spinor[0];
            Complex v = pt.spinor[1];
            pt.spinor[0] = u * cos(b_field * actual_dz) - v * sin(b_field * actual_dz);
            pt.spinor[1] = u * sin(b_field * actual_dz) + v * cos(b_field * actual_dz);
        }

        return pt;
    }
};

// -----------------------------------------------------------------------------
// Renderer: Wizualizacja Manifoldu Lagrangianowskiego
// -----------------------------------------------------------------------------
class QuantumLatticeView : public Fl_Gl_Window {
    AppState* state;
    RigorousQuantumEngine engine;
public:
    QuantumLatticeView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw_lattice_structure() {
        float a = state->lattice.a;
        glPointSize(5.0f);
        glBegin(GL_POINTS);
        for(float x = -18; x <= 18; x += a) {
            for(float z = -18; z <= 18; z += a) {
                glColor4f(0.5f, 0.6f, 1.0f, 0.3f); // Jony Na+
                glVertex3f(x, -5, z);
                glColor4f(0.4f, 1.0f, 0.4f, 0.3f); // Jony Cl-
                glVertex3f(x + a*0.5f, -5, z + a*0.5f);
            }
        }
        glEnd();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.005f, 0.008f, 0.012f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(42, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(70, 45, 90, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if(state->draw_points) draw_lattice_structure();

        // Generowanie snopa sekcji w przestrzeni fazowej
        int res_x = 110, res_p = 10;
        float x_range = 8.0f;

        for(int j=0; j<res_p; j++) {
            float p_init = -1.2f + (2.4f * j / res_p);
            glBegin(GL_LINE_STRIP);
            for(int i=0; i<=res_x; i++) {
                float x_init = -x_range + (2.0f * x_range * i / res_x);
                auto p = engine.evolve(x_init, p_init, *state);
                
                // Obliczenie fazy kwantowej: exp(i * S / h_bar)
                double phase = (p.S * 2.0 * PI) / state->source.lambda;
                float color_mod = (float)sin(phase) * 0.5f + 0.5f;
                float spin_mod = (float)std::abs(p.spinor[0]);

                glColor4f(spin_mod, color_mod, 1.0f - spin_mod, 0.7f);
                // Wizualizacja w układzie: Położenie (X), Pęd (Z), Akcja (Y)
                glVertex3f(p.x * 4.0f, (float)p.S * 1.2f, p.p * 15.0f);
            }
            glEnd();
        }
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            state->rotY += (float)(Fl::event_x()-lx) * 0.4f;
            state->rotX += (float)(Fl::event_y()-ly) * 0.4f;
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

// -----------------------------------------------------------------------------
// UI i Main
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Rygorystyczny Snop Mikrolokalny: NaCl");
    
    int sidebar_w = 380;
    Fl_Group* sidebar = new Fl_Group(0, 0, sidebar_w, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(10, 12, 18));

    Fl_Box* title = new Fl_Box(20, 20, sidebar_w-40, 40, "KWANTOWY FORMALIZM NaCl");
    title->labelcolor(FL_WHITE); title->labelfont(FL_BOLD); title->labelsize(18);

    int y = 70;
    auto add_s = [&](const char* l, float min, float max, float def, float* v) {
        auto s = new Fl_Value_Slider(20, y+25, sidebar_w-40, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* p){ *(float*)p = (float)((Fl_Value_Slider*)w)->value(); }, v);
        y += 55; return s;
    };

    add_s("Ewolucja Unitarna (z)", 0.0f, 15.0f, 0.0f, &g_state.evolution_z);
    add_s("Stała Sieci a [Angstrom]", 3.0f, 8.0f, 5.64f, &g_state.lattice.a);
    add_s("Głębia Potencjału Na+", 0.0f, 6.0f, 3.2f, &g_state.lattice.V_Na);
    add_s("Głębia Potencjału Cl-", 0.0f, 6.0f, 2.1f, &g_state.lattice.V_Cl);
    
    y += 15;
    Fl_Box* b_title = new Fl_Box(20, y, sidebar_w-40, 30, "PROFIL WIĄZKI (STALKS)");
    b_title->labelcolor(fl_rgb_color(100, 150, 255)); y += 35;

    add_s("Lambda de Broglie", 0.1f, 1.2f, 0.45f, &g_state.source.lambda);
    add_s("Szerokość Pakietu (w)", 0.5f, 4.0f, 1.2f, &g_state.source.beam_width);
    add_s("Faza Berry'ego (Scaling)", 0.0f, 2.0f, 0.5f, &g_state.berry_phase_scale);

    auto cb_pts = new Fl_Check_Button(20, y, 200, 25, "Renderuj Strukturę FCC");
    cb_pts->labelcolor(FL_WHITE); cb_pts->value(1);
    cb_pts->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->draw_points = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto cb_sim = new Fl_Check_Button(20, y, 200, 25, "Symulacja Dynamiki");
    cb_sim->labelcolor(FL_WHITE); cb_sim->value(0);
    cb_sim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->simulation_active = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();
    QuantumLatticeView* view = new QuantumLatticeView(sidebar_w, 0, 1280-sidebar_w, 900, &g_state);
    
    Fl::add_idle([](void* v){
        QuantumLatticeView* gv = (QuantumLatticeView*)v;
        if(g_state.simulation_active) {
            g_state.time_step += 0.015f;
            g_state.evolution_z = 7.5f + 7.5f * sinf(g_state.time_step * 0.4f);
        }
        gv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
