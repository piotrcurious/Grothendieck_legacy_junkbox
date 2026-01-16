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
// Parametry Fizyczne Kryształu NaCl i Wiązki
// -----------------------------------------------------------------------------
struct NaClCrystal {
    float lattice_constant = 5.64f; // Stała sieci NaCl w angstremach (skalowana)
    float ionic_potential = 2.0f;   // Głębokość dołka potencjału Na/Cl
    float damping = 0.1f;           // Tłumienie fononowe
};

struct LightSource {
    float wavelength = 0.5f;        // Długość fali (lambda)
    float waist_w0 = 1.0f;          // Talia wiązki Gaussa
    float initial_phase = 0.0f;     // Faza początkowa (Gouy)
};

struct AppState {
    float rotX = 25.0f, rotY = -45.0f;
    NaClCrystal crystal;
    LightSource source;
    
    float prop_z = 0.0f;            // Głębokość penetracji kryształu
    float spin_interaction = 0.5f;  // Sprzężenie spin-orbita w sieci
    bool show_wavefunction = true;
    bool animate = false;
    float time = 0.0f;
};

// -----------------------------------------------------------------------------
// Silnik Snopów Kwantowych (Kwadratura Fazy Stacjonarnej)
// -----------------------------------------------------------------------------
class NaClQuantumEngine {
public:
    struct LocalSection {
        float x, p, S;
        Complex psi;                // Lokalna wartość funkcji falowej
        float spinor[2];            // Składowe spinowe (Pauli)
    };

    // Obliczenie potencjału sieci NaCl: V(x) = V0 * sin(G*x)
    // Modeluje periodyczność krystalograficzną
    float get_nacl_potential(float x, const NaClCrystal& c) {
        float G = 2.0f * PI / c.lattice_constant;
        return c.ionic_potential * (sinf(G * x) - cosf(G * x * 0.5f));
    }

    // Warunek początkowy: Sekcja snopa jako wiązka Gaussa
    LocalSection initialize_section(float x0, const LightSource& s, float spin_param) {
        // Amplituda Gaussa: exp(-x^2 / w0^2)
        float amplitude = expf(-(x0 * x0) / (s.waist_w0 * s.waist_w0));
        float phase = s.initial_phase;
        
        // Pęd początkowy wynika z gradientu fazy (eikonał)
        float p0 = 0.0f; 

        // Inicjalizacja spinora (zakaz Pauliego jako orientacja w SU(2))
        float s_up = cosf(x0 * spin_param);
        float s_dn = sinf(x0 * spin_param);

        return {x0, p0, phase, Complex(amplitude, 0), {s_up, s_dn}};
    }

    // Propagacja przez strukturę krystaliczną (Algorytm Symplektyczny + Faza Maslova)
    LocalSection propagate(float x_start, float p_offset, const AppState& state) {
        LocalSection ls = initialize_section(x_start, state.source, state.spin_interaction);
        ls.p += p_offset;

        float z_target = state.prop_z;
        float dz = 0.05f;
        int steps = std::abs((int)(z_target / dz));
        float step_sign = (z_target > 0) ? 1.0f : -1.0f;

        for (int i = 0; i < steps; ++i) {
            // Hamiltonian: H = p^2/2m + V(x)
            // 1. Krok pędu (oddziaływanie z jonami Na+ / Cl-)
            float force = - (2.0f * PI / state.crystal.lattice_constant) * state.crystal.ionic_potential * cosf(2.0f * PI * ls.x / state.crystal.lattice_constant);
            
            ls.p += force * 0.5f * dz * step_sign;
            
            // 2. Krok położenia
            ls.x += ls.p * dz * step_sign;
            
            // 3. Drugi krok pędu (Verlet)
            force = - (2.0f * PI / state.crystal.lattice_constant) * state.crystal.ionic_potential * cosf(2.0f * PI * ls.x / state.crystal.lattice_constant);
            ls.p += force * 0.5f * dz * step_sign;

            // 4. Akcja (S) i Ewolucja Fazy (Lagranżjan)
            // L = p*x_dot - H = p^2/2 - V(x)
            float pot = get_nacl_potential(ls.x, state.crystal);
            float lagrangian = 0.5f * ls.p * ls.p - pot;
            ls.S += lagrangian * dz * step_sign;

            // 5. Zakaz Pauliego (Rotacja spinora w polu sieci)
            float theta = ls.p * state.spin_interaction * dz;
            float s0 = ls.spinor[0];
            ls.spinor[0] = s0 * cosf(theta) - ls.spinor[1] * sinf(theta);
            ls.spinor[1] = s0 * sinf(theta) + ls.spinor[1] * cosf(theta);
        }

        // Konstrukcja końcowej funkcji falowej (psi = A * exp(iS))
        float k = 2.0f * PI / state.source.wavelength;
        ls.psi = Complex(cos(ls.S * k), sin(ls.S * k)) * (double)expf(-(ls.x*ls.x)/10.0f);

        return ls;
    }
};

// -----------------------------------------------------------------------------
// Interfejs Graficzny i Wizualizacja Snopa
// -----------------------------------------------------------------------------
class CrystalView : public Fl_Gl_Window {
    AppState* state;
    NaClQuantumEngine engine;
public:
    CrystalView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw_lattice() {
        glBegin(GL_POINTS);
        for(float x = -20; x <= 20; x += state->crystal.lattice_constant) {
            for(float z = -20; z <= 20; z += state->crystal.lattice_constant) {
                glColor4f(0.8, 0.8, 1.0, 0.3); // Na+
                glVertex3f(x*2, -10, z*2);
                glColor4f(0.5, 1.0, 0.5, 0.3); // Cl-
                glVertex3f((x + state->crystal.lattice_constant/2)*2, -10, z*2);
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
        glClearColor(0.01f, 0.01f, 0.03f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(80, 50, 120, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        draw_lattice();

        // Renderowanie Przestrzeni Etalnej (Etale Space)
        int x_res = 100, p_res = 15;
        float ap = 5.0f; // Apertura

        for(int j=0; j<p_res; j++) {
            float p_off = -1.0f + (2.0f * j / p_res);
            glBegin(GL_LINE_STRIP);
            for(int i=0; i<=x_res; i++) {
                float x_start = -ap + (2.0f * ap * i / x_res);
                auto res = engine.propagate(x_start, p_off, *state);
                
                // Kolor zależy od fazy kwantowej i spinu
                float phase_col = (float)res.psi.real() * 0.5f + 0.5f;
                float spin_col = res.spinor[0] * 0.5f + 0.5f;
                
                glColor4f(spin_col, phase_col, 1.0f - spin_col, 0.6f);
                glVertex3f(res.x * 5, res.S * 2, res.p * 10);
            }
            glEnd();
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
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "NaCl Microlocal Sheaf Engine");
    
    int sw = 380;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 900);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(20, 20, 25));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "SIEĆ KRYSTALICZNA NaCl");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(20);

    int y = 70;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv, float* target) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* ptr){ *(float*)ptr = (float)((Fl_Value_Slider*)w)->value(); }, target);
        y += 60; return s;
    };

    create_s("Penetracja Sieci (z)", 0.0f, 15.0f, 0.0f, &g_state.prop_z);
    create_s("Stała Sieci (Lattice a)", 1.0f, 10.0f, 5.64f, &g_state.crystal.lattice_constant);
    create_s("Potencjał Jonowy (V0)", 0.0f, 5.0f, 2.0f, &g_state.crystal.ionic_potential);
    
    y += 10;
    Fl_Box* lbox = new Fl_Box(20, y, sw-40, 30, "PARAMETRY WIĄZKI");
    lbox->labelcolor(fl_rgb_color(150,150,255)); y += 35;

    create_s("Długość fali (lambda)", 0.1f, 2.0f, 0.5f, &g_state.source.wavelength);
    create_s("Talia wiązki (w0)", 0.5f, 4.0f, 1.0f, &g_state.source.waist_w0);
    create_s("Oddziaływanie Spinora", 0.0f, 2.0f, 0.5f, &g_state.spin_interaction);

    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Animacja Fononowa");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    sidebar->end();
    CrystalView* view = new CrystalView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        CrystalView* cv = (CrystalView*)v;
        if(g_state.animate) {
            g_state.time += 0.02f;
            g_state.prop_z = 7.5f + 7.5f * sinf(g_state.time * 0.5f);
        }
        cv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
