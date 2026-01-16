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
// Model Fizyczny: Hamiltonian Sieci i Geometria Snopa
// -----------------------------------------------------------------------------
struct CrystalDynamics {
    float a = 5.64f;            // Stała sieci [Angstrom]
    float V0_Na = 2.5f;         // Amplituda potencjału kationu
    float V0_Cl = 1.8f;         // Amplituda potencjału anionu
    float mass_eff = 1.0f;      // Masa efektywna nośnika
};

struct LightJet {
    float lambda = 0.5f;        // Długość fali
    float w0 = 0.8f;            // Minimalna talia wiązki
    float zR;                   // Długość Rayleigha (obliczana)
};

struct AppState {
    float rotX = 20.0f, rotY = -40.0f;
    CrystalDynamics crystal;
    LightJet source;
    
    float z_depth = 0.0f;       // Współrzędna ewolucji (z)
    float pauli_coupling = 0.3f;// Siła oddziaływania spin-orbita
    bool show_lattice = true;
    bool animate = false;
    float t = 0.0f;
};

// -----------------------------------------------------------------------------
// Silnik Mikrolokalny: Integrator Form Różniczkowych
// -----------------------------------------------------------------------------
class NaClMicroEngine {
public:
    struct FiberState {
        float x, p, S;          // Współrzędne w wiązce 1-jetów
        Complex spinor[2];      // Lokalna sekcja snopa spinorowego
    };

    // Obliczenie potencjału metodą sumy harmonicznych sieci odwrotnej
    // V(x) = V_Na * cos(G*x) + V_Cl * cos(G*x/2) dla NaCl
    float compute_V(float x, const CrystalDynamics& c) {
        float G = 2.0f * PI / c.a;
        return c.V0_Na * cosf(G * x) + c.V0_Cl * cosf(G * x * 0.5f);
    }

    // Gradient potencjału (siła działająca na pęd p)
    float compute_dVdx(float x, const CrystalDynamics& c) {
        float G = 2.0f * PI / c.a;
        return -G * (c.V0_Na * sinf(G * x) + 0.5f * c.V0_Cl * sinf(G * x * 0.5f));
    }

    // Inicjalizacja wiązki Gaussa w przestrzeni fazowej
    FiberState init_beam(float x0, float p0_ext, const LightJet& b) {
        float amp = expf(-(x0 * x0) / (b.w0 * b.w0));
        // Spinor zainicjowany w stanie własnym Pauli-Z
        return {x0, p0_ext, 0.0f, {Complex(amp, 0), Complex(0, 0)}};
    }

    // Ewolucja symplektyczna (Lie-Hamilton Flow)
    FiberState solve_jet(float x_init, float p_init, const AppState& state) {
        FiberState fs = init_beam(x_init, p_init, state.source);
        
        float dz = 0.04f;
        int steps = std::abs((int)(state.z_depth / dz));
        float sgn = (state.z_depth > 0) ? 1.0f : -1.0f;

        for (int i = 0; i < steps; ++i) {
            // Hamiltonian: H(x,p) = p^2/2m + V(x)
            // Krok 1: Ewolucja pędu (p = p - dV/dx * dz/2)
            fs.p -= compute_dVdx(fs.x, state.crystal) * 0.5f * dz * sgn;
            
            // Krok 2: Ewolucja położenia (x = x + p/m * dz)
            fs.x += (fs.p / state.crystal.mass_eff) * dz * sgn;
            
            // Krok 3: Ponowna ewolucja pędu (Verlet)
            fs.p -= compute_dVdx(fs.x, state.crystal) * 0.5f * dz * sgn;

            // Krok 4: Całkowanie akcji (S) wzdłuż formy p dx - H dz
            float V = compute_V(fs.x, state.crystal);
            float L = (0.5f * fs.p * fs.p / state.crystal.mass_eff) - V;
            fs.S += L * dz * sgn;

            // Krok 5: Ewolucja Spinora (Zakaz Pauliego jako rotacja fazy Berry'ego)
            // Rotacja SU(2) indukowana przez gradient potencjału
            float omega = fs.p * state.pauli_coupling * dz;
            Complex u = fs.spinor[0];
            fs.spinor[0] = cos(omega)*fs.spinor[0] - sin(omega)*fs.spinor[1];
            fs.spinor[1] = sin(omega)*u + cos(omega)*fs.spinor[1];
        }

        return fs;
    }
};

// -----------------------------------------------------------------------------
// Renderer Sceny Krystalograficznej
// -----------------------------------------------------------------------------
class NaClView : public Fl_Gl_Window {
    AppState* state;
    NaClMicroEngine engine;
public:
    NaClView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw_lattice_nodes() {
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        float a = state->crystal.a;
        for(float x = -15; x <= 15; x += a) {
            for(float z = -15; z <= 15; z += a) {
                // Na+ (Sód)
                glColor4f(0.6f, 0.7f, 1.0f, 0.4f);
                glVertex3f(x*2, -8, z*2);
                // Cl- (Chlor)
                glColor4f(0.4f, 1.0f, 0.4f, 0.4f);
                glVertex3f((x + a/2.0f)*2, -8, z*2);
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
        glClearColor(0.01f, 0.015f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(40, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(85, 55, 110, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if(state->show_lattice) draw_lattice_nodes();

        // Rendering Przestrzeni Etalnej Snopa (Sheaf Etale Space)
        int sections = 120, stalks = 12;
        float ap = 6.0f; // Apertura wiązki

        for(int j=0; j<stalks; j++) {
            float p_bias = -0.8f + (1.6f * j / stalks);
            glBegin(GL_LINE_STRIP);
            for(int i=0; i<=sections; i++) {
                float x0 = -ap + (2.0f * ap * i / sections);
                auto res = engine.solve_jet(x0, p_bias, *state);
                
                // Mapowanie fazy kwantowej (S) i spinu na model kolorów
                float phase_vis = sinf(res.S * (2.0f * PI / state->source.lambda));
                float spin_vis = std::abs(res.spinor[0]);
                
                glColor4f(0.2f + spin_vis*0.8f, 0.4f + phase_vis*0.4f, 1.0f - spin_vis, 0.6f);
                // Osie: X (Położenie), Y (Akcja/Faza), Z (Pęd/Płaszczyzna fazowa)
                glVertex3f(res.x * 5, res.S * 1.5f, res.p * 12);
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

// -----------------------------------------------------------------------------
// Główny Interfejs Sterujący
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Mikrolokalny Model NaCl - Przestrzeń Snopów");
    
    int sw = 380;
    Fl_Group* ui = new Fl_Group(0, 0, sw, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(18, 20, 24));

    Fl_Box* header = new Fl_Box(20, 20, sw-40, 40, "HAMILTONIAN NaCl (100)");
    header->labelcolor(FL_WHITE); header->labelfont(FL_BOLD); header->labelsize(20);

    int y = 70;
    auto add_slider = [&](const char* lbl, float min, float max, float def, float* ptr) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* p){ *(float*)p = (float)((Fl_Value_Slider*)w)->value(); }, ptr);
        y += 60; return s;
    };

    add_slider("Penetracja Kryształu (z)", 0.0f, 12.0f, 0.0f, &g_state.z_depth);
    add_slider("Stała Sieci (a) [A]", 3.0f, 10.0f, 5.64f, &g_state.crystal.a);
    add_slider("Potencjał Na+ (V_Na)", 0.0f, 5.0f, 2.5f, &g_state.crystal.V0_Na);
    add_slider("Potencjał Cl- (V_Cl)", 0.0f, 5.0f, 1.8f, &g_state.crystal.V0_Cl);
    
    y += 10;
    Fl_Box* sub = new Fl_Box(20, y, sw-40, 30, "GEOMETRIA WIĄZKI WEJŚCIOWEJ");
    sub->labelcolor(fl_rgb_color(140, 160, 255)); y += 35;

    add_slider("Długość fali (lambda)", 0.1f, 1.5f, 0.5f, &g_state.source.lambda);
    add_slider("Talia Wiązki (w0)", 0.2f, 3.0f, 0.8f, &g_state.source.w0);
    add_slider("Sprzężenie Pauliego", 0.0f, 1.5f, 0.3f, &g_state.pauli_coupling);

    auto cb_lat = new Fl_Check_Button(20, y, 200, 25, "Pokaż Sieć Jonową");
    cb_lat->labelcolor(FL_WHITE); cb_lat->value(1);
    cb_lat->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->show_lattice = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto cb_anim = new Fl_Check_Button(20, y, 200, 25, "Oscylacje Termiczne");
    cb_anim->labelcolor(FL_WHITE); cb_anim->value(0);
    cb_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    ui->end();
    NaClView* view = new NaClView(sw, 0, 1280-sw, 900, &g_state);
    
    Fl::add_idle([](void* v){
        NaClView* cv = (NaClView*)v;
        if(g_state.animate) {
            g_state.t += 0.03f;
            // Symulacja "pływania" potencjału (fonony)
            g_state.z_depth = 6.0f + 6.0f * sinf(g_state.t * 0.4f);
        }
        cv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
