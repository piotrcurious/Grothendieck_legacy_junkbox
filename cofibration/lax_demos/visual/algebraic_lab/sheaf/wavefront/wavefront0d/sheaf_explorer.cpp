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
// Rygorystyczny Model Hamiltonowski Sieci NaCl (Przestrzeń Spektralna)
// -----------------------------------------------------------------------------
struct CrystalPhysics {
    double a = 5.64;           // Stała sieci (Angstrom)
    double V_Na = 4.5;         // Głębokość studni sodu
    double V_Cl = 3.2;         // Głębokość studni chloru
    double hbar = 1.0;         // Jednostki naturalne
    double mass = 1.0;         // Masa efektywna
};

struct WavePacket {
    double k0 = 0.0;           // Średni pęd (Bloch k)
    double sigma = 1.5;        // Szerokość pakietu (Coherent state width)
    double wavelength = 0.5;   // Skala de Broglie'a
};

struct AppState {
    float rotX = 25.0f, rotY = -40.0f;
    CrystalPhysics physics;
    WavePacket packet;
    
    float z_depth = 0.0f;      // Parametr ewolucji (czasoprzestrzenny)
    float pauli_mixing = 0.5f; // Mieszanie spinorowe (Zakaz Pauliego)
    bool show_potential = true;
    bool animate = false;
    float time = 0.0f;
};

// -----------------------------------------------------------------------------
// Silnik Propagacji Widmowej (Split-Step Sheaf Propagator)
// -----------------------------------------------------------------------------
class SpectralSheafEngine {
public:
    struct SheafSection {
        double x;
        Complex psi;           // Lokalna amplituda prawdopodobieństwa
        Complex spinor[2];     // Stan wewnętrzny (SU(2) bundle)
        double berry_phase;    // Zakumulowana faza geometryczna
    };

    // Operator Potencjału V(x) dla NaCl (Model Dual-Basis)
    double V(double x, const CrystalPhysics& p) {
        double G = 2.0 * PI / p.a;
        // NaCl to dwie podsieci FCC przesunięte o a/2
        // Używamy funkcji wygładzającej (Gaussian-like) dla rygoru analitycznego
        return -p.V_Na * exp(-0.5 * pow(sin(G * x / 2.0), 2)) 
               - p.V_Cl * exp(-0.5 * pow(sin(G * (x - p.a/2.0) / 2.0), 2));
    }

    // Propagator Unitarny: U(dz) = exp(-i H dz / hbar)
    SheafSection propagate(double x0, double k_init, const AppState& state) {
        SheafSection s;
        s.x = x0;
        
        // Inicjalizacja pakietu falowego (Section of the sheaf)
        double env = exp(-(x0 * x0) / (2.0 * state.packet.sigma * state.packet.sigma));
        s.psi = Complex(env * cos(k_init * x0), env * sin(k_init * x0));
        
        // Inicjalizacja spinora (Pauli Exclusion State)
        s.spinor[0] = Complex(cos(x0 * (double)state.pauli_mixing), 0);
        s.spinor[1] = Complex(sin(x0 * (double)state.pauli_mixing), 0);
        s.berry_phase = 0.0;

        double dz = 0.1;
        int steps = std::abs((int)(state.z_depth / (float)dz));
        double actual_dz = (state.z_depth >= 0) ? dz : -dz;

        for (int i = 0; i < steps; ++i) {
            // 1. Krok Potencjalny (Faza lokalna)
            double pot = V(s.x, state.physics);
            double dS_pot = -pot * actual_dz / state.physics.hbar;
            s.psi *= std::polar(1.0, dS_pot);

            // 2. Krok Kinetyczny (Dyfrakcja i transport pędu)
            double v = k_init / state.physics.mass; 
            s.x += v * actual_dz;
            double dS_kin = 0.5 * state.physics.mass * v * v * actual_dz / state.physics.hbar;
            s.psi *= std::polar(1.0, dS_kin);

            // 3. Ewolucja Topologiczna (Spinor Wilson Line)
            double d_berry = (double)state.pauli_mixing * pot * actual_dz;
            s.berry_phase += d_berry;
            
            Complex u = s.spinor[0];
            s.spinor[0] = u * cos(d_berry) - s.spinor[1] * sin(d_berry);
            s.spinor[1] = u * sin(d_berry) + s.spinor[1] * cos(d_berry);
        }

        return s;
    }
};

// -----------------------------------------------------------------------------
// Renderer: Wizualizacja Przestrzeni Stanów
// -----------------------------------------------------------------------------
class SheafGLView : public Fl_Gl_Window {
    AppState* state;
    SpectralSheafEngine engine;
public:
    SheafGLView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.005f, 0.005f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(80, 50, 100, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if (state->show_potential) {
            glBegin(GL_LINES);
            glColor4f(0.2f, 0.3f, 0.5f, 0.3f);
            for(float x = -20; x <= 20; x += 0.5f) {
                glVertex3f(x*4, -10, -30); glVertex3f(x*4, -10, 30);
                glVertex3f(-80, -10, x*1.5f); glVertex3f(80, -10, x*1.5f);
            }
            glEnd();
        }

        int num_k = 12;     
        int num_x = 150;    
        
        for (int j = 0; j < num_k; ++j) {
            double k_val = -1.5 + (3.0 * j / num_k);
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= num_x; ++i) {
                double x_val = -10.0 + (20.0 * i / num_x);
                auto res = engine.propagate(x_val, k_val, *state);
                
                double mag = std::abs(res.psi);
                double phase = std::arg(res.psi);
                float spin_ratio = (float)std::abs(res.spinor[0]);

                glColor4f(spin_ratio, 0.5f + 0.5f*sin(phase), 1.0f - spin_ratio, (float)mag * 0.8f);
                glVertex3f((float)res.x * 5.0f, (float)(res.psi.real() * 10.0 + res.berry_phase * 2.0), (float)(res.psi.imag() * 10.0 + k_val * 15.0));
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
// Interfejs Użytkownika
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Rygorystyczny Propagator Snopów NaCl");
    
    int ui_w = 380;
    Fl_Group* ui = new Fl_Group(0, 0, ui_w, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(15, 15, 22));

    Fl_Box* title = new Fl_Box(20, 25, ui_w-40, 40, "KWANTOWY MODEL SPEKTRALNY");
    title->labelcolor(FL_WHITE); title->labelfont(FL_BOLD); title->labelsize(18);

    int y = 80;
    // Poprawiona lambda obsługująca oba typy wskaźników (float i double)
    auto slider = [&](const char* l, float min, float max, float def, auto* p) {
        auto s = new Fl_Value_Slider(20, y+25, ui_w-40, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ 
            using T = std::remove_pointer_t<decltype(p)>;
            *(T*)d = (T)((Fl_Value_Slider*)w)->value(); 
        }, (void*)p);
        y += 60; return s;
    };

    slider("Głębokość Ewolucji (z)", 0.0f, 15.0f, 0.0f, &g_state.z_depth);
    slider("Stała Sieci NaCl (a)", 3.0f, 10.0f, 5.64f, &g_state.physics.a);
    slider("Potencjał Na+ (V_Na)", 0.0f, 10.0f, 4.5f, &g_state.physics.V_Na);
    slider("Potencjał Cl- (V_Cl)", 0.0f, 10.0f, 3.2f, &g_state.physics.V_Cl);
    
    y += 10;
    slider("Mieszanie Spinowe (Pauli)", 0.0f, 2.0f, 0.5f, &g_state.pauli_mixing);
    slider("Szerokość Pakietu (Sigma)", 0.5f, 5.0f, 1.5f, &g_state.packet.sigma);

    auto cb_pot = new Fl_Check_Button(20, y, 200, 25, "Widok Siatki Krystalicznej");
    cb_pot->labelcolor(FL_WHITE); cb_pot->value(1);
    cb_pot->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->show_potential = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto cb_anim = new Fl_Check_Button(20, y, 200, 25, "Dynamika Unitarnego Snopa");
    cb_anim->labelcolor(FL_WHITE); cb_anim->value(0);
    cb_anim->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    ui->end();
    SheafGLView* view = new SheafGLView(ui_w, 0, 1280-ui_w, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafGLView* gv = (SheafGLView*)v;
        if(g_state.animate) {
            g_state.time += 0.02f;
            g_state.z_depth = 7.5f + 7.5f * sinf(g_state.time * 0.4f);
        }
        gv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
