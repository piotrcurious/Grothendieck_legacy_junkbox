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
#include <random>

static constexpr float PI = 3.14159265359f;
using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Rygorystyczne Stałe Materiałowe NaCl (Zablokowane)
// -----------------------------------------------------------------------------
struct NaClConstants {
    const double a = 5.6402;      // Stała sieci [Angstrom]
    const double V_Na = 4.52;     // Potencjał jonizacji Na [eV]
    const double V_Cl = 3.61;     // Powinowactwo elektronowe Cl [eV]
    const double omega_LO = 0.032;// Energia fononu optycznego [eV]
    const double hbar = 1.0;      // Skalowane jednostki atomowe
    const double mass = 1.0;      // Masa efektywna m*
};

struct AppState {
    float rotX = 25.0f, rotY = -40.0f;
    NaClConstants crystal;
    
    float temperature = 300.0f;   // Temperatura [K] - steruje fononami
    float coupling = 0.15f;       // Siła sprzężenia elektron-fonon
    float evolution_time = 0.0f;
    bool show_lattice = true;
    bool active_transport = false;
    float global_t = 0.0f;
};

// -----------------------------------------------------------------------------
// Silnik Dynamiki Sieci: Fonony i Transfer Energii
// -----------------------------------------------------------------------------
class PhononDynamicsEngine {
    std::mt19937 gen{std::random_device{}()};
public:
    struct QuantumState {
        double x;
        Complex psi;
        double energy_transfer;    // Skumulowana energia oddana do sieci
        double phonon_displacement; // Lokalna fluktuacja sieci
    };

    // Statyczny potencjał krystaliczny NaCl
    double V_static(double x, const NaClConstants& c) {
        double G = 2.0 * PI / c.a;
        return -c.V_Na * exp(-0.5 * pow(sin(G * x / 2.0), 2)) 
               - c.V_Cl * exp(-0.5 * pow(sin(G * (x - c.a/2.0) / 2.0), 2));
    }

    // Fluktuacja fononowa (Mody LO) zależna od temperatury
    double get_phonon_field(double x, double t, float temp, const NaClConstants& c) {
        // Obsadzenie termiczne Bosego-Einsteina n = 1/(exp(hw/kT)-1)
        double k_B = 8.617e-5; // eV/K
        double n_ph = 1.0 / (exp(c.omega_LO / (k_B * temp)) - 1.0 + 1e-9);
        double amplitude = 0.05 * sqrt(n_ph + 0.5); // Amplituda drgań zerowych + termicznych
        
        // Suma modów fononowych (uproszczona do dominującego k=0 dla LO)
        return amplitude * sin(c.omega_LO * t + x * (2.0 * PI / c.a));
    }

    QuantumState propagate(double x0, double k_init, const AppState& st) {
        QuantumState s;
        s.x = x0;
        s.energy_transfer = 0.0;
        
        // Inicjalizacja lokalnej sekcji snopa
        double sigma = 1.2;
        double env = exp(-(x0 * x0) / (2.0 * sigma * sigma));
        s.psi = Complex(env * cos(k_init * x0), env * sin(k_init * x0));

        double dt = 0.05;
        int steps = std::abs((int)(st.evolution_time / dt));
        
        for (int i = 0; i < steps; ++i) {
            double current_t = i * dt;
            
            // 1. Potencjał zaburzony fononami: V(x,t) = V_stat(x + u)
            s.phonon_displacement = get_phonon_field(s.x, current_t, st.temperature, st.crystal);
            double pot = V_static(s.x + s.phonon_displacement, st.crystal);
            
            // 2. Sprzężenie elektron-fonon (transfer energii)
            // Energia przekazywana do sieci przez gradient potencjału
            double force = (V_static(s.x + 0.01, st.crystal) - V_static(s.x, st.crystal)) / 0.01;
            double dE = st.coupling * force * s.phonon_displacement * dt;
            s.energy_transfer += std::abs(dE);

            // 3. Ewolucja fazy (Unitarna + Dyssyptywna faza Berry'ego)
            double phase_shift = -(pot * dt) / st.crystal.hbar;
            s.psi *= std::polar(exp(-s.energy_transfer * 0.1), phase_shift); // Tłumienie amplitudy (transfer)

            // 4. Transport kinetyczny
            double v = (k_init / st.crystal.mass);
            s.x += v * dt;
        }

        return s;
    }
};

// -----------------------------------------------------------------------------
// Renderer 3D
// -----------------------------------------------------------------------------
class NaClPhononView : public Fl_Gl_Window {
    AppState* state;
    PhononDynamicsEngine engine;
public:
    NaClPhononView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw_lattice() {
        double a = state->crystal.a;
        glPointSize(4.0f);
        glBegin(GL_POINTS);
        for(double x = -20; x <= 20; x += a) {
            for(double z = -10; z <= 10; z += a) {
                // Na+
                glColor4f(0.6f, 0.7f, 1.0f, 0.4f);
                glVertex3f(x*3, -8, z*5);
                // Cl-
                glColor4f(0.4f, 1.0f, 0.4f, 0.4f);
                glVertex3f((x + a/2.0)*3, -8, (z + a/2.0)*5);
            }
        }
        glEnd();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.01f, 0.01f, 0.015f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(90, 60, 110, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if (state->show_lattice) draw_lattice();

        // Rendering Sekcji Snopa z uwzględnieniem fluktuacji termicznych
        int stalks = 14;
        int resolution = 140;

        for (int j = 0; j < stalks; j++) {
            double k_vec = -1.2 + (2.4 * j / stalks);
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= resolution; i++) {
                double x_init = -12.0 + (24.0 * i / resolution);
                auto res = engine.propagate(x_init, k_vec, *state);
                
                // Wizualizacja transferu energii i fazy
                double amp = std::abs(res.psi);
                double ph = std::arg(res.psi);
                
                // Kolorystyka: błękit (unitarność), czerwień (transfer energii do fononów)
                float r = 0.2f + (float)res.energy_transfer * 2.0f;
                float g = 0.4f + (float)sin(ph)*0.3f;
                float b = 0.8f - r*0.5f;

                glColor4f(r, g, b, (float)amp);
                // Oś Y reprezentuje teraz lokalne zaburzenie sieci + funkcję falową
                glVertex3f(res.x * 4, (float)(res.psi.real() * 8.0 + res.phonon_displacement * 15.0), (float)(k_vec * 20.0));
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
// UI: Kontrola Parametrów Termicznych i Dynamiki
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "NaCl: Dynamika Elektron-Fonon i Transfer Energii");
    
    int side_w = 360;
    Fl_Group* ui = new Fl_Group(0, 0, side_w, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(20, 22, 28));

    Fl_Box* t1 = new Fl_Box(20, 25, side_w-40, 40, "DYNAMIKA SIECI NaCl");
    t1->labelcolor(FL_WHITE); t1->labelfont(FL_BOLD); t1->labelsize(18);

    int y = 80;
    auto slider = [&](const char* l, float min, float max, float def, float* p) {
        auto s = new Fl_Value_Slider(20, y+25, side_w-40, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ *(float*)d = (float)((Fl_Value_Slider*)w)->value(); }, p);
        y += 65; return s;
    };

    slider("Czas ewolucji [fs]", 0.0f, 10.0f, 0.0f, &g_state.evolution_time);
    slider("Temperatura sieci [K]", 0.0f, 1000.0f, 300.0f, &g_state.temperature);
    slider("Sprzężenie Elektron-Fonon", 0.0f, 1.0f, 0.15f, &g_state.coupling);
    
    y += 20;
    Fl_Box* info = new Fl_Box(20, y, side_w-40, 60, "Parametry kryształu (a, V_Na, V_Cl)\nsą stałymi fizycznymi NaCl.");
    info->labelcolor(fl_rgb_color(160, 160, 180)); info->labelsize(12);
    y += 70;

    auto cb_lat = new Fl_Check_Button(20, y, 200, 25, "Pokaż węzły sieci");
    cb_lat->labelcolor(FL_WHITE); cb_lat->value(1);
    cb_lat->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->show_lattice = ((Fl_Check_Button*)w)->value(); }, &g_state);

    y += 40;
    auto cb_run = new Fl_Check_Button(20, y, 200, 25, "Uruchom transport energii");
    cb_run->labelcolor(FL_WHITE); cb_run->value(0);
    cb_run->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->active_transport = ((Fl_Check_Button*)w)->value(); }, &g_state);

    ui->end();
    NaClPhononView* view = new NaClPhononView(side_w, 0, 1280-side_w, 900, &g_state);
    
    Fl::add_idle([](void* v){
        NaClPhononView* gv = (NaClPhononView*)v;
        if(g_state.active_transport) {
            g_state.global_t += 0.015f;
            g_state.evolution_time = 5.0f + 5.0f * sinf(g_state.global_t * 0.5f);
        }
        gv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
