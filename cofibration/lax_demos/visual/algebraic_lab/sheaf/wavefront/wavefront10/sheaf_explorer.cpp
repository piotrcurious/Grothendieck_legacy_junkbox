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
// Rygorystyczny Formalizm Topologiczny i Kinetyczny
// -----------------------------------------------------------------------------
struct PhysicsEngine {
    // Parametry fundamentalne oddziaływania
    double berry_curvature = 1.0;  // Krzywizna snopa (topologia)
    double scattering_rate = 0.05; // Jądro kolizyjne (transfer energii)
    double mass_renorm = 1.0;      // Efektywna masa polaronowa
    double decoherence_lvl = 0.1;  // Dekoherencja kwantowa
};

struct AppState {
    float rotX = 20.0f, rotY = -35.0f;
    PhysicsEngine engine;
    
    float evolution_param = 0.0f;  // Parametr t (czasoprzestrzeń)
    float wave_k = 1.0f;           // Pęd centralny pakietu
    bool show_topology = true;
    bool auto_evolve = false;
    float global_clock = 0.0f;
};

// -----------------------------------------------------------------------------
// Propagator w Przestrzeni Snopów (Sheaf Propagator)
// -----------------------------------------------------------------------------
class SheafKineticSolver {
public:
    struct SheafPoint {
        double x, y, z;
        double phase;
        double amplitude;
        double topology_index; // Lokalna gęstość miary Cherna
    };

    // Oblicza ewolucję sekcji snopa uwzględniając poprawki topologiczne
    SheafPoint compute_section(double k_base, double s_param, const AppState& st) {
        SheafPoint p;
        
        // 1. Zmodyfikowana relacja dyspersji (E = k^2/2m + Berry Term)
        double k_eff = k_base * st.engine.mass_renorm;
        double omega = (k_eff * k_eff) / 2.0;
        
        // 2. Wpływ topologii na trajektorię (Anomalous Velocity)
        // v = dE/dk + (k x BerryCurvature)
        double velocity = k_eff + st.engine.berry_curvature * sin(k_base);
        double t = (double)st.evolution_param;
        
        p.x = velocity * t + s_param; 
        
        // 3. Ewolucja fazy z uwzględnieniem transportu równoległego w snopie
        // Faza Berry'ego akumuluje się wzdłuż sekcji
        double berry_phase = st.engine.berry_curvature * k_base * t;
        p.phase = (omega * t) + berry_phase;
        
        // 4. Dekoherencja i transfer energii do termicznego snopa fononowego
        // Tłumienie wynika z jądra kolizyjnego (scattering)
        p.amplitude = exp(-st.engine.scattering_rate * t) * exp(-(s_param * s_param) / 2.0); // Koperta pakietu
        
        p.y = p.amplitude * cos(p.phase) * 10.0;
        p.z = k_base * 15.0; // Rozwarstwienie snopa względem pędu
        
        // 5. Lokalny invariant topologiczny (kodowanie kolorem)
        p.topology_index = st.engine.berry_curvature * cos(k_base * t);
        
        return p;
    }
};

// -----------------------------------------------------------------------------
// Wizualizacja 3D (Manifold & Sections)
// -----------------------------------------------------------------------------
class SheafSpaceView : public Fl_Gl_Window {
    AppState* state;
    SheafKineticSolver solver;
public:
    SheafSpaceView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1.0, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(100, 60, 120, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Siatka bazowa przestrzeni Etale
        glBegin(GL_LINES);
        glColor4f(0.2f, 0.2f, 0.4f, 0.3f);
        for(int i = -10; i <= 10; i++) {
            glVertex3f(i*10.0f, -20, -50); glVertex3f(i*10.0f, -20, 50);
            glVertex3f(-100, -20, i*5.0f); glVertex3f(100, -20, i*5.0f);
        }
        glEnd();

        int k_stalks = 20; // Liczba warstw snopa (stalks)
        int x_steps = 100; // Rozdzielczość sekcji

        for (int j = 0; j < k_stalks; j++) {
            double k = -2.0 + (4.0 * j / k_stalks);
            glBegin(GL_LINE_STRIP);
            for (int i = 0; i <= x_steps; i++) {
                double s = -8.0 + (16.0 * i / x_steps);
                auto pt = solver.compute_section(k, s, *state);
                
                // Kolorowanie: R-topologia, G-faza, B-koherencja
                float r = (float)(0.5 + 0.5 * pt.topology_index);
                float g = (float)(0.5 + 0.5 * sin(pt.phase));
                float b = (float)pt.amplitude;
                
                glColor4f(r, g, b, (float)pt.amplitude);
                glVertex3f((float)pt.x * 5.0f, (float)pt.y, (float)pt.z);
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
// Interfejs Modelowania Snopów
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Formalny Model Kinetyczny Snopów");
    
    int sidebar = 360;
    Fl_Group* ui = new Fl_Group(0, 0, sidebar, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(18, 18, 25));

    Fl_Box* header = new Fl_Box(20, 25, sidebar-40, 40, "KWANTOWY FORMALIZM SNOPÓW");
    header->labelcolor(FL_WHITE); header->labelfont(FL_BOLD); header->labelsize(16);

    int y = 80;
    auto add_s = [&](const char* l, float min, float max, float def, auto* v) {
        auto s = new Fl_Value_Slider(20, y+25, sidebar-40, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ 
            using T = std::remove_pointer_t<decltype(v)>;
            *(T*)d = (T)((Fl_Value_Slider*)w)->value(); 
        }, (void*)v);
        y += 65; return s;
    };

    add_s("Parametr Ewolucji (t)", 0.0f, 20.0f, 0.0f, &g_state.evolution_param);
    add_s("Krzywizna Berry'ego (Topologia)", -5.0f, 5.0f, 1.0f, &g_state.engine.berry_curvature);
    add_s("Współczynnik Rozpraszania (Scattering)", 0.0f, 0.5f, 0.05f, &g_state.engine.scattering_rate);
    add_s("Renormalizacja Masy (m*)", 0.1f, 3.0f, 1.0f, &g_state.engine.mass_renorm);

    y += 20;
    auto cb_auto = new Fl_Check_Button(20, y, 200, 25, "Automatyczna Ewolucja");
    cb_auto->labelcolor(FL_WHITE); cb_auto->value(0);
    cb_auto->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->auto_evolve = ((Fl_Check_Button*)w)->value(); }, &g_state);

    ui->end();
    SheafSpaceView* view = new SheafSpaceView(sidebar, 0, 1280-sidebar, 900, &g_state);
    
    Fl::add_idle([](void* v){
        SheafSpaceView* gv = (SheafSpaceView*)v;
        if(g_state.auto_evolve) {
            g_state.global_clock += 0.02f;
            g_state.evolution_param = 10.0f + 10.0f * sinf(g_state.global_clock * 0.3f);
        }
        gv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
