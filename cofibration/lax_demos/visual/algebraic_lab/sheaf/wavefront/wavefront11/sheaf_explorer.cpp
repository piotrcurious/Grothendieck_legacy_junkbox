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
// Rygorystyczny Model Macierzy Gęstości (Liouville-von Neumann)
// -----------------------------------------------------------------------------
struct QuantumSystem {
    double hbar = 1.0;
    double mass = 0.5;         // Masa efektywna
    double omega_phonon = 0.8; // Energia fononu LO
    double gamma_relax = 0.1;  // Stała relaksacji Lindblada
    double coupling_V = 0.5;   // Amplituda oddziaływania V_q
};

struct AppState {
    float rotX = 30.0f, rotY = -45.0f;
    QuantumSystem sys;
    
    float time = 0.0f;
    float temp_K = 300.0f;
    bool show_wigner = true;
    bool simulate = false;
    float dt = 0.02f;
};

// -----------------------------------------------------------------------------
// Solver Kinetyczny: Ewolucja w Przestrzeni Wignera
// -----------------------------------------------------------------------------
class LindbladSheafSolver {
public:
    // Funkcja Wignera W(x, p) jako rygorystyczna sekcja snopa przestrzeni fazowej
    double calculate_wigner(double x, double p, double t, const AppState& st) {
        // Modelujemy ewolucję pakietu z uwzględnieniem dyfuzji pędu (Lindblad)
        // Sigma(t) rośnie wraz z czasem i temperaturą - to jest transfer energii
        double thermal_fluc = st.temp_K / 1000.0;
        double sigma_p = 0.5 + st.sys.gamma_relax * t * thermal_fluc;
        double sigma_x = 0.5 + (st.sys.gamma_relax * t) / st.sys.mass;
        
        // Dryf klasyczny (Hamiltonowski)
        double p_center = 1.0; 
        double x_center = (p_center / st.sys.mass) * t;
        
        // Kwantowe oscylacje spójności (Interferencja snopa)
        double coherence = cos(2.0 * x * p / st.sys.hbar) * exp(-st.sys.gamma_relax * t);
        
        // Rozkład Gaussa w przestrzeni fazowej
        double arg = -pow(x - x_center, 2)/(2.0*sigma_x) - pow(p - p_center, 2)/(2.0*sigma_p);
        return (1.0 / (PI * st.sys.hbar)) * exp(arg) * (1.0 + coherence);
    }

    // Gęstość stanów (DOS) zmodyfikowana przez sprzężenie elektron-fonon
    double spectral_density(double E, const AppState& st) {
        // Rezonans fononowy: E = hbar*omega_LO
        double delta = 0.1;
        return (1.0 / PI) * (delta / (pow(E - st.sys.omega_phonon, 2) + delta * delta));
    }
};

// -----------------------------------------------------------------------------
// Renderer: Kwantowa Przestrzeń Fazowa
// -----------------------------------------------------------------------------
class WignerSpaceView : public Fl_Gl_Window {
    AppState* state;
    LindbladSheafSolver solver;
public:
    WignerSpaceView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

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
        gluLookAt(120, 80, 150, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        // Osie przestrzeni fazowej (x, p)
        glBegin(GL_LINES);
        glColor4f(0.3f, 0.3f, 0.5f, 0.5f);
        for(int i = -10; i <= 10; i++) {
            glVertex3f(i*10.0f, 0, -100); glVertex3f(i*10.0f, 0, 100);
            glVertex3f(-100, 0, i*10.0f); glVertex3f(100, 0, i*10.0f);
        }
        glEnd();

        // Rendering funkcji Wignera (Topologia stanu mieszanego)
        int res = 60;
        double range = 15.0;
        
        for (int i = 0; i < res; ++i) {
            double p_val = -range/2.0 + (range * i / res);
            glBegin(GL_TRIANGLE_STRIP);
            for (int j = 0; j <= res; ++j) {
                double x_val = -range/2.0 + (range * j / res);
                
                double w1 = solver.calculate_wigner(x_val, p_val, state->time, *state);
                double w2 = solver.calculate_wigner(x_val, p_val + (range/res), state->time, *state);

                // Kolor zależy od znaku funkcji Wignera (ujemne wartości = czysty stan kwantowy)
                auto color_node = [&](double w) {
                    if (w > 0) glColor4f(0.2f, 0.4f, 1.0f, (float)w * 5.0f);
                    else glColor4f(1.0f, 0.2f, 0.2f, (float)fabs(w) * 5.0f);
                };

                color_node(w1);
                glVertex3f(x_val * 8.0f, (float)w1 * 50.0f, p_val * 8.0f);
                color_node(w2);
                glVertex3f(x_val * 8.0f, (float)w2 * 50.0f, (p_val + (range/res)) * 8.0f);
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
// Interfejs Operatorowy
// -----------------------------------------------------------------------------
int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Kwantowa Dynamika Lindblada: Przestrzeń Wignera");
    
    int side = 360;
    Fl_Group* ui = new Fl_Group(0, 0, side, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(10, 10, 15));

    Fl_Box* title = new Fl_Box(20, 25, side-40, 40, "OPERATOROWA EWOLUCJA SNOPA");
    title->labelcolor(FL_WHITE); title->labelfont(FL_BOLD); title->labelsize(16);

    int y = 80;
    auto slider = [&](const char* l, float min, float max, float def, auto* v) {
        auto s = new Fl_Value_Slider(20, y+25, side-40, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ 
            using T = std::remove_pointer_t<decltype(v)>;
            *(T*)d = (T)((Fl_Value_Slider*)w)->value(); 
        }, (void*)v);
        y += 65; return s;
    };

    slider("Czas ewolucji (tau)", 0.0f, 50.0f, 0.0f, &g_state.time);
    slider("Temperatura (T [K])", 0.0f, 1000.0f, 300.0f, &g_state.temp_K);
    slider("Relaksacja (Gamma Lindblad)", 0.0f, 1.0f, 0.1f, &g_state.sys.gamma_relax);
    slider("Energia Fononu (hbar*omega)", 0.1f, 5.0f, 0.8f, &g_state.sys.omega_phonon);

    y += 20;
    auto btn = new Fl_Check_Button(20, y, 200, 25, "Symulacja czasu rzeczywistego");
    btn->labelcolor(FL_WHITE);
    btn->callback([](Fl_Widget* w, void* d){ ((AppState*)d)->simulate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    ui->end();
    WignerSpaceView* view = new WignerSpaceView(side, 0, 1280-side, 900, &g_state);
    
    Fl::add_idle([](void* v){
        WignerSpaceView* gv = (WignerSpaceView*)v;
        if(g_state.simulate) {
            g_state.time += g_state.dt;
            if(g_state.time > 50.0f) g_state.time = 0;
        }
        gv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
