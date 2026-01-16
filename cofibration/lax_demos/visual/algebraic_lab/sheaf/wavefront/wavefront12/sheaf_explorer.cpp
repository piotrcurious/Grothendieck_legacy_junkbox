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
#include <algorithm>

static constexpr float PI = 3.14159265359f;
using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Rygorystyczny Model Kwantowy: Macierz Gęstości i Operator Lindblada
// -----------------------------------------------------------------------------
struct SimulationSpace {
    static const int N = 64; // Rozmiar bazy (siatka dyskretna)
    double L = 20.0;         // Rozmiar pudła [j.a.]
    double dx = L / N;
    double dk = 2.0 * PI / L;
    
    std::vector<std::vector<Complex>> rho; // Macierz gęstości rho(x, x')

    SimulationSpace() {
        rho.assign(N, std::vector<Complex>(N, 0.0));
        // Inicjalizacja: czysty stan - pakiet falowy Gaussa
        double x0 = -4.0, p0 = 2.0, sigma = 1.2;
        std::vector<Complex> psi(N);
        for(int i=0; i<N; ++i) {
            double x = -L/2.0 + i*dx;
            double env = exp(-pow(x-x0,2)/(4.0*sigma*sigma));
            psi[i] = Complex(env * cos(p0*x), env * sin(p0*x));
        }
        // rho = |psi><psi|
        for(int i=0; i<N; ++i)
            for(int j=0; j<N; ++j)
                rho[i][j] = psi[i] * std::conj(psi[j]);
    }
};

struct PhysicsParams {
    double m_eff = 0.5;      // Masa efektywna (NaCl)
    double omega_LO = 1.2;   // Energia fononu LO
    double gamma = 0.05;     // Siła sprzężenia (Lindblad)
    double dt = 0.05;
};

// -----------------------------------------------------------------------------
// Solver: Integracja Równania Master (SPO + Lindblad)
// -----------------------------------------------------------------------------
class MasterEquationSolver {
public:
    void step(SimulationSpace& s, const PhysicsParams& p) {
        // 1. Ewolucja Hamiltonianowa (Część kinetyczna w przestrzeni pędów)
        // U(dt) = exp(-i H dt / hbar)
        for(int i=0; i<s.N; ++i) {
            for(int j=0; j<s.N; ++j) {
                double ki = (i < s.N/2) ? i*s.dk : (i-s.N)*s.dk;
                double kj = (j < s.N/2) ? j*s.dk : (j-s.N)*s.dk;
                double Ei = (ki*ki)/(2.0*p.m_eff);
                double Ej = (kj*kj)/(2.0*p.m_eff);
                s.rho[i][j] *= std::polar(1.0, -(Ei - Ej) * p.dt);
            }
        }

        // 2. Człon Lindblada (Dyssypacja i Transfer Energii)
        // d(rho)/dt = gamma * (L rho L+ - 0.5{L+L, rho})
        // Modelujemy relaksację pędu do stanu podstawowego (emisja fononu)
        for(int i=0; i<s.N; ++i) {
            for(int j=0; j<s.N; ++j) {
                // Tłumienie wyrazów pozadiagonalnych (dekoherencja)
                if(i != j) s.rho[i][j] *= exp(-p.gamma * p.dt);
                // Relaksacja energii (uproszczony operator skoku w przestrzeni energii)
                if(i == j && i > 0) {
                    double decay = s.rho[i][i].real() * p.gamma * p.dt;
                    s.rho[i][i] -= decay;
                    s.rho[0][0] += decay; // Transfer do stanu o najniższej energii
                }
            }
        }
    }

    // Bezpośrednie obliczenie funkcji Wignera: W(x, p) = FT[ rho(x+y/2, x-y/2) ]
    double get_wigner(const SimulationSpace& s, double x_idx, double p_val) {
        Complex res = 0;
        int xi = (int)x_idx;
        for(int dy = -s.N/2; dy < s.N/2; ++dy) {
            int i1 = (xi + dy + s.N) % s.N;
            int i2 = (xi - dy + s.N) % s.N;
            double y = dy * s.dx;
            res += s.rho[i1][i2] * std::polar(1.0, -2.0 * p_val * y);
        }
        return res.real() / s.N;
    }
};

// -----------------------------------------------------------------------------
// Renderer 3D
// -----------------------------------------------------------------------------
class RigorousWignerView : public Fl_Gl_Window {
    SimulationSpace space;
    PhysicsParams params;
    MasterEquationSolver solver;
    float rotX = 35, rotY = -45;
public:
    RigorousWignerView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {}

    void update_sim() {
        solver.step(space, params);
        redraw();
    }

    void set_gamma(float g) { params.gamma = g; }

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
        gluLookAt(100, 70, 130, 0, 0, 0, 0, 1, 0);
        glRotatef(rotX, 1, 0, 0);
        glRotatef(rotY, 0, 1, 0);

        // Osie Przestrzeni Fazowej
        glBegin(GL_LINES);
        glColor4f(0.2f, 0.3f, 0.5f, 0.6f);
        glVertex3f(-60, 0, 0); glVertex3f(60, 0, 0);
        glVertex3f(0, 0, -60); glVertex3f(0, 0, 60);
        glEnd();

        // Siatka Funkcji Wignera
        int res = space.N;
        for(int i = 0; i < res - 1; ++i) {
            glBegin(GL_TRIANGLE_STRIP);
            for(int j = 0; j < res; ++j) {
                double p = (j - res/2) * space.dk;
                double w1 = solver.get_wigner(space, i, p);
                double w2 = solver.get_wigner(space, i+1, p);

                auto color = [](double w) {
                    if(w > 0) glColor4f(0.2f, 0.5f, 1.0f, (float)w * 10.0f);
                    else glColor4f(1.0f, 0.2f, 0.3f, (float)fabs(w) * 10.0f);
                };

                color(w1);
                glVertex3f((i - res/2)*2.0f, (float)w1*80.0f, (float)p*25.0f);
                color(w2);
                glVertex3f((i+1 - res/2)*2.0f, (float)w2*80.0f, (float)p*25.0f);
            }
            glEnd();
        }
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            rotY += (Fl::event_x()-lx); rotX += (Fl::event_y()-ly);
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

int main() {
    Fl_Double_Window* win = new Fl_Double_Window(1280, 900, "Rygorystyczny Solver Lindblada (NaCl Polaron)");
    
    RigorousWignerView* view = new RigorousWignerView(300, 0, 980, 900);
    
    Fl_Group* ui = new Fl_Group(0, 0, 300, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(20, 20, 30));
    
    Fl_Box* lbl = new Fl_Box(10, 20, 280, 40, "SOLVER MACIERZY GESTOSCI");
    lbl->labelcolor(FL_WHITE); lbl->labelfont(FL_BOLD);

    auto s1 = new Fl_Value_Slider(20, 100, 260, 20, "Sprzężenie Elektron-Fonon (Gamma)");
    s1->type(FL_HOR_NICE_SLIDER); s1->bounds(0, 0.5); s1->value(0.05);
    s1->labelcolor(FL_WHITE); s1->callback([](Fl_Widget* w, void* d){
        ((RigorousWignerView*)d)->set_gamma(((Fl_Value_Slider*)w)->value());
    }, view);

    ui->end();

    Fl::add_idle([](void* d){ ((RigorousWignerView*)d)->update_sim(); }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
