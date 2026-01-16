#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Button.H>
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
// Spectral Split-Operator Dynamics (Unconditionally Stable Kinetics)
// -----------------------------------------------------------------------------
struct QuantumSystem {
    static const int N = 64; 
    double L = 25.0;         
    double dx = L / N;
    
    std::vector<std::vector<Complex>> rho_initial;
    std::vector<std::vector<Complex>> rho;

    void initialize(double x0, double p0, double sigma) {
        rho_initial.assign(N, std::vector<Complex>(N, 0.0));
        std::vector<Complex> psi(N);
        double norm = 0;
        for(int i=0; i<N; ++i) {
            double x = -L/2.0 + i*dx;
            double env = exp(-pow(x-x0,2)/(4.0*sigma*sigma));
            psi[i] = Complex(env * cos(p0*x), env * sin(p0*x));
            norm += std::norm(psi[i]);
        }
        norm = sqrt(norm * dx);
        for(int i=0; i<N; i++) psi[i] /= norm;

        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
                rho_initial[i][j] = psi[i] * std::conj(psi[j]);
        
        rho = rho_initial;
    }

    QuantumSystem() { initialize(-5.0, 2.5, 1.5); }
};

class SpectralLvNSolver {
public:
    void transform(std::vector<std::vector<Complex>>& data, bool forward) {
        int N = data.size();
        std::vector<std::vector<Complex>> out = data;
        double sign = forward ? -1.0 : 1.0;
        double invN = forward ? 1.0 : 1.0/N;

        for(int i=0; i<N; i++) {
            for(int k=0; k<N; k++) {
                Complex sum = 0;
                for(int j=0; j<N; j++) {
                    sum += data[i][j] * std::polar(1.0, sign * 2.0 * PI * k * j / N);
                }
                out[i][k] = sum * invN;
            }
        }
        data = out;
    }

    void step(QuantumSystem& s, double dt, double m_eff, double V_latt, double gamma) {
        int N = s.N;
        
        // 1. Position Space Step
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                double xi = -s.L/2.0 + i*s.dx;
                double xj = -s.L/2.0 + j*s.dx;
                double Vi = V_latt * cos(2.0 * PI * xi / 5.64); 
                double Vj = V_latt * cos(2.0 * PI * xj / 5.64);
                s.rho[i][j] *= std::polar(1.0, -(Vi - Vj) * dt);
                if (i != j) s.rho[i][j] *= exp(-gamma * pow(xi - xj, 2) * dt);
            }
        }

        // 2. Momentum Space Step
        transform(s.rho, true); 
        for(int k1=0; k1<N; k1++) {
            for(int k2=0; k2<N; k2++) {
                double p1 = (k1 < N/2) ? k1 : k1 - N;
                double p2 = (k2 < N/2) ? k2 : k2 - N;
                double dp = 2.0 * PI / s.L;
                double e1 = pow(p1 * dp, 2) / (2.0 * m_eff);
                double e2 = pow(p2 * dp, 2) / (2.0 * m_eff);
                s.rho[k1][k2] *= std::polar(1.0, -(e1 - e2) * dt);
            }
        }
        transform(s.rho, false);

        // 3. Normalization
        double trace = 0;
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) s.rho[i][j] = (s.rho[i][j] + std::conj(s.rho[j][i])) * 0.5;
            trace += s.rho[i][i].real();
        }
        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++) s.rho[i][j] /= (trace * s.dx);
    }

    double get_wigner(const QuantumSystem& s, int x_idx, double p) {
        Complex sum = 0;
        int N = s.N;
        for(int dy = -N/4; dy < N/4; dy++) {
            int i1 = (x_idx + dy + N) % N;
            int i2 = (x_idx - dy + N) % N;
            double window = 0.5 * (1.0 + cos(2.0 * PI * dy / (N/2.0)));
            sum += s.rho[i1][i2] * std::polar(window, -2.0 * p * (dy * s.dx));
        }
        return sum.real() / N;
    }
};

class PhysicalWignerView : public Fl_Gl_Window {
public:
    QuantumSystem sys;
    SpectralLvNSolver solver;
    float rotX = 25, rotY = -35;
    float gamma = 0.005f, V0 = 1.0f, m = 1.0f;
    float currentTime = 0.0f, prevTime = 0.0f;
    float p0 = 2.0f, sigma = 1.2f, x0 = -6.0f;
    bool running = false;

    PhysicalWignerView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {}

    void reset_simulation() {
        sys.initialize(x0, p0, sigma);
        currentTime = 0; prevTime = 0; redraw();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.003f, 0.003f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1, 1000);
        gluLookAt(130, 100, 150, 0, 0, 0, 0, 1, 0);
        glRotatef(rotX, 1, 0, 0); glRotatef(rotY, 0, 1, 0);

        if(running) {
            solver.step(sys, 0.02, m, V0, gamma);
            currentTime += 0.02f;
            prevTime = currentTime;
        } else if (std::abs(currentTime - prevTime) > 0.01f) {
            sys.rho = sys.rho_initial;
            for(double t=0; t<currentTime; t+=0.02) solver.step(sys, 0.02, m, V0, gamma);
            prevTime = currentTime;
        }

        int N = sys.N;
        double dk = 2.0 * PI / sys.L;
        for(int i=0; i<N-1; i++) {
            glBegin(GL_TRIANGLE_STRIP);
            for(int j=0; j<N; j++) {
                double p_val = (j - N/2) * dk;
                double w1 = solver.get_wigner(sys, i, p_val);
                double w2 = solver.get_wigner(sys, i+1, p_val);
                auto color = [](double w) {
                    float a = std::min(1.0f, (float)fabs(w)*30.0f);
                    if(w > 0) glColor4f(0.2f, 0.5f, 1.0f, a);
                    else glColor4f(1.0f, 0.1f, 0.3f, a);
                };
                color(w1); glVertex3f((i-N/2)*3.0f, (float)w1*180.0f, (float)p_val*35.0f);
                color(w2); glVertex3f((i+1-N/2)*3.0f, (float)w2*180.0f, (float)p_val*35.0f);
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

struct AppCtx { 
    PhysicalWignerView* view; 
    Fl_Value_Slider* slider; 
};

int main() {
    Fl_Double_Window* win = new Fl_Double_Window(1400, 900, "Stable Spectral Solver (NaCl)");
    PhysicalWignerView* view = new PhysicalWignerView(350, 0, 1050, 900);
    
    Fl_Group* ui = new Fl_Group(0, 0, 350, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(10, 10, 20));

    int y = 50;
    auto sld = [&](const char* l, float min, float max, float def, float* v) {
        auto s = new Fl_Value_Slider(20, y, 310, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ *(float*)d = (float)((Fl_Value_Slider*)w)->value(); }, v);
        y += 60; return s;
    };

    auto t_sld = sld("Time (τ)", 0.0f, 10.0f, 0.0f, &view->currentTime);
    sld("Lattice V0", 0.0f, 10.0f, 1.0f, &view->V0);
    sld("Decoherence γ", 0.0f, 0.1f, 0.005f, &view->gamma);
    sld("Mass m*", 0.5f, 3.0f, 1.0f, &view->m);
    sld("Momentum p0", -5.0f, 5.0f, 2.0f, &view->p0);
    sld("Width σ", 0.5f, 3.0f, 1.2f, &view->sigma);

    auto play = new Fl_Check_Button(20, y, 100, 25, "Play");
    play->labelcolor(FL_WHITE);
    play->callback([](Fl_Widget* w, void* d){ ((PhysicalWignerView*)d)->running = ((Fl_Check_Button*)w)->value(); }, view);
    
    auto init = new Fl_Button(130, y, 180, 25, "Re-Init Source");
    init->callback([](Fl_Widget* w, void* d){ ((PhysicalWignerView*)d)->reset_simulation(); }, view);

    ui->end();
    static AppCtx ctx = { view, t_sld };
    Fl::add_idle([](void* d){ 
        auto c = (AppCtx*)d; 
        c->view->redraw();
        if(c->view->running) c->slider->value(c->view->currentTime);
    }, &ctx);

    win->resizable(view); win->show();
    return Fl::run();
}
