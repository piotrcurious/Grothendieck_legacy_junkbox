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
// Numerical Stability Enhancements
// -----------------------------------------------------------------------------
struct QuantumSystem {
    static const int N = 64; 
    double L = 25.0;         
    double dx = L / N;
    double dk = 2.0 * PI / L;
    
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
        // Rigorous Normalization
        norm = sqrt(norm * dx);
        for(int i=0; i<N; i++) psi[i] /= norm;

        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
                rho_initial[i][j] = psi[i] * std::conj(psi[j]);
        
        rho = rho_initial;
    }

    QuantumSystem() {
        initialize(-5.0, 2.5, 1.5);
    }
};

class NumericalLvNSolver {
public:
    // We use a stabilized RK2 (Heun's method) to prevent divergence
    void step(QuantumSystem& s, double dt, double m_eff, double V_latt, double gamma) {
        int N = s.N;
        
        auto compute_derivative = [&](const std::vector<std::vector<Complex>>& r) {
            std::vector<std::vector<Complex>> dr(N, std::vector<Complex>(N, 0.0));
            for(int i=0; i<N; i++) {
                for(int j=0; j<N; j++) {
                    double xi = -s.L/2.0 + i*s.dx;
                    double xj = -s.L/2.0 + j*s.dx;
                    
                    double Vi = V_latt * cos(2.0 * PI * xi / 5.64); 
                    double Vj = V_latt * cos(2.0 * PI * xj / 5.64);

                    // Kinetic term using a 4th-order central difference for better stability
                    auto laplacian = [&](const std::vector<std::vector<Complex>>& mat, int x, int y, bool is_row) {
                        int p2 = (x+2)%N, p1 = (x+1)%N, m1 = (x-1+N)%N, m2 = (x-2+N)%N;
                        Complex val;
                        if(is_row) val = (-mat[p2][y] + 16.0*mat[p1][y] - 30.0*mat[x][y] + 16.0*mat[m1][y] - mat[m2][y]) / (12.0 * s.dx * s.dx);
                        else val = (-mat[y][p2] + 16.0*mat[y][p1] - 30.0*mat[y][x] + 16.0*mat[y][m1] - mat[y][m2]) / (12.0 * s.dx * s.dx);
                        return val;
                    };

                    Complex kin_i = -laplacian(r, i, j, true) / (2.0 * m_eff);
                    Complex kin_j = laplacian(r, j, i, false) / (2.0 * m_eff);
                    
                    // LvN Commutator: -i[H, rho]
                    dr[i][j] = Complex(0, -1) * (kin_i - kin_j + (Vi - Vj) * r[i][j]);
                    
                    // Lindblad: Decays off-diagonals (spatial decoherence)
                    if (i != j) {
                        dr[i][j] -= gamma * pow(xi - xj, 2) * r[i][j];
                    }
                }
            }
            return dr;
        };

        // Predictor step
        auto k1 = compute_derivative(s.rho);
        std::vector<std::vector<Complex>> rho_p(N, std::vector<Complex>(N, 0.0));
        for(int i=0; i<N; i++)
            for(int j=0; j<N; j++)
                rho_p[i][j] = s.rho[i][j] + k1[i][j] * dt;

        // Corrector step
        auto k2 = compute_derivative(rho_p);
        double trace = 0;
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                s.rho[i][j] += (k1[i][j] + k2[j][i]) * 0.5 * dt;
            }
            trace += s.rho[i][i].real();
        }

        // Trace Renormalization (Crucial for stability)
        if(trace > 0) {
            for(int i=0; i<N; i++)
                for(int j=0; j<N; j++)
                    s.rho[i][j] /= (trace * s.dx);
        }
    }

    double get_wigner(const QuantumSystem& s, int x_idx, double p) {
        Complex sum = 0;
        int N = s.N;
        // Windowing to prevent alias noise
        for(int dy = -N/4; dy < N/4; dy++) {
            int i1 = (x_idx + dy + N) % N;
            int i2 = (x_idx - dy + N) % N;
            double y = dy * s.dx;
            double window = 0.5 * (1.0 + cos(2.0 * PI * dy / (N/2.0)));
            sum += s.rho[i1][i2] * std::polar(window, -2.0 * p * y);
        }
        return sum.real() / N;
    }
};

class PhysicalWignerView : public Fl_Gl_Window {
public:
    QuantumSystem sys;
    NumericalLvNSolver solver;
    float rotX = 30, rotY = -40;
    
    float gamma = 0.01f, V0 = 1.0f, m = 0.8f;
    float currentTime = 0.0f;
    float prevTime = 0.0f;
    float source_p0 = 2.0f, source_sigma = 1.2f, source_x0 = -6.0f;

    bool running = false;
    double time_step = 0.01; // Smaller time step for stability

    PhysicalWignerView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {}

    void reset_simulation() {
        sys.initialize(source_x0, source_p0, source_sigma);
        currentTime = 0.0f;
        prevTime = 0.0f;
        redraw();
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.002f, 0.002f, 0.008f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 1, 1000);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(120, 90, 140, 0, 0, 0, 0, 1, 0);
        glRotatef(rotX, 1, 0, 0); glRotatef(rotY, 0, 1, 0);

        if(running) {
            // Sub-stepping for stability: Perform 5 physics steps per frame
            for(int i=0; i<5; i++) {
                solver.step(sys, time_step, m, V0, gamma);
                currentTime += (float)time_step;
            }
            prevTime = currentTime;
        } else if (std::abs(currentTime - prevTime) > 0.01f) {
            sys.rho = sys.rho_initial;
            double t = 0;
            while(t < currentTime) {
                solver.step(sys, time_step, m, V0, gamma);
                t += time_step;
            }
            prevTime = currentTime;
        }

        // Draw phase space boundaries
        glBegin(GL_LINES);
        glColor4f(0.1f, 0.2f, 0.4f, 0.3f);
        for(float i=-80; i<=80; i+=20) {
            glVertex3f(i, 0, -80); glVertex3f(i, 0, 80);
            glVertex3f(-80, 0, i); glVertex3f(80, 0, i);
        }
        glEnd();

        int N = sys.N;
        for(int i=0; i<N-1; i++) {
            glBegin(GL_TRIANGLE_STRIP);
            for(int j=0; j<N; j++) {
                double p = (j - N/2) * sys.dk;
                double w1 = solver.get_wigner(sys, i, p);
                double w2 = solver.get_wigner(sys, i+1, p);
                
                auto color = [](double w) {
                    float intense = std::min(1.0f, (float)fabs(w)*25.0f);
                    if(w > 0) glColor4f(0.2f, 0.6f, 1.0f, intense);
                    else glColor4f(1.0f, 0.1f, 0.2f, intense);
                };

                color(w1); glVertex3f((i-N/2)*3.0f, (float)w1*150.0f, (float)p*30.0f);
                color(w2); glVertex3f((i+1-N/2)*3.0f, (float)w2*150.0f, (float)p*30.0f);
            }
            glEnd();
        }
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            rotY += (Fl::event_x()-lx)*0.5f; rotX += (Fl::event_y()-ly)*0.5f;
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

struct AppContext {
    PhysicalWignerView* view;
    Fl_Value_Slider* timeSlider;
};

int main() {
    Fl_Double_Window* win = new Fl_Double_Window(1400, 900, "Stable Lindblad Phase-Space Solver");
    PhysicalWignerView* view = new PhysicalWignerView(350, 0, 1050, 900);
    
    Fl_Group* ui = new Fl_Group(0, 0, 350, 900);
    ui->box(FL_FLAT_BOX); ui->color(fl_rgb_color(10, 10, 18));

    int y = 50;
    auto create_slider = [&](const char* l, float min, float max, float def, float* v) {
        auto s = new Fl_Value_Slider(20, y, 310, 20, l);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(min, max); s->value(def);
        s->labelcolor(FL_WHITE); s->align(FL_ALIGN_TOP_LEFT);
        s->callback([](Fl_Widget* w, void* d){ *(float*)d = (float)((Fl_Value_Slider*)w)->value(); }, v);
        y += 60; return s;
    };

    Fl_Value_Slider* t_sld = create_slider("Time (τ)", 0.0f, 20.0f, 0.0f, &view->currentTime);
    create_slider("Lattice V0", 0.0f, 10.0f, 1.0f, &view->V0);
    create_slider("Coupling γ", 0.0f, 0.2f, 0.01f, &view->gamma);
    create_slider("Mass m*", 0.5f, 5.0f, 0.8f, &view->m);

    y += 20;
    create_slider("Momentum p0", -5.0f, 5.0f, 2.0f, &view->source_p0);
    create_slider("Width σ", 0.5f, 4.0f, 1.2f, &view->source_sigma);

    auto play = new Fl_Check_Button(20, y, 150, 25, "Live");
    play->labelcolor(FL_WHITE);
    play->callback([](Fl_Widget* w, void* d){ ((PhysicalWignerView*)d)->running = ((Fl_Check_Button*)w)->value(); }, view);
    
    auto reinit = new Fl_Button(20, y+40, 150, 35, "Init Source");
    reinit->callback([](Fl_Widget* w, void* d){ ((PhysicalWignerView*)d)->reset_simulation(); }, view);

    auto rest = new Fl_Button(180, y+40, 150, 35, "Reset τ=0");
    rest->callback([](Fl_Widget* w, void* d){ 
        ((PhysicalWignerView*)d)->currentTime = 0;
        ((PhysicalWignerView*)d)->sys.rho = ((PhysicalWignerView*)d)->sys.rho_initial;
    }, view);

    ui->end();
    static AppContext ctx = { view, t_sld };
    Fl::add_idle([](void* d){ 
        auto c = (AppContext*)d;
        c->view->redraw();
        if(c->view->running) c->timeSlider->value(c->view->currentTime);
    }, &ctx);

    win->resizable(view);
    win->show();
    return Fl::run();
}
