// analytical_sheaf_polylines.cpp
// Compile with:
// g++ -o analytical_sheaf_polylines analytical_sheaf_polylines.cpp -lfltk -lfltk_gl -lGL -lGLU

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Scroll.H>
#include <FL/Fl_Pack.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <complex>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

static constexpr float PI = 3.14159265358979323846f;

// -------------------------------------------------------------------------
// Complex Math Utilities
// -------------------------------------------------------------------------
static std::complex<float> complex_sqrt_sheet(const std::complex<float>& z, int sheet = 0) {
    float r = std::abs(z);
    float theta = std::arg(z);
    float half = theta * 0.5f;
    float mag = std::sqrt(r);
    std::complex<float> w = std::polar(mag, half);
    if (sheet == 1) w = -w;
    return w;
}

static void colorByPhase(const std::complex<float>& w, float alpha) {
    float angle = std::arg(w);
    float h = (angle + PI) / (2.0f * PI);
    float s = 1.0f, v = 1.0f;
    float c = v * s;
    float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    float rr=0, gg=0, bb=0;
    float sixh = h * 6.0f;
    if (sixh < 1.0f) { rr = c; gg = x; bb = 0; }
    else if (sixh < 2.0f) { rr = x; gg = c; bb = 0; }
    else if (sixh < 3.0f) { rr = 0; gg = c; bb = x; }
    else if (sixh < 4.0f) { rr = 0; gg = x; bb = c; }
    else if (sixh < 5.0f) { rr = x; gg = 0; bb = c; }
    else { rr = c; gg = 0; bb = x; }
    glColor4f(rr + m, gg + m, bb + m, alpha);
}

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------
struct PathSample { std::complex<float> z; std::complex<float> w; };

struct MorphismState {
    float branchRe = 1.0f;
    float branchIm = 0.0f;
    float transparency = 0.6f;
    float probeRe = 0.0f;
    float probeIm = 0.0f;
    float pathStartRe = -1.5f;
    float pathStartIm = -0.5f;
    float pathEndRe = 1.5f;
    float pathEndIm = -0.5f;
    bool showPath = true;
    std::vector<std::complex<float>> drawnPath;
    bool drawMode = false;
    bool showFiber = true;
    bool showBranchCut = true;
    bool showProjectionLines = true;
    bool showLocalTrivialization = false;
    int heightMode = 0;
    bool animateMonodromy = false;
    float monodromyAngle = 0.0f;
    float zoom = 1.0f; // Zoom factor
    std::string pathResult = "Ready";
} g_state;

// -------------------------------------------------------------------------
// Continuity Logic
// -------------------------------------------------------------------------
static void sampleSegment(const std::complex<float>& a, const std::complex<float>& b, int n, std::vector<std::complex<float>>& out) {
    for (int i = 0; i < n; ++i) {
        float t = (float)i / (float)n;
        out.push_back({a.real() + t * (b.real() - a.real()), a.imag() + t * (b.imag() - a.imag())});
    }
}

bool testPathContinuityPolyline(const std::vector<std::complex<float>>& poly, std::vector<PathSample>& outSamples, std::string& outMessage) {
    outSamples.clear();
    if (poly.size() < 2) { outMessage = "Path too short."; return false; }
    std::complex<float> A(g_state.branchRe, g_state.branchIm);
    std::vector<std::complex<float>> samplesZ;
    float totalLen = 0.0f;
    for (size_t i=0;i+1<poly.size();++i) totalLen += std::hypot(poly[i+1].real()-poly[i].real(), poly[i+1].imag()-poly[i].imag());
    if (totalLen < 1e-6f) { outMessage = "Zero length path."; return false; }
    
    const int targetTotalSamples = 300;
    for (size_t i=0;i+1<poly.size();++i) {
        float segLen = std::hypot(poly[i+1].real()-poly[i].real(), poly[i+1].imag()-poly[i].imag());
        int segSamples = std::max(4, (int)std::round((segLen / totalLen) * targetTotalSamples));
        sampleSegment(poly[i], poly[i+1], segSamples, samplesZ);
    }
    samplesZ.push_back(poly.back());

    std::complex<float> w_prev = complex_sqrt_sheet(samplesZ.front() - A, 0);
    outSamples.push_back({samplesZ.front(), w_prev});
    for (size_t i=1;i<samplesZ.size();++i) {
        std::complex<float> z = samplesZ[i];
        std::complex<float> wp = complex_sqrt_sheet(z - A, 0);
        std::complex<float> wm = -wp;
        w_prev = (std::abs(wp - w_prev) <= std::abs(wm - w_prev)) ? wp : wm;
        outSamples.push_back({z, w_prev});
    }

    float rel = std::abs(outSamples.back().w - outSamples.front().w);
    float relSwap = std::abs(outSamples.back().w + outSamples.front().w);
    if (rel <= 1e-2f) { outMessage = "Success: Continuous section found."; return true; }
    if (relSwap <= 1e-2f) { outMessage = "Monodromy: Flipped to second sheet."; return false; }
    outMessage = "Indeterminate path."; return false;
}

// -------------------------------------------------------------------------
// GL Window
// -------------------------------------------------------------------------
class SheafWindow : public Fl_Gl_Window {
    float rotX = 25.0f, rotY = -35.0f;
    int lastX, lastY;
    bool dragging = false;

public:
    SheafWindow(int x,int y,int w,int h):Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB|FL_DOUBLE|FL_DEPTH|FL_ALPHA);
    }

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            valid(1);
        }
        glClearColor(0.08f,0.09f,0.11f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        float aspect = (float)w()/(float)h();
        gluPerspective(45.0 / g_state.zoom, aspect, 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        gluLookAt(0,6,12, 0,0,0, 0,1,0);
        glRotatef(rotX, 1, 0, 0);
        glRotatef(rotY, 0, 1, 0);

        drawBaseGrid();
        if (g_state.showBranchCut) drawBranchCut();
        drawRiemannSurface();
        if (g_state.showPath) drawPath();
        if (g_state.showFiber) drawFiberInteraction();
    }

    int handle(int event) override {
        switch(event) {
            case FL_MOUSEWHEEL:
                g_state.zoom -= Fl::event_dy() * 0.1f;
                if (g_state.zoom < 0.1f) g_state.zoom = 0.1f;
                if (g_state.zoom > 10.0f) g_state.zoom = 10.0f;
                redraw();
                return 1;
            case FL_SHORTCUT:
                if (Fl::event_key() == '+') { g_state.zoom *= 1.1f; redraw(); return 1; }
                if (Fl::event_key() == '-') { g_state.zoom /= 1.1f; redraw(); return 1; }
                return 0;
            case FL_PUSH:
                if (g_state.drawMode) {
                    float wx, wz;
                    if (screenToPlane(Fl::event_x(), Fl::event_y(), -2.0f, wx, wz)) {
                        g_state.drawnPath.emplace_back(wx, wz);
                        redraw();
                    }
                } else {
                    lastX = Fl::event_x(); lastY = Fl::event_y(); dragging = true;
                }
                return 1;
            case FL_DRAG:
                if (dragging) {
                    rotY += (Fl::event_x() - lastX);
                    rotX += (Fl::event_y() - lastY);
                    lastX = Fl::event_x(); lastY = Fl::event_y();
                    redraw();
                }
                return 1;
            case FL_RELEASE:
                dragging = false; return 1;
            default: return Fl_Gl_Window::handle(event);
        }
    }

private:
    bool screenToPlane(int mx, int my, float planeY, float &outX, float &outZ) {
        GLint viewport[4]; GLdouble model[16], proj[16];
        glGetIntegerv(GL_VIEWPORT, viewport);
        glGetDoublev(GL_MODELVIEW_MATRIX, model);
        glGetDoublev(GL_PROJECTION_MATRIX, proj);
        double winY = viewport[3] - my;
        GLdouble x0, y0, z0, x1, y1, z1;
        gluUnProject(mx, winY, 0.0, model, proj, viewport, &x0, &y0, &z0);
        gluUnProject(mx, winY, 1.0, model, proj, viewport, &x1, &y1, &z1);
        double t = (planeY - y0) / (y1 - y0);
        outX = (float)(x0 + t * (x1 - x0));
        outZ = (float)(z0 + t * (z1 - z0));
        return true;
    }

    void drawBaseGrid() {
        float y = -2.0f;
        glLineWidth(1.0f);
        glColor4f(0.5f, 0.5f, 0.5f, 0.2f);
        glBegin(GL_LINES);
        for(int i=-5; i<=5; ++i) {
            glVertex3f(i, y, -5); glVertex3f(i, y, 5);
            glVertex3f(-5, y, i); glVertex3f(5, y, i);
        }
        glEnd();
        // Branch point marker
        glPushMatrix();
        glTranslatef(g_state.branchRe, y, g_state.branchIm);
        glColor3f(1, 0, 0);
        GLUquadric* q = gluNewQuadric(); gluSphere(q, 0.1, 16, 16); gluDeleteQuadric(q);
        glPopMatrix();
    }

    void drawBranchCut() {
        glLineWidth(2.0f); glColor4f(1, 0.5f, 0, 0.8f);
        glBegin(GL_LINES);
        glVertex3f(g_state.branchRe, -2.0f, g_state.branchIm);
        glVertex3f(g_state.branchRe - 10.0f, -2.0f, g_state.branchIm);
        glEnd();
    }

    void drawRiemannSurface() {
        int res = 50; float lim = 4.0f;
        for(int s=0; s<2; ++s) {
            for(int i=0; i<res; ++i) {
                float u0 = -lim + (2*lim*i)/res, u1 = -lim + (2*lim*(i+1))/res;
                glBegin(GL_TRIANGLE_STRIP);
                for(int j=0; j<=res; ++j) {
                    float v = -lim + (2*lim*j)/res;
                    std::complex<float> z0(u0, v), z1(u1, v), A(g_state.branchRe, g_state.branchIm);
                    std::complex<float> w0 = complex_sqrt_sheet(z0-A, s), w1 = complex_sqrt_sheet(z1-A, s);
                    auto h = [&](const std::complex<float>& w) { 
                        return (g_state.heightMode == 1) ? w.imag() : (g_state.heightMode == 2 ? std::abs(w) : w.real()); 
                    };
                    colorByPhase(w0, g_state.transparency); glVertex3f(u0, h(w0), v);
                    colorByPhase(w1, g_state.transparency); glVertex3f(u1, h(w1), v);
                }
                glEnd();
            }
        }
    }

    void drawPath() {
        float y = -1.99f;
        std::vector<PathSample> samples; std::string msg;
        if (g_state.drawnPath.size() >= 2) {
            glLineWidth(3.0f); glColor3f(1, 1, 0);
            glBegin(GL_LINE_STRIP); for(auto &p : g_state.drawnPath) glVertex3f(p.real(), y, p.imag()); glEnd();
            testPathContinuityPolyline(g_state.drawnPath, samples, msg);
        } else {
            glLineWidth(2.0f); glColor3f(0, 0.8f, 1);
            glBegin(GL_LINES); glVertex3f(g_state.pathStartRe, y, g_state.pathStartIm); glVertex3f(g_state.pathEndRe, y, g_state.pathEndIm); glEnd();
            testPathContinuityStraight(g_state.pathStartRe, g_state.pathStartIm, g_state.pathEndRe, g_state.pathEndIm, samples, msg);
        }
        if (!samples.empty()) {
            glLineWidth(5.0f); glColor4f(0, 1, 0, 0.6f);
            glBegin(GL_LINE_STRIP); for(auto &s : samples) glVertex3f(s.z.real(), y+0.01f, s.z.imag()); glEnd();
        }
        g_state.pathResult = msg;
    }

    void drawFiberInteraction() {
        float zr = g_state.probeRe, zi = g_state.probeIm;
        std::complex<float> z(zr, zi), A(g_state.branchRe, g_state.branchIm);
        std::complex<float> w0 = complex_sqrt_sheet(z-A, 0), w1 = complex_sqrt_sheet(z-A, 1);
        auto h = [&](const std::complex<float>& w) { return (g_state.heightMode==1)?w.imag():(g_state.heightMode==2?std::abs(w):w.real()); };
        float h0 = h(w0), h1 = h(w1);
        if (g_state.showProjectionLines) {
            glBegin(GL_LINES); glColor4f(1, 1, 1, 0.4f);
            glVertex3f(zr, -2.0f, zi); glVertex3f(zr, h0, zi);
            glVertex3f(zr, -2.0f, zi); glVertex3f(zr, h1, zi);
            glEnd();
        }
        glPointSize(12.0f); glBegin(GL_POINTS); glColor3f(0, 1, 0.5f);
        glVertex3f(zr, h0, zi); glVertex3f(zr, h1, zi);
        glEnd();
    }

    void testPathContinuityStraight(float sx, float sy, float ex, float ey, std::vector<PathSample>& out, std::string& msg) {
        std::vector<std::complex<float>> p = {{sx, sy}, {ex, ey}};
        testPathContinuityPolyline(p, out, msg);
    }
};

SheafWindow* g_glWin = nullptr;
Fl_Box* g_infoLabel = nullptr;

// -------------------------------------------------------------------------
// UI Construction
// -------------------------------------------------------------------------
void update_ui() {
    if (!g_infoLabel) return;
    std::ostringstream ss;
    ss << "Branch Point A: (" << std::fixed << std::setprecision(1) << g_state.branchRe << ", " << g_state.branchIm << ")\n"
       << "Probe Position: (" << g_state.probeRe << ", " << g_state.probeIm << ")\n"
       << "Zoom: " << std::setprecision(2) << g_state.zoom << "x\n"
       << "Result: " << g_state.pathResult;
    g_infoLabel->label(ss.str().c_str());
    if (g_glWin) g_glWin->redraw();
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1100, 700, "Analytical Sheaf Visualizer");
    g_glWin = new SheafWindow(10, 10, 780, 680);

    // Scrollable control panel
    Fl_Scroll* scroll = new Fl_Scroll(800, 10, 290, 680);
    scroll->type(Fl_Scroll::VERTICAL);
    
    Fl_Pack* pack = new Fl_Pack(800, 10, 260, 1200);
    pack->spacing(15);

    g_infoLabel = new Fl_Box(0, 0, 260, 100, "Status");
    g_infoLabel->box(FL_FLAT_BOX);
    g_infoLabel->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE | FL_ALIGN_TOP);

    auto add_slider = [&](const char* name, float min, float max, float val, float& target) {
        Fl_Box* b = new Fl_Box(0, 0, 260, 20, name);
        b->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
        Fl_Value_Slider* s = new Fl_Value_Slider(0, 0, 260, 25);
        s->type(FL_HORIZONTAL); s->bounds(min, max); s->value(val);
        s->callback([](Fl_Widget* w, void* data){
            *(float*)data = ((Fl_Value_Slider*)w)->value();
            update_ui();
        }, &target);
    };

    add_slider("Branch Point Re", -3.0, 3.0, g_state.branchRe, g_state.branchRe);
    add_slider("Branch Point Im", -3.0, 3.0, g_state.branchIm, g_state.branchIm);
    add_slider("Surface Alpha", 0.1, 1.0, g_state.transparency, g_state.transparency);
    add_slider("Probe X", -4.0, 4.0, g_state.probeRe, g_state.probeRe);
    add_slider("Probe Y", -4.0, 4.0, g_state.probeIm, g_state.probeIm);

    Fl_Check_Button* cbDraw = new Fl_Check_Button(0, 0, 260, 25, "Draw Path Mode (Click GL)");
    cbDraw->callback([](Fl_Widget* w, void*){ g_state.drawMode = ((Fl_Check_Button*)w)->value(); });

    Fl_Button* btnClear = new Fl_Button(0, 0, 260, 30, "Clear Drawn Path");
    btnClear->callback([](Fl_Widget*, void*){ g_state.drawnPath.clear(); update_ui(); });

    Fl_Choice* choice = new Fl_Choice(0, 0, 260, 30, "Height Map");
    choice->add("Real Part"); choice->add("Imaginary Part"); choice->add("Magnitude");
    choice->value(0);
    choice->callback([](Fl_Widget* w, void*){ g_state.heightMode = ((Fl_Choice*)w)->value(); update_ui(); });

    Fl_Check_Button* cbCut = new Fl_Check_Button(0, 0, 260, 25, "Show Branch Cut");
    cbCut->value(1);
    cbCut->callback([](Fl_Widget* w, void*){ g_state.showBranchCut = ((Fl_Check_Button*)w)->value(); update_ui(); });

    Fl_Box* hint = new Fl_Box(0, 0, 260, 60, "Controls:\n- Drag to rotate\n- Wheel to zoom\n- Mouse click adds points");
    hint->align(FL_ALIGN_LEFT | FL_ALIGN_INSIDE);
    hint->labelfont(FL_HELVETICA_ITALIC);

    pack->end();
    scroll->end();
    win->end();
    
    win->resizable(g_glWin);
    update_ui();
    win->show(argc, argv);
    return Fl::run();
}
