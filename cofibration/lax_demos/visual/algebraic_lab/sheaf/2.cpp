// analytical_sheaf_improved_with_branchIm.cpp
// Compile with:
// g++ -o analytical_sheaf_improved_with_branchIm analytical_sheaf_improved_with_branchIm.cpp -lfltk -lfltk_gl -lGL -lGLU

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
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

// -------------------------------------------------------------------------
static constexpr float PI = 3.14159265358979323846f;

// Compute sqrt with explicit control of the sheet: sheet==0 -> principal
static std::complex<float> complex_sqrt_sheet(const std::complex<float>& z, int sheet = 0) {
    float r = std::abs(z);
    float theta = std::arg(z); // (-pi, pi]
    float half = theta * 0.5f;
    float mag = std::sqrt(r);
    std::complex<float> w = std::polar(mag, half);
    if (sheet == 1) w = -w;
    return w;
}

// Color mapping: argument -> HSV-like to RGB, alpha supported
static void colorByPhase(const std::complex<float>& w, float alpha) {
    float angle = std::arg(w);
    float h = (angle + PI) / (2.0f * PI); // map to [0,1)
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
// Global State representing the Morphism / UI state
// -------------------------------------------------------------------------
struct MorphismState {
    // Branch point A (we now allow full movement in C)
    float branchRe = 1.0f;
    float branchIm = 0.0f; // Imaginary part added
    float transparency = 0.6f;

    // Probe (base point z) used to show the stalk / fiber
    float probeRe = 0.0f;
    float probeIm = 0.0f;

    // Toggles for educational layers
    bool showFiber = true;
    bool showBranchCut = true;
    bool showProjectionLines = true;
    bool showLocalTrivialization = false;

    // Height mapping: 0=Re(w), 1=Im(w), 2=|w|
    int heightMode = 0;

    // Animation / monodromy demo
    bool animateMonodromy = false;
    float monodromyAngle = 0.0f; // parameter for animation [0, 2pi)
} g_state;

// -------------------------------------------------------------------------
// OpenGL rendering window
// -------------------------------------------------------------------------
class SheafWindow : public Fl_Gl_Window {
    float rotX, rotY;
    int lastX, lastY;
    bool dragging;

public:
    SheafWindow(int x, int y, int w, int h)
        : Fl_Gl_Window(x,y,w,h), rotX(25), rotY(-35), dragging(false) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA);
    }

    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(50.0, (float)w() / (float)h(), 0.1, 100.0);
            glMatrixMode(GL_MODELVIEW);
        }

        // Background
        glClearColor(0.08f, 0.09f, 0.11f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();

        // Camera
        gluLookAt(0, 6, 12, 0, 0, 0, 0, 1, 0);
        glRotatef(rotX, 1, 0, 0);
        glRotatef(rotY, 0, 1, 0);

        // If animation running, update probe to circle around branch point to
        // demonstrate monodromy (switching sheets when winding once).
        if (g_state.animateMonodromy) {
            float r = 1.0f;
            float A_re = g_state.branchRe;
            float A_im = g_state.branchIm;
            float angle = g_state.monodromyAngle;
            g_state.probeRe = A_re + r * std::cos(angle);
            g_state.probeIm = A_im + r * std::sin(angle);
            // advance angle for next frame
            g_state.monodromyAngle += 0.02f;
            if (g_state.monodromyAngle > 2.0f * PI) g_state.monodromyAngle -= 2.0f * PI;
            // ensure UI updates
            Fl::repeat_timeout(0.01, [](void*){ if(g_glWin) g_glWin->redraw(); }, nullptr);
        }

        // Draw elements
        drawBaseGrid();
        if (g_state.showBranchCut) drawBranchCut();
        drawRiemannSurface();
        if (g_state.showFiber) drawFiberInteraction();
        if (g_state.showLocalTrivialization) drawLocalTrivialization();
    }

    int handle(int event) override {
        switch(event) {
            case FL_PUSH:
                lastX = Fl::event_x(); lastY = Fl::event_y(); dragging = true; return 1;
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
            default:
                return Fl_Gl_Window::handle(event);
        }
    }

private:
    // Draw the base plane representing the complex z-plane (y = -2)
    void drawBaseGrid() {
        float y_base = -2.0f;
        float size = 4.0f;
        int divs = 10;

        glLineWidth(1.0f);
        glColor4f(0.45f, 0.45f, 0.45f, 0.25f);
        glBegin(GL_LINES);
        for (int i=-divs; i<=divs; ++i) {
            float v = (float)i / divs * size;
            glVertex3f(-size, y_base, v); glVertex3f(size, y_base, v);
            glVertex3f(v, y_base, -size); glVertex3f(v, y_base, size);
        }
        glEnd();

        // Branch point marker (now placed at (branchRe, branchIm) in the plane)
        glPushMatrix();
        glTranslatef(g_state.branchRe, y_base, g_state.branchIm);
        glColor4f(1.0f, 0.25f, 0.25f, 1.0f);
        GLUquadric* q = gluNewQuadric();
        gluSphere(q, 0.12f, 14, 14);
        gluDeleteQuadric(q);
        glPopMatrix();
    }

    // Visualize a branch cut: ray from A towards -real direction (relative to A)
    void drawBranchCut() {
        float y_base = -2.0f;
        float A_re = g_state.branchRe;
        float A_im = g_state.branchIm;
        float length = 6.0f;
        glLineWidth(3.0f);
        glColor4f(0.8f, 0.5f, 0.1f, 0.8f);
        glBegin(GL_LINES);
        glVertex3f(A_re, y_base, A_im);
        glVertex3f(A_re - length, y_base, A_im);
        glEnd();

        // small arrow head
        glBegin(GL_TRIANGLES);
        glVertex3f(A_re - length, y_base, A_im);
        glVertex3f(A_re - length + 0.2f, y_base + 0.08f, A_im);
        glVertex3f(A_re - length + 0.2f, y_base - 0.08f, A_im);
        glEnd();
    }

    // Draw the Riemann surface X (two sheets). Color by Arg(w), height via mode.
    void drawRiemannSurface() {
        int res = 80;
        float limit = 3.0f;

        for (int sheet=0; sheet<2; ++sheet) {
            for (int i=0; i<res; ++i) {
                float u0 = (float)i / res * 2.0f * limit - limit;
                float u1 = (float)(i+1) / res * 2.0f * limit - limit;

                glBegin(GL_TRIANGLE_STRIP);
                for (int j=0; j<=res; ++j) {
                    float v = (float)j / res * 2.0f * limit - limit;

                    std::complex<float> z0(u0, v), z1(u1, v);
                    std::complex<float> A(g_state.branchRe, g_state.branchIm);

                    std::complex<float> w0 = complex_sqrt_sheet(z0 - A, sheet);
                    std::complex<float> w1 = complex_sqrt_sheet(z1 - A, sheet);

                    auto height = [&](const std::complex<float>& w)->float{
                        switch (g_state.heightMode) {
                            case 0: return w.real();
                            case 1: return w.imag();
                            case 2: return std::abs(w);
                            default: return w.real();
                        }
                    };

                    colorByPhase(w0, g_state.transparency);
                    glVertex3f(u0, height(w0), v);

                    colorByPhase(w1, g_state.transparency);
                    glVertex3f(u1, height(w1), v);
                }
                glEnd();
            }
        }
    }

    // Draw the fiber (stalk) above the probe z and the two corresponding points
    void drawFiberInteraction() {
        float y_base = -2.0f;
        float z_re = g_state.probeRe;
        float z_im = g_state.probeIm;

        // Draw base probe
        glPushMatrix();
        glTranslatef(z_re, y_base, z_im);
        glColor4f(1.0f, 1.0f, 0.2f, 1.0f);
        GLUquadric* q = gluNewQuadric();
        gluSphere(q, 0.09f, 12, 12);
        gluDeleteQuadric(q);
        glPopMatrix();

        std::complex<float> z(z_re, z_im);
        std::complex<float> A(g_state.branchRe, g_state.branchIm);

        // Compute both sheet values
        std::complex<float> w0 = complex_sqrt_sheet(z - A, 0);
        std::complex<float> w1 = complex_sqrt_sheet(z - A, 1);

        auto heightVal = [&](const std::complex<float>& w)->float{
            switch (g_state.heightMode) {
                case 0: return w.real();
                case 1: return w.imag();
                case 2: return std::abs(w);
                default: return w.real();
            }
        };

        float y0 = heightVal(w0);
        float y1 = heightVal(w1);

        // Projection lines / stalks
        glLineWidth(2.5f);
        if (g_state.showProjectionLines) {
            glBegin(GL_LINES);
            glColor4f(1.0f, 1.0f, 0.0f, 0.5f);
            glVertex3f(z_re, y_base, z_im);
            glVertex3f(z_re, y0, z_im);

            glVertex3f(z_re, y_base, z_im);
            glVertex3f(z_re, y1, z_im);
            glEnd();
        }

        // Points on the surface
        glPushMatrix();
        glTranslatef(z_re, y0, z_im);
        glColor4f(0.2f, 1.0f, 0.3f, 1.0f);
        GLUquadric* q2 = gluNewQuadric();
        gluSphere(q2, 0.09f, 12, 12);
        gluDeleteQuadric(q2);
        glPopMatrix();

        glPushMatrix();
        glTranslatef(z_re, y1, z_im);
        glColor4f(0.2f, 1.0f, 0.3f, 1.0f);
        GLUquadric* q3 = gluNewQuadric();
        gluSphere(q3, 0.09f, 12, 12);
        gluDeleteQuadric(q3);
        glPopMatrix();
    }

    // Illustrate a local trivialization
    void drawLocalTrivialization() {
        float baseY = -2.0f;
        float R = 0.6f;
        int segs = 20;
        float cx = g_state.probeRe;
        float cz = g_state.probeIm;

        glPushMatrix();
        glBegin(GL_TRIANGLE_FAN);
        glColor4f(0.8f, 0.8f, 0.2f, 0.08f);
        glVertex3f(cx, baseY+0.001f, cz);
        for (int i=0;i<=segs;++i) {
            float a = (float)i / segs * 2.0f * PI;
            glVertex3f(cx + R*std::cos(a), baseY+0.001f, cz + R*std::sin(a));
        }
        glEnd();

        std::complex<float> z(cx, cz);
        std::complex<float> A(g_state.branchRe, g_state.branchIm);
        std::complex<float> w0 = complex_sqrt_sheet(z - A, 0);
        std::complex<float> w1 = complex_sqrt_sheet(z - A, 1);
        auto h = [&](const std::complex<float>& w){
            switch (g_state.heightMode) { case 0: return w.real(); case 1: return w.imag(); case 2: return std::abs(w);} return w0.real();
        };
        float y0 = h(w0); float y1 = h(w1);

        glLineWidth(2.0f);
        glBegin(GL_LINES);
        glColor4f(0.9f, 0.9f, 0.2f, 0.25f);
        for (int i=0;i<6;++i) {
            float a = (float)i / 6.0f * 2.0f * PI;
            float bx = cx + 0.4f * std::cos(a);
            float bz = cz + 0.4f * std::sin(a);
            glVertex3f(bx, baseY, bz); glVertex3f(bx, y0, bz);
            glVertex3f(bx, baseY, bz); glVertex3f(bx, y1, bz);
        }
        glEnd();

        glPopMatrix();
    }
};

// Global pointer for callbacks
SheafWindow* g_glWin = nullptr;
Fl_Box* g_infoBox = nullptr;

// -------------------------------------------------------------------------
// UI Callbacks
// -------------------------------------------------------------------------
void update_ui_text() {
    if (!g_infoBox) return;
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    ss << "w^2 = z - A    A = (" << g_state.branchRe << ", " << g_state.branchIm << ")\n";
    ss << "Probe z = (" << g_state.probeRe << ", " << g_state.probeIm << ")\n\n";

    ss << "Concepts:\n";
    ss << "- Base S: the z-plane (grid).\n";
    ss << "- Total Space X: the Riemann surface of the multivalued function.\n";
    ss << "- Projection π: (x,y,z) -> z (vertical drop).\n";
    ss << "- Stalk π^{-1}(z): the fiber (two points generically).\n";
    ss << "- Ramification: at A the fiber size effectively 'collapses'.\n\n";

    ss << "Local vs Global:\n";
    ss << "- Locally X ≅ U × {sheet set} (local trivialization).\n";
    ss << "- Globally there is monodromy: circling A swaps the sheets,\n  so there is no single-valued global section.\n";

    g_infoBox->label(ss.str().c_str());
    g_infoBox->redraw();
}

void slider_branch_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.branchRe = s->value();
    update_ui_text();
    if (g_glWin) g_glWin->redraw();
}
void slider_branch_im_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.branchIm = s->value();
    update_ui_text();
    if (g_glWin) g_glWin->redraw();
}
void slider_alpha_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.transparency = s->value();
    if (g_glWin) g_glWin->redraw();
}
void slider_probe_x_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.probeRe = s->value();
    update_ui_text(); if (g_glWin) g_glWin->redraw();
}
void slider_probe_y_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.probeIm = s->value();
    update_ui_text(); if (g_glWin) g_glWin->redraw();
}

void toggle_fiber_cb(Fl_Widget* w, void*) { g_state.showFiber = ((Fl_Check_Button*)w)->value(); if(g_glWin) g_glWin->redraw(); }
void toggle_branchcut_cb(Fl_Widget* w, void*) { g_state.showBranchCut = ((Fl_Check_Button*)w)->value(); if(g_glWin) g_glWin->redraw(); }
void toggle_proj_cb(Fl_Widget* w, void*) { g_state.showProjectionLines = ((Fl_Check_Button*)w)->value(); if(g_glWin) g_glWin->redraw(); }
void toggle_trivial_cb(Fl_Widget* w, void*) { g_state.showLocalTrivialization = ((Fl_Check_Button*)w)->value(); if(g_glWin) g_glWin->redraw(); }

void choice_height_cb(Fl_Widget* w, void*) {
    Fl_Choice* c = (Fl_Choice*)w;
    g_state.heightMode = c->value();
    if (g_glWin) g_glWin->redraw();
}

// Monodromy animation
void btn_monodromy_cb(Fl_Widget* w, void*) {
    Fl_Button* b = (Fl_Button*)w;
    g_state.animateMonodromy = !g_state.animateMonodromy;
    if (g_state.animateMonodromy) {
        b->label("Stop Monodromy");
        Fl::add_timeout(0.01, [](void*){ if(g_glWin) g_glWin->redraw(); }, nullptr);
    } else {
        b->label("Trace Monodromy");
    }
}

// -------------------------------------------------------------------------
// Main
// -------------------------------------------------------------------------
int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1000, 650, "Grothendieck: Etale / Cover Visualization (branchIm added)");

    g_glWin = new SheafWindow(10, 10, 720, 630);

    Fl_Group* ui = new Fl_Group(740, 10, 250, 630);

    // Info box
    g_infoBox = new Fl_Box(750, 20, 230, 220, "");
    g_infoBox->align(FL_ALIGN_INSIDE | FL_ALIGN_TOP | FL_ALIGN_WRAP);
    g_infoBox->labelfont(FL_COURIER);

    int y = 250;
    Fl_Box* label1 = new Fl_Box(740, y, 250, 25, "Visualization Controls:");
    label1->labelfont(FL_HELVETICA_BOLD);
    y += 28;

    Fl_Value_Slider* sA = new Fl_Value_Slider(750, y, 220, 24, "Branch A (Re)");
    sA->bounds(-2.5, 2.5); sA->value(g_state.branchRe); sA->callback(slider_branch_cb);
    y += 44;

    // NEW: Branch imaginary part slider
    Fl_Value_Slider* sA_im = new Fl_Value_Slider(750, y, 220, 24, "Branch A (Im)");
    sA_im->bounds(-2.5, 2.5); sA_im->value(g_state.branchIm); sA_im->callback(slider_branch_im_cb);
    y += 44;

    Fl_Value_Slider* sAlpha = new Fl_Value_Slider(750, y, 220, 24, "Surface Transparency");
    sAlpha->bounds(0.05, 1.0); sAlpha->value(g_state.transparency); sAlpha->callback(slider_alpha_cb);
    y += 44;

    Fl_Box* fiberSep = new Fl_Box(740, y, 250, 22, "--- Fiber / Probe ---"); fiberSep->labelfont(FL_HELVETICA_BOLD); y += 26;

    Fl_Value_Slider* spx = new Fl_Value_Slider(750, y, 220, 24, "Probe z (Re)"); spx->bounds(-3.0, 3.0); spx->value(g_state.probeRe); spx->callback(slider_probe_x_cb); y += 44;
    Fl_Value_Slider* spy = new Fl_Value_Slider(750, y, 220, 24, "Probe z (Im)"); spy->bounds(-3.0, 3.0); spy->value(g_state.probeIm); spy->callback(slider_probe_y_cb); y += 44;

    Fl_Check_Button* cbFiber = new Fl_Check_Button(750, y, 220, 20, "Show Fiber / Stalk"); cbFiber->value(g_state.showFiber); cbFiber->callback(toggle_fiber_cb); y += 26;
    Fl_Check_Button* cbProj = new Fl_Check_Button(750, y, 220, 20, "Show Projection Lines"); cbProj->value(g_state.showProjectionLines); cbProj->callback(toggle_proj_cb); y += 26;
    Fl_Check_Button* cbBranchCut = new Fl_Check_Button(750, y, 220, 20, "Show Branch Cut"); cbBranchCut->value(g_state.showBranchCut); cbBranchCut->callback(toggle_branchcut_cb); y += 26;
    Fl_Check_Button* cbTriv = new Fl_Check_Button(750, y, 220, 20, "Highlight Local Trivialization"); cbTriv->value(g_state.showLocalTrivialization); cbTriv->callback(toggle_trivial_cb); y += 32;

    Fl_Box* mappingSep = new Fl_Box(740, y, 250, 22, "--- Height Mapping ---"); mappingSep->labelfont(FL_HELVETICA_BOLD); y += 26;

    Fl_Choice* chMap = new Fl_Choice(750, y, 220, 24, "Height");
    chMap->add("Re(w)"); chMap->add("Im(w)"); chMap->add("|w|"); chMap->value(g_state.heightMode); chMap->callback(choice_height_cb); y += 40;

    Fl_Button* btnMono = new Fl_Button(750, y, 220, 30, "Trace Monodromy"); btnMono->callback(btn_monodromy_cb); y += 44;

    ui->end();

    win->end();
    win->resizable(g_glWin);

    update_ui_text();
    win->show(argc, argv);
    return Fl::run();
}
