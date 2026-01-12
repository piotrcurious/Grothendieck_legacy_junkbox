// analytical_sheaf_polylines.cpp // Compile with: // g++ -o analytical_sheaf_polylines analytical_sheaf_polylines.cpp -lfltk -lfltk_gl -lGL -lGLU

// Extension: add ability to draw non-straight (polyline) paths on the // base plane by clicking in the GL view. The student can then test whether // a continuous choice of the square-root exists along that drawn route. // Left-click in the GL view when "Draw Path Mode" is enabled to add // vertices. Use "Clear Path" to remove. The Test button will use the // drawn path if present; otherwise it falls back to the straight start->end test.

#include <FL/Fl.H> #include <FL/Fl_Double_Window.H> #include <FL/Fl_Gl_Window.H> #include <FL/Fl_Group.H> #include <FL/Fl_Value_Slider.H> #include <FL/Fl_Box.H> #include <FL/Fl_Button.H> #include <FL/Fl_Check_Button.H> #include <FL/Fl_Choice.H> #include <FL/gl.h> #include <GL/glu.h>

#include <complex> #include <cmath> #include <vector> #include <string> #include <sstream> #include <iomanip> #include <algorithm>

static constexpr float PI = 3.14159265358979323846f;

static std::complex<float> complex_sqrt_sheet(const std::complex<float>& z, int sheet = 0) { float r = std::abs(z); float theta = std::arg(z); float half = theta * 0.5f; float mag = std::sqrt(r); std::complex<float> w = std::polar(mag, half); if (sheet == 1) w = -w; return w; }

static void colorByPhase(const std::complex<float>& w, float alpha) { float angle = std::arg(w); float h = (angle + PI) / (2.0f * PI); float s = 1.0f, v = 1.0f; float c = v * s; float x = c * (1.0f - std::fabs(std::fmod(h * 6.0f, 2.0f) - 1.0f)); float m = v - c; float rr=0, gg=0, bb=0; float sixh = h * 6.0f; if (sixh < 1.0f) { rr = c; gg = x; bb = 0; } else if (sixh < 2.0f) { rr = x; gg = c; bb = 0; } else if (sixh < 3.0f) { rr = 0; gg = c; bb = x; } else if (sixh < 4.0f) { rr = 0; gg = x; bb = c; } else if (sixh < 5.0f) { rr = x; gg = 0; bb = c; } else { rr = c; gg = 0; bb = x; } glColor4f(rr + m, gg + m, bb + m, alpha); }

// ------------------------------------------------------------------------- // State // ------------------------------------------------------------------------- struct PathSample { std::complex<float> z; std::complex<float> w; };

struct MorphismState { float branchRe = 1.0f; float branchIm = 0.0f; float transparency = 0.6f;

float probeRe = 0.0f;
float probeIm = 0.0f;

// Start / End for straight-line test
float pathStartRe = -1.5f;
float pathStartIm = -0.5f;
float pathEndRe = 1.5f;
float pathEndIm = -0.5f;
bool showPath = true;

// Drawn polyline path (in base coordinates)
std::vector<std::complex<float>> drawnPath;
bool drawMode = false; // when true, clicks add points

bool showFiber = true;
bool showBranchCut = true;
bool showProjectionLines = true;
bool showLocalTrivialization = false;

int heightMode = 0;

bool animateMonodromy = false;
float monodromyAngle = 0.0f;

std::string pathResult = "[No test run yet]";

} g_state;

// ------------------------------------------------------------------------- // Path continuity tests // -------------------------------------------------------------------------

// Helper: sample a straight segment between a->b into n steps (including a, excluding b) static void sampleSegment(const std::complex<float>& a, const std::complex<float>& b, int n, std::vector<std::complex<float>>& out) { for (int i = 0; i < n; ++i) { float t = (float)i / (float)n; out.push_back({a.real() + t * (b.real() - a.real()), a.imag() + t * (b.imag() - a.imag())}); } }

// Test along a polyline defined by points (p0, p1, p2, ...) bool testPathContinuityPolyline(const std::vector<std::complex<float>>& poly, std::vector<PathSample>& outSamples, std::string& outMessage) { outSamples.clear(); if (poly.size() < 2) { outMessage = "Path must contain at least two points."; return false; } std::complex<float> A(g_state.branchRe, g_state.branchIm);

// Quick check: if any vertex is too close to branch point, fail
auto nearA = [&](const std::complex<float>& p){ return std::hypot(p.real() - g_state.branchRe, p.imag() - g_state.branchIm) < 1e-3f; };
for (auto &p : poly) if (nearA(p)) { outMessage = "Path hits the branch point A: fiber collapses — not continuous."; return false; }

// Build a dense list of z samples along the polyline
std::vector<std::complex<float>> samplesZ;
// determine total poly length
float totalLen = 0.0f;
for (size_t i=0;i+1<poly.size();++i) totalLen += std::hypot(poly[i+1].real()-poly[i].real(), poly[i+1].imag()-poly[i].imag());
if (totalLen < 1e-6f) { outMessage = "Path is degenerate (zero length)."; return false; }

// target total samples similar to straight test (400)
const int targetTotalSamples = 400;
for (size_t i=0;i+1<poly.size();++i) {
    float segLen = std::hypot(poly[i+1].real()-poly[i].real(), poly[i+1].imag()-poly[i].imag());
    int segSamples = std::max(4, (int)std::round((segLen / totalLen) * targetTotalSamples));
    sampleSegment(poly[i], poly[i+1], segSamples, samplesZ);
}
// ensure last point included
samplesZ.push_back(poly.back());

// continuity selection along samples
std::complex<float> w_prev = complex_sqrt_sheet(samplesZ.front() - A, 0);
outSamples.push_back({samplesZ.front(), w_prev});
for (size_t i=1;i<samplesZ.size();++i) {
    auto z = samplesZ[i];
    std::complex<float> wp = complex_sqrt_sheet(z - A, 0);
    std::complex<float> wm = -wp;
    if (std::abs(wp - w_prev) <= std::abs(wm - w_prev)) w_prev = wp; else w_prev = wm;
    outSamples.push_back({z, w_prev});
}

// final comparison
std::complex<float> start_w = outSamples.front().w;
std::complex<float> end_w = outSamples.back().w;
float rel = std::abs(end_w - start_w);
float relSwap = std::abs(end_w + start_w);

if (rel <= 1e-2f) { outMessage = "Success: continuous choice exists along drawn path."; return true; }
else if (relSwap <= 1e-2f) { outMessage = "No: the section flips sign (monodromy) along the path."; return false; }
else { outMessage = "Indeterminate: numerical disagreement — try refining the path."; return false; }

}

// Fallback: straight-line test (keeps compatibility with earlier behavior) bool testPathContinuityStraight(float sx,float sy,float ex,float ey,std::vector<PathSample>& outSamples,std::string& outMessage) { const int steps = 400; outSamples.clear(); std::complex<float> A(g_state.branchRe,g_state.branchIm); auto distToA=[&](float x,float y){ return std::hypot(x - g_state.branchRe, y - g_state.branchIm); }; for (int i=0;i<=steps;++i){ float t=(float)i/steps; float x=sx+(ex-sx)*t; float y=sy+(ey-sy)*t; if (distToA(x,y) < 1e-3f){ outMessage = "Path hits the branch point A: fiber collapses — not continuous."; return false; }} std::complex<float> z0(sx,sy); std::complex<float> w_prev = complex_sqrt_sheet(z0-A,0); outSamples.push_back({z0,w_prev}); for (int i=1;i<=steps;++i){ float t=(float)i/steps; float x=sx+(ex-sx)*t; float y=sy+(ey-sy)*t; std::complex<float> z(x,y); std::complex<float> wp = complex_sqrt_sheet(z-A,0); std::complex<float> wm = -wp; if (std::abs(wp - w_prev) <= std::abs(wm - w_prev)) w_prev = wp; else w_prev = wm; outSamples.push_back({z,w_prev}); } std::complex<float> start_w = outSamples.front().w; std::complex<float> end_w = outSamples.back().w; float rel = std::abs(end_w - start_w); float relSwap = std::abs(end_w + start_w); if (rel <= 1e-2f) { outMessage = "Success: There exists a continuous choice of sqrt along the path."; return true; } else if (relSwap <= 1e-2f) { outMessage = "No: the section flips sign (monodromy). You ended on the other sheet."; return false; } else { outMessage = "Indeterminate: numerical disagreement — sample finer or adjust path."; return false; } }

// ------------------------------------------------------------------------- // GL window // ------------------------------------------------------------------------- class SheafWindow : public Fl_Gl_Window { float rotX, rotY; int lastX, lastY; bool dragging; public: SheafWindow(int x,int y,int w,int h):Fl_Gl_Window(x,y,w,h),rotX(25),rotY(-35),dragging(false){ mode(FL_RGB|FL_DOUBLE|FL_DEPTH|FL_ALPHA); }

void draw() override {
    if (!valid()) { glViewport(0,0,w(),h()); glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(50.0, (float)w()/h(), 0.1, 100.0); glMatrixMode(GL_MODELVIEW); }
    glClearColor(0.08f,0.09f,0.11f,1.0f); glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT); glLoadIdentity(); gluLookAt(0,6,12,0,0,0,0,1,0); glRotatef(rotX,1,0,0); glRotatef(rotY,0,1,0);
    if (g_state.animateMonodromy) { float r=1.0f; float A_re=g_state.branchRe; float A_im=g_state.branchIm; float angle=g_state.monodromyAngle; g_state.probeRe=A_re + r*std::cos(angle); g_state.probeIm=A_im + r*std::sin(angle); g_state.monodromyAngle += 0.02f; if (g_state.monodromyAngle>2*PI) g_state.monodromyAngle-=2*PI; Fl::repeat_timeout(0.01, [](void*){ if(g_glWin) g_glWin->redraw(); }, nullptr); }
    drawBaseGrid(); if (g_state.showBranchCut) drawBranchCut(); drawRiemannSurface(); if (g_state.showPath) drawPath(); if (g_state.showFiber) drawFiberInteraction(); if (g_state.showLocalTrivialization) drawLocalTrivialization();
}

int handle(int event) override {
    switch(event) {
        case FL_PUSH: {
            int mx = Fl::event_x(); int my = Fl::event_y();
            if (g_state.drawMode) {
                float wx, wz;
                if (screenToPlane(mx, my, -2.0f, wx, wz)) {
                    g_state.drawnPath.emplace_back(wx, wz);
                    if (g_glWin) g_glWin->redraw();
                }
                return 1;
            } else {
                lastX = mx; lastY = my; dragging = true; return 1;
            }
        }
        case FL_DRAG: if (!g_state.drawMode && dragging) { rotY += (Fl::event_x() - lastX); rotX += (Fl::event_y() - lastY); lastX = Fl::event_x(); lastY = Fl::event_y(); redraw(); } return 1;
        case FL_RELEASE: dragging = false; return 1;
        default: return Fl_Gl_Window::handle(event);
    }
}

private: // Convert screen coords (mx,my) to world intersection point with plane y = planeY bool screenToPlane(int mx, int my, float planeY, float &outX, float &outZ) { GLint viewport[4]; GLdouble model[16], proj[16]; glGetIntegerv(GL_VIEWPORT, viewport); glGetDoublev(GL_MODELVIEW_MATRIX, model); glGetDoublev(GL_PROJECTION_MATRIX, proj); double winX = mx; double winY = viewport[3] - my; // invert Y // unproject at near and far GLdouble x0, y0, z0, x1, y1, z1; if (!gluUnProject(winX, winY, 0.0, model, proj, viewport, &x0, &y0, &z0)) return false; if (!gluUnProject(winX, winY, 1.0, model, proj, viewport, &x1, &y1, &z1)) return false; double dy = y1 - y0; if (std::abs(dy) < 1e-9) return false; double t = (planeY - y0) / dy; double ix = x0 + t * (x1 - x0); double iz = z0 + t * (z1 - z0); outX = (float)ix; outZ = (float)iz; return true; }

void drawBaseGrid(){ float y_base=-2.0f; float size=4.0f; int divs=10; glLineWidth(1.0f); glColor4f(0.45f,0.45f,0.45f,0.25f); glBegin(GL_LINES); for(int i=-divs;i<=divs;++i){ float v=(float)i/divs*size; glVertex3f(-size,y_base,v); glVertex3f(size,y_base,v); glVertex3f(v,y_base,-size); glVertex3f(v,y_base,size);} glEnd(); glPushMatrix(); glTranslatef(g_state.branchRe,y_base,g_state.branchIm); glColor4f(1.0f,0.25f,0.25f,1.0f); GLUquadric* q=gluNewQuadric(); gluSphere(q,0.12f,14,14); gluDeleteQuadric(q); glPopMatrix(); }

void drawBranchCut(){ float y_base=-2.0f; float A_re=g_state.branchRe; float A_im=g_state.branchIm; float length=6.0f; glLineWidth(3.0f); glColor4f(0.8f,0.5f,0.1f,0.8f); glBegin(GL_LINES); glVertex3f(A_re,y_base,A_im); glVertex3f(A_re-length,y_base,A_im); glEnd(); glBegin(GL_TRIANGLES); glVertex3f(A_re-length,y_base,A_im); glVertex3f(A_re-length+0.2f,y_base+0.08f,A_im); glVertex3f(A_re-length+0.2f,y_base-0.08f,A_im); glEnd(); }

void drawRiemannSurface(){ int res=80; float limit=3.0f; for(int sheet=0;sheet<2;++sheet){ for(int i=0;i<res;++i){ float u0=(float)i/res*2.0f*limit-limit; float u1=(float)(i+1)/res*2.0f*limit-limit; glBegin(GL_TRIANGLE_STRIP); for(int j=0;j<=res;++j){ float v=(float)j/res*2.0f*limit-limit; std::complex<float> z0(u0,v), z1(u1,v); std::complex<float> A(g_state.branchRe,g_state.branchIm); std::complex<float> w0=complex_sqrt_sheet(z0-A,sheet); std::complex<float> w1=complex_sqrt_sheet(z1-A,sheet); auto height=[&](const std::complex<float>& w)->float{ switch(g_state.heightMode){ case 0: return w.real(); case 1: return w.imag(); case 2: return std::abs(w); default: return w.real(); }}; colorByPhase(w0,g_state.transparency); glVertex3f(u0,height(w0),v); colorByPhase(w1,g_state.transparency); glVertex3f(u1,height(w1),v);} glEnd(); }}

// Draw either drawn polyline path (if present) or straight start->end path
void drawPath() {
    float y_base = -2.0f;
    std::vector<PathSample> samples; std::string msg;
    bool ok = false;
    if (!g_state.drawnPath.empty() && g_state.drawnPath.size() >= 2) {
        // visualize drawn polyline points
        glPointSize(6.0f); glBegin(GL_POINTS); glColor4f(1.0f,0.8f,0.2f,1.0f);
        for (auto &p : g_state.drawnPath) glVertex3f(p.real(), y_base, p.imag()); glEnd();
        // draw polyline
        glLineWidth(3.0f); glBegin(GL_LINE_STRIP); for (auto &p : g_state.drawnPath) glVertex3f(p.real(), y_base, p.imag()); glEnd();
        // test
        ok = testPathContinuityPolyline(g_state.drawnPath, samples, msg);
    } else {
        // straight-line
        std::vector<std::complex<float>> straight = {{g_state.pathStartRe, g_state.pathStartIm},{g_state.pathEndRe, g_state.pathEndIm}};
        ok = testPathContinuityStraight(g_state.pathStartRe, g_state.pathStartIm, g_state.pathEndRe, g_state.pathEndIm, samples, msg);
        // draw start/end
        glPointSize(6.0f); glBegin(GL_POINTS); glColor4f(0.2f,0.8f,1.0f,1.0f); glVertex3f(g_state.pathStartRe,y_base,g_state.pathStartIm); glVertex3f(g_state.pathEndRe,y_base,g_state.pathEndIm); glEnd();
        glLineWidth(3.0f); glBegin(GL_LINES); glVertex3f(g_state.pathStartRe,y_base,g_state.pathStartIm); glVertex3f(g_state.pathEndRe,y_base,g_state.pathEndIm); glEnd();
    }

    // draw sampled track with color indicating ok/fail
    if (!samples.empty()) {
        glLineWidth(4.0f);
        if (ok) glColor4f(0.2f,0.9f,0.2f,0.9f); else glColor4f(0.9f,0.2f,0.2f,0.9f);
        glBegin(GL_LINE_STRIP);
        for (auto &s : samples) glVertex3f(s.z.real(), y_base + 0.01f, s.z.imag());
        glEnd();
        // arrows
        int step = std::max(1, (int)samples.size()/16);
        for (int i=0;i<(int)samples.size()-step;i+=step) {
            auto &a = samples[i]; auto &b = samples[i+step]; float ax=a.z.real(), az=a.z.imag(); float bx=b.z.real(), bz=b.z.imag(); float mx=(ax+bx)/2.0f, mz=(az+bz)/2.0f; float vx=bx-ax, vz=bz-az; float L = std::hypot(vx,vz); if (L<1e-6) continue; vx/=L; vz/=L; glBegin(GL_TRIANGLES); glVertex3f(mx, y_base+0.02f, mz); glVertex3f(mx-0.06f*vz, y_base+0.02f, mz+0.06f*vx); glVertex3f(mx+0.06f*vz, y_base+0.02f, mz-0.06f*vx); glEnd(); }
    }

    // store for UI
    g_state.pathResult = msg;
}

void drawFiberInteraction(){ float y_base=-2.0f; float z_re=g_state.probeRe; float z_im=g_state.probeIm; glPushMatrix(); glTranslatef(z_re,y_base,z_im); glColor4f(1.0f,1.0f,0.2f,1.0f); GLUquadric* q=gluNewQuadric(); gluSphere(q,0.09f,12,12); gluDeleteQuadric(q); glPopMatrix(); std::complex<float> z(z_re,z_im); std::complex<float> A(g_state.branchRe,g_state.branchIm); std::complex<float> w0=complex_sqrt_sheet(z-A,0); std::complex<float> w1=complex_sqrt_sheet(z-A,1); auto heightVal=[&](const std::complex<float>& w)->float{ switch(g_state.heightMode){ case 0: return w.real(); case 1: return w.imag(); case 2: return std::abs(w); default: return w.real(); }}; float y0=heightVal(w0), y1=heightVal(w1); glLineWidth(2.5f); if (g_state.showProjectionLines){ glBegin(GL_LINES); glColor4f(1.0f,1.0f,0.0f,0.5f); glVertex3f(z_re,y_base,z_im); glVertex3f(z_re,y0,z_im); glVertex3f(z_re,y_base,z_im); glVertex3f(z_re,y1,z_im); glEnd(); } glPushMatrix(); glTranslatef(z_re,y0,z_im); glColor4f(0.2f,1.0f,0.3f,1.0f); GLUquadric* q2=gluNewQuadric(); gluSphere(q2,0.09f,12,12); gluDeleteQuadric(q2); glPopMatrix(); glPushMatrix(); glTranslatef(z_re,y1,z_im); glColor4f(0.2f,1.0f,0.3f,1.0f); GLUquadric* q3=gluNewQuadric(); gluSphere(q3,0.09f,12,12); gluDeleteQuadric(q3); glPopMatrix(); }

void drawLocalTrivialization(){ float baseY=-2.0f; float R=0.6f; int segs=20; float cx=g_state.probeRe; float cz=g_state.probeIm; glPushMatrix(); glBegin(GL_TRIANGLE_FAN); glColor4f(0.8f,0.8f,0.2f,0.08f); glVertex3f(cx,baseY+0.001f,cz); for(int i=0;i<=segs;++i){ float a=(float)i/segs*2.0f*PI; glVertex3f(cx+R*std::cos(a),baseY+0.001f,cz+R*std::sin(a)); } glEnd(); std::complex<float> z(cx,cz); std::complex<float> A(g_state.branchRe,g_state.branchIm); std::complex<float> w0=complex_sqrt_sheet(z-A,0); std::complex<float> w1=complex_sqrt_sheet(z-A,1); auto h=[&](const std::complex<float>& w){ switch(g_state.heightMode){ case 0: return w.real(); case 1: return w.imag(); case 2: return std::abs(w);} return w0.real(); }; float y0=h(w0), y1=h(w1); glLineWidth(2.0f); glBegin(GL_LINES); glColor4f(0.9f,0.9f,0.2f,0.25f); for(int i=0;i<6;++i){ float a=(float)i/6.0f*2.0f*PI; float bx=cx+0.4f*std::cos(a); float bz=cz+0.4f*std::sin(a); glVertex3f(bx,baseY,bz); glVertex3f(bx,y0,bz); glVertex3f(bx,baseY,bz); glVertex3f(bx,y1,bz); } glEnd(); glPopMatrix(); }

};

SheafWindow* g_glWin = nullptr; Fl_Box* g_infoBox = nullptr;

// ------------------------------------------------------------------------- // UI & Callbacks // -------------------------------------------------------------------------

void update_ui_text() { if (!g_infoBox) return; std::ostringstream ss; ss << std::fixed << std::setprecision(2); ss << "w^2 = z - A    A = (" << g_state.branchRe << ", " << g_state.branchIm << ")\n"; ss << "Probe z = (" << g_state.probeRe << ", " << g_state.probeIm << ")\n\n"; ss << "Student task (common-sense):\n"; ss << "Pick up a folded map (two-faced). Start at Start and walk to End.\n"; ss << "Can you keep the same face up continuously along the route?\n\n"; ss << "Path test result:\n"; ss << g_state.pathResult << "\n\n"; ss << "Controls:\n"; ss << " - Toggle 'Draw Path Mode' and left-click the GL view to add path vertices.\n"; ss << " - Press 'Clear Path' to remove drawn path.\n"; ss << "Concepts:\n"; ss << "- Base S: z-plane (grid).\n"; ss << "- Total X: Riemann surface (two sheets).\n"; ss << "- Stalk π^{-1}(z): fiber (two values).\n"; ss << "- Ramification at A collapses the fiber.\n"; ss << "- Monodromy: winding around A flips sheets (map face).\n"; g_infoBox->label(ss.str().c_str()); g_infoBox->redraw(); }

void slider_branch_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.branchRe=s->value(); update_ui_text(); if (g_glWin) g_glWin->redraw(); } void slider_branch_im_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.branchIm=s->value(); update_ui_text(); if (g_glWin) g_glWin->redraw(); } void slider_alpha_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.transparency=s->value(); if (g_glWin) g_glWin->redraw(); } void slider_probe_x_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.probeRe=s->value(); update_ui_text(); if (g_glWin) g_glWin->redraw(); } void slider_probe_y_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.probeIm=s->value(); update_ui_text(); if (g_glWin) g_glWin->redraw(); }

void slider_path_sx_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.pathStartRe=s->value(); if (g_glWin) g_glWin->redraw(); } void slider_path_sy_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.pathStartIm=s->value(); if (g_glWin) g_glWin->redraw(); } void slider_path_ex_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.pathEndRe=s->value(); if (g_glWin) g_glWin->redraw(); } void slider_path_ey_cb(Fl_Widget* w, void*){ Fl_Value_Slider* s=(Fl_Value_Slider*)w; g_state.pathEndIm=s->value(); if (g_glWin) g_glWin->redraw(); }

void toggle_fiber_cb(Fl_Widget* w, void*){ g_state.showFiber = ((Fl_Check_Button*)w)->value(); if (g_glWin) g_glWin->redraw(); } void toggle_branchcut_cb(Fl_Widget* w, void*){ g_state.showBranchCut = ((Fl_Check_Button*)w)->value(); if (g_glWin) g_glWin->redraw(); } void toggle_proj_cb(Fl_Widget* w, void*){ g_state.showProjectionLines = ((Fl_Check_Button*)w)->value(); if (g_glWin) g_glWin->redraw(); } void toggle_trivial_cb(Fl_Widget* w, void*){ g_state.showLocalTrivialization = ((Fl_Check_Button*)w)->value(); if (g_glWin) g_glWin->redraw(); }

void choice_height_cb(Fl_Widget* w, void*){ Fl_Choice* c=(Fl_Choice*)w; g_state.heightMode=c->value(); if (g_glWin) g_glWin->redraw(); }

void btn_monodromy_cb(Fl_Widget* w, void*){ Fl_Button* b=(Fl_Button*)w; g_state.animateMonodromy=!g_state.animateMonodromy; if (g_state.animateMonodromy){ b->label("Stop Monodromy"); Fl::add_timeout(0.01, { if(g_glWin) g_glWin->redraw(); }, nullptr);} else { b->label("Trace Monodromy"); } }

void btn_test_path_cb(Fl_Widget* w, void*){ std::vector<PathSample> samples; std::string msg; bool ok = false; if (!g_state.drawnPath.empty() && g_state.drawnPath.size() >= 2) { ok = testPathContinuityPolyline(g_state.drawnPath, samples, msg); } else { ok = testPathContinuityStraight(g_state.pathStartRe, g_state.pathStartIm, g_state.pathEndRe, g_state.pathEndIm, samples, msg); } g_state.pathResult = msg; update_ui_text(); if (g_glWin) g_glWin->redraw(); }

void btn_clear_path_cb(Fl_Widget* w, void*){ g_state.drawnPath.clear(); if (g_glWin) g_glWin->redraw(); }

void toggle_drawmode_cb(Fl_Widget* w, void*){ g_state.drawMode = ((Fl_Check_Button*)w)->value(); }

// ------------------------------------------------------------------------- // Main // ------------------------------------------------------------------------- int main(int argc,char** argv) { Fl_Double_Window* win = new Fl_Double_Window(1100,700,"Grothendieck: Draw Path & Test"); g_glWin = new SheafWindow(10,10,760,680);

Fl_Group* ui = new Fl_Group(780,10,310,680);
g_infoBox = new Fl_Box(790,20,290,260,""); g_infoBox->align(FL_ALIGN_INSIDE|FL_ALIGN_TOP|FL_ALIGN_WRAP); g_infoBox->labelfont(FL_COURIER);

int y = 300;
Fl_Box* label1 = new Fl_Box(780,y,310,25,"Visualization Controls:"); label1->labelfont(FL_HELVETICA_BOLD); y += 28;

Fl_Value_Slider* sA = new Fl_Value_Slider(790,y,270,24,"Branch A (Re)"); sA->bounds(-2.5,2.5); sA->value(g_state.branchRe); sA->callback(slider_branch_cb); y += 44;
Fl_Value_Slider* sA_im = new Fl_Value_Slider(790,y,270,24,"Branch A (Im)"); sA_im->bounds(-2.5,2.5); sA_im->value(g_state.branchIm); sA_im->callback(slider_branch_im_cb); y += 44;
Fl_Value_Slider* sAlpha = new Fl_Value_Slider(790,y,270,24,"Surface Transparency"); sAlpha->bounds(0.05,1.0); sAlpha->value(g_state.transparency); sAlpha->callback(slider_alpha_cb); y += 44;

Fl_Box* fiberSep = new Fl_Box(780,y,310,22,"--- Fiber / Probe ---"); fiberSep->labelfont(FL_HELVETICA_BOLD); y += 26;
Fl_Value_Slider* spx = new Fl_Value_Slider(790,y,270,24,"Probe z (Re)"); spx->bounds(-3.0,3.0); spx->value(g_state.probeRe); spx->callback(slider_probe_x_cb); y += 44;
Fl_Value_Slider* spy = new Fl_Value_Slider(790,y,270,24,"Probe z (Im)"); spy->bounds(-3.0,3.0); spy->value(g_state.probeIm); spy->callback(slider_probe_y_cb); y += 44;
Fl_Check_Button* cbFiber = new Fl_Check_Button(790,y,270,20,"Show Fiber / Stalk"); cbFiber->value(g_state.showFiber); cbFiber->callback(toggle_fiber_cb); y += 26;
Fl_Check_Button* cbProj = new Fl_Check_Button(790,y,270,20,"Show Projection Lines"); cbProj->value(g_state.showProjectionLines); cbProj->callback(toggle_proj_cb); y += 26;
Fl_Check_Button* cbBranchCut = new Fl_Check_Button(790,y,270,20,"Show Branch Cut"); cbBranchCut->value(g_state.showBranchCut); cbBranchCut->callback(toggle_branchcut_cb); y += 26;
Fl_Check_Button* cbTriv = new Fl_Check_Button(790,y,270,20,"Highlight Local Trivialization"); cbTriv->value(g_state.showLocalTrivialization); cbTriv->callback(toggle_trivial_cb); y += 32;

Fl_Box* mappingSep = new Fl_Box(780,y,310,22,"--- Path Test (Draw a Route) ---"); mappingSep->labelfont(FL_HELVETICA_BOLD); y += 26;
Fl_Value_Slider* psx = new Fl_Value_Slider(790,y,130,24,"Start (Re)"); psx->bounds(-3.0,3.0); psx->value(g_state.pathStartRe); psx->callback(slider_path_sx_cb);
Fl_Value_Slider* psy = new Fl_Value_Slider(930,y,130,24,"Start (Im)"); psy->bounds(-3.0,3.0); psy->value(g_state.pathStartIm); psy->callback(slider_path_sy_cb); y += 44;
Fl_Value_Slider* pex = new Fl_Value_Slider(790,y,130,24,"End (Re)"); pex->bounds(-3.0,3.0); pex->value(g_state.pathEndRe); pex->callback(slider_path_ex_cb);
Fl_Value_Slider* pey = new Fl_Value_Slider(930,y,130,24,"End (Im)"); pey->bounds(-3.0,3.0); pey->value(g_state.pathEndIm); pey->callback(slider_path_ey_cb); y += 44;

Fl_Button* testBtn = new Fl_Button(790,y,270,30,"Test Path Continuity"); testBtn->callback(btn_test_path_cb); y += 44;
Fl_Check_Button* cbShowPath = new Fl_Check_Button(790,y,130,20,"Show Path"); cbShowPath->value(g_state.showPath); cbShowPath->callback([](Fl_Widget*,void*){ g_state.showPath = !g_state.showPath; if(g_glWin) g_glWin->redraw(); });
Fl_Check_Button* cbDrawMode = new Fl_Check_Button(930,y,130,20,"Draw Path Mode"); cbDrawMode->value(g_state.drawMode); cbDrawMode->callback(toggle_drawmode_cb); y += 28;
Fl_Button* clearBtn = new Fl_Button(790,y,270,28,"Clear Path"); clearBtn->callback(btn_clear_path_cb); y += 40;

Fl_Box* mappingSep2 = new Fl_Box(780,y,310,22,"--- Height Mapping ---"); mappingSep2->labelfont(FL_HELVETICA_BOLD); y += 26;
Fl_Choice* chMap = new Fl_Choice(790,y,270,24,"Height"); chMap->add("Re(w)"); chMap->add("Im(w)"); chMap->add("|w|"); chMap->value(g_state.heightMode); chMap->callback(choice_height_cb); y += 40;
Fl_Button* btnMono = new Fl_Button(790,y,270,30,"Trace Monodromy"); btnMono->callback(btn_monodromy_cb); y += 44;

ui->end(); win->end(); win->resizable(g_glWin); update_ui_text(); win->show(argc,argv); return Fl::run(); }
