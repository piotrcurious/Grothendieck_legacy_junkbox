// grothendieck_log.cpp
// 
// THE GROTHENDIECK TOOLKIT: CECH COHOMOLOGY OF THE LOGARITHM SHEAF
//
// Concepts Visualized:
// 1. Open Cover {U, V}: We split the domain into two overlapping sets.
// 2. Local Sections (s_U, s_V): We define log(z) locally on each set.
// 3. Presheaf Restriction: We look at the sections only on the overlap.
// 4. Cech Cocycle (s_U - s_V): The obstruction to global gluing.
// 5. Etale Space: The infinite Helicoid representing all possible germs.

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Output.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <complex>
#include <vector>
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>

using Complex = std::complex<float>;
static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Math & Sheaf Logic
// -----------------------------------------------------------------------------

// We define two Open Sets in the Base Space C \ {0}
// Set U: The "Left" side (-20 deg to 200 deg)
// Set V: The "Right" side (160 deg to 380 deg)
// They overlap at the Top (approx 90 deg) and Bottom (approx 270 deg).

bool inSetU(float angleDeg) {
    // Normalize to 0-360
    while(angleDeg < 0) angleDeg += 360;
    while(angleDeg >= 360) angleDeg -= 360;
    // U covers roughly 20 to 340 (avoiding the positive real axis cut for V)
    // Let's make U: 10 degrees to 350 degrees? 
    // Actually, let's use standard overlapping half-planes for clarity.
    
    // U: Arg in (-PI/2 - eps, PI/2 + eps)  -> Right Half
    // V: Arg in (PI/2 - eps, 3PI/2 + eps)  -> Left Half
    // Overlap 1: Top (near PI/2)
    // Overlap 2: Bottom (near -PI/2 or 3PI/2)
    
    // To allow standard log branches, let's rotate the sets.
    // U = (-10, 190) deg
    // V = (170, 370) deg
    return (angleDeg > -20.0f && angleDeg < 200.0f) || (angleDeg > 340.0f); 
}

// -----------------------------------------------------------------------------
// State
// -----------------------------------------------------------------------------
struct AppState {
    float rotX = 30.0f, rotY = -45.0f;
    float zoom = 0.8f;
    
    // Sheet Selection (The Integer 'k' in log(z) + 2*pi*i*k)
    int k_U = 0; // Branch choice for Set U
    int k_V = 0; // Branch choice for Set V
    
    bool showU = true;
    bool showV = true;
    bool showHelicoid = true; // The full Etale space ghost
    bool showCocycle = true;  // Highlight the difference
    
    // Metrics
    std::string overlapTopStatus = "";
    std::string overlapBotStatus = "";
} g_state;

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
class CohomologyView : public Fl_Gl_Window {
    int lastX, lastY;

public:
    CohomologyView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
    }

    void draw() override {
        if(!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_LINE_SMOOTH);
        }
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45.0, (float)w()/h(), 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 0, 18, 0, 0, 0, 0, 1, 0); // Looking down/forward
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        // 1. Draw the Base Plane Covers (The Open Sets U and V)
        drawBaseCovers();

        // 2. Draw the Ghost Etale Space (The full Helicoid)
        if(g_state.showHelicoid) drawHelicoidGhost();

        // 3. Draw the Local Sections s_U and s_V
        if(g_state.showU) drawSection(0, g_state.k_U); // Set 0 = U
        if(g_state.showV) drawSection(1, g_state.k_V); // Set 1 = V
        
        // 4. Visualize the Cohomology (The vertical gaps)
        if(g_state.showCocycle) drawGlues();
    }

    int handle(int e) override {
        if(e == FL_PUSH) { lastX = Fl::event_x(); lastY = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            g_state.rotY += (Fl::event_x() - lastX);
            g_state.rotX += (Fl::event_y() - lastY);
            lastX = Fl::event_x(); lastY = Fl::event_y();
            redraw();
            return 1;
        }
        return Fl_Gl_Window::handle(e);
    }

private:
    // Helper to get 3D coord of log(z) on a specific branch
    void getLogPoint(float r, float thetaRad, int k, float pts[3]) {
        // z = r * exp(i*theta)
        // w = ln(r) + i*(theta + 2*pi*k)
        // We map w to 3D: x = Re(z), z = Im(z), y = Im(w)/Scale
        // Wait, standard visualization:
        // x = Re(z), z = Im(z) (The Base Plane)
        // y = Im(w) (The imaginary part of the function value)
        // We ignore Re(w)=ln(r) for height, or we look weird. 
        // The "Riemann Surface of log z" is usually plotted as (x, y, Im(log z)).
        
        float x = r * cos(thetaRad);
        float z = r * sin(thetaRad);
        float y = (thetaRad + 2.0f * PI * k) * 0.5f; // Scale height
        
        pts[0] = x; pts[1] = y; pts[2] = z;
    }

    void drawBaseCovers() {
        float y = -5.0f;
        glLineWidth(2.0f);
        
        // Draw Set U (Red Sector) - Right side roughly
        // -20 deg to 200 deg
        glColor4f(1.0f, 0.0f, 0.0f, 0.2f);
        drawSectorStrip(y, -20 * PI/180.0f, 200 * PI/180.0f);

        // Draw Set V (Blue Sector) - Left side roughly
        // 160 deg to 380 deg (overlap is 160-200 and 340-380)
        glColor4f(0.0f, 0.0f, 1.0f, 0.2f);
        drawSectorStrip(y, 160 * PI/180.0f, 380 * PI/180.0f);
        
        // Labels (using simple lines/quads)
    }

    void drawSectorStrip(float y, float startAng, float endAng) {
        float rMin = 2.0f, rMax = 5.0f;
        int segs = 40;
        glBegin(GL_QUAD_STRIP);
        for(int i=0; i<=segs; ++i) {
            float t = (float)i/segs;
            float ang = startAng + t*(endAng - startAng);
            glVertex3f(rMin*cos(ang), y, rMin*sin(ang));
            glVertex3f(rMax*cos(ang), y, rMax*sin(ang));
        }
        glEnd();
    }

    void drawHelicoidGhost() {
        // The Etale space E.
        // Transparent grey spiral.
        glColor4f(1.0f, 1.0f, 1.0f, 0.05f);
        glEnable(GL_CULL_FACE); // Optimization
        for(int k=-2; k<=2; ++k) {
            float rMin=2, rMax=5;
            int segs = 60;
            glBegin(GL_QUAD_STRIP);
            for(int i=0; i<=segs; ++i) {
                float ang = -PI + 2.0f*PI * ((float)i/segs);
                float p1[3], p2[3];
                getLogPoint(rMin, ang, k, p1);
                getLogPoint(rMax, ang, k, p2);
                glVertex3fv(p1); glVertex3fv(p2);
            }
            glEnd();
        }
        glDisable(GL_CULL_FACE);
    }

    void drawSection(int setID, int k) {
        // Define angular range for the sets
        float startAng, endAng;
        float rCol[3];
        
        if (setID == 0) { // Set U
            startAng = -20 * PI/180.0f;
            endAng = 200 * PI/180.0f;
            rCol[0]=1; rCol[1]=0.3; rCol[2]=0.3; // Reddish
        } else { // Set V
            startAng = 160 * PI/180.0f;
            endAng = 380 * PI/180.0f;
            rCol[0]=0.3; rCol[1]=0.3; rCol[2]=1; // Bluish
        }

        // Draw the surface patch
        glColor4f(rCol[0], rCol[1], rCol[2], 0.8f);
        float rMin=2, rMax=5;
        int segs = 40;
        
        // Mesh
        glBegin(GL_QUAD_STRIP);
        for(int i=0; i<=segs; ++i) {
            float t = (float)i/segs;
            float ang = startAng + t*(endAng - startAng);
            
            // NOTE: Key Grothendieck concept here.
            // The function is defined LOCALLY. 
            // For Set V, if ang > PI, we are wrapping.
            // But we treat the domain as simply connected for the section definition.
            
            float p1[3], p2[3];
            getLogPoint(rMin, ang, k, p1);
            getLogPoint(rMax, ang, k, p2);
            
            // Lighting hack based on normal
            glVertex3fv(p1);
            glVertex3fv(p2);
        }
        glEnd();
        
        // Draw Wireframe outline
        glLineWidth(2.0f);
        glColor3f(1,1,1);
        glBegin(GL_LINE_LOOP);
            for(int i=0; i<=segs; ++i) {
                float t = (float)i/segs;
                float ang = startAng + t*(endAng - startAng);
                float p[3]; getLogPoint(rMax, ang, k, p); glVertex3fv(p);
            }
            for(int i=segs; i>=0; --i) {
                float t = (float)i/segs;
                float ang = startAng + t*(endAng - startAng);
                float p[3]; getLogPoint(rMin, ang, k, p); glVertex3fv(p);
            }
        glEnd();
    }

    void drawGlues() {
        // We have two overlaps:
        // 1. "Top" Intersection: roughly 160 to 200 degrees.
        // 2. "Bottom" Intersection: roughly 340 to 380 (-20) degrees.
        
        // Calculate the difference between Section U and Section V at these centers.
        
        // Top Center ~ 180 deg (PI)
        float angTop = PI; 
        float pU_top[3], pV_top[3];
        getLogPoint(5.0f, angTop, g_state.k_U, pU_top);
        getLogPoint(5.0f, angTop, g_state.k_V, pV_top);
        
        // Draw connection line
        drawErrorBar(pU_top, pV_top);
        
        // Bottom Center ~ 0 deg (0)
        // Note: For Set V, 0 degrees is actually 360 (2PI) in our parameterization above
        float angBotU = 0.0f;
        float angBotV = 2.0f * PI; // Because V wraps around 160->380
        
        float pU_bot[3], pV_bot[3];
        getLogPoint(5.0f, angBotU, g_state.k_U, pU_bot);
        getLogPoint(5.0f, angBotV, g_state.k_V, pV_bot);
        
        drawErrorBar(pU_bot, pV_bot);
        
        // Update Status Strings for UI
        float diffTop = (pU_top[1] - pV_top[1]) / (PI); // In units of PI roughly
        float diffBot = (pU_bot[1] - pV_bot[1]) / (PI);
        
        std::stringstream ss;
        if (std::abs(pU_top[1] - pV_top[1]) < 0.1) ss << "Top: GLUED (0)";
        else ss << "Top: GAP (" << (g_state.k_U - g_state.k_V) << "*2pi*i)";
        g_state.overlapTopStatus = ss.str();
        
        std::stringstream ss2;
        // The V angle was 2PI, U was 0. So inherent difference of 1 sheet if k's are equal.
        int k_diff_effective = g_state.k_U - (g_state.k_V + 1);
        if (std::abs(pU_bot[1] - pV_bot[1]) < 0.1) ss2 << "Bot: GLUED (0)";
        else ss2 << "Bot: GAP (" << k_diff_effective << "*2pi*i)";
        g_state.overlapBotStatus = ss2.str();
    }
    
    void drawErrorBar(float p1[3], float p2[3]) {
        if (std::abs(p1[1] - p2[1]) < 0.05f) {
            // Glued! Green
            glPointSize(10.0f);
            glColor3f(0, 1, 0);
            glBegin(GL_POINTS); glVertex3fv(p1); glEnd();
        } else {
            // Gap! Red dashed line
            glLineWidth(3.0f);
            glColor3f(1, 0, 1); // Magenta
            glBegin(GL_LINES); glVertex3fv(p1); glVertex3fv(p2); glEnd();
            
            // Arrow heads
            glPointSize(6.0f);
            glBegin(GL_POINTS); glVertex3fv(p1); glVertex3fv(p2); glEnd();
        }
    }
};

// -----------------------------------------------------------------------------
// UI Construction
// -----------------------------------------------------------------------------
CohomologyView* glWin = nullptr;
Fl_Output* outTop = nullptr;
Fl_Output* outBot = nullptr;

void update_cb(Fl_Widget*, void*) {
    if(glWin) glWin->redraw();
    // Force redraw to update strings in drawGlues first? 
    // Actually draw() updates strings, we need to flush.
    glWin->redraw();
    Fl::flush(); // Hack to ensure GL draw runs
    if(outTop) outTop->value(g_state.overlapTopStatus.c_str());
    if(outBot) outBot->value(g_state.overlapBotStatus.c_str());
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1200, 700, "Sheaf Cohomology Visualizer");

    Fl_Group* grp = new Fl_Group(10, 10, 300, 680);
    
    Fl_Box* title = new Fl_Box(10, 15, 290, 40, "THE LOGARITHM SHEAF");
    title->labelfont(FL_BOLD); title->labelsize(16);
    
    Fl_Box* desc = new Fl_Box(10, 50, 290, 100, 
        "Visualizing H^1(X, Z).\n"
        "Can you glue the local sections\n"
        "(Red and Blue) to make a global one?\n"
        "Hint: The spiral never closes.");
    desc->align(FL_ALIGN_INSIDE | FL_ALIGN_WRAP);
    desc->box(FL_BORDER_BOX);

    int y = 160;
    
    // U Controls
    Fl_Box* lblU = new Fl_Box(10, y, 290, 20, "Section s_U (Red) - Branch k");
    lblU->labelfont(FL_BOLD); lblU->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    y+=25;
    Fl_Value_Slider* sldU = new Fl_Value_Slider(10, y, 290, 25);
    sldU->type(FL_HOR_NICE_SLIDER); sldU->bounds(-2, 2); sldU->step(1); sldU->value(0);
    sldU->callback([](Fl_Widget* w, void*){ g_state.k_U = (int)((Fl_Value_Slider*)w)->value(); update_cb(nullptr,nullptr); });
    y+=40;

    // V Controls
    Fl_Box* lblV = new Fl_Box(10, y, 290, 20, "Section s_V (Blue) - Branch k");
    lblV->labelfont(FL_BOLD); lblV->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    y+=25;
    Fl_Value_Slider* sldV = new Fl_Value_Slider(10, y, 290, 25);
    sldV->type(FL_HOR_NICE_SLIDER); sldV->bounds(-2, 2); sldV->step(1); sldV->value(0);
    sldV->callback([](Fl_Widget* w, void*){ g_state.k_V = (int)((Fl_Value_Slider*)w)->value(); update_cb(nullptr,nullptr); });
    y+=40;

    Fl_Check_Button* chkH = new Fl_Check_Button(10, y, 290, 25, "Show Total Etale Space (Ghost)");
    chkH->value(1);
    chkH->callback([](Fl_Widget* w, void*){ g_state.showHelicoid = ((Fl_Check_Button*)w)->value(); update_cb(nullptr,nullptr); });
    y+=30;

    // Output
    Fl_Box* lblRes = new Fl_Box(10, y, 290, 20, "Restriction & Gluing Status:");
    lblRes->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    y+=25;
    
    outTop = new Fl_Output(10, y, 290, 30);
    y+=40;
    outBot = new Fl_Output(10, y, 290, 30);
    
    grp->end();

    glWin = new CohomologyView(320, 10, 870, 680);
    glWin->end();

    win->resizable(glWin);
    win->show(argc, argv);
    
    // Initial update
    update_cb(nullptr, nullptr);
    
    return Fl::run();
}
