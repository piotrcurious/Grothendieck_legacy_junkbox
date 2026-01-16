// manifold_sheaf.cpp
//
// VISUALIZING SHEAF COHOMOLOGY VIA TEXTURE MAPPING
// Concept: Constructing Global Topology from Local Charts
//
// 1. Base Space: The Circle S1.
// 2. Cover: Two Charts, U (Red/Top) and V (Blue/Bottom).
// 3. Sheaf: The sheaf of local geometric orientations (Texture Coordinates).
// 4. Cohomology: The transition function g_UV at the boundaries.
//    - Trivial Cocycles -> Cylinder (Orientable)
//    - Non-Trivial Cocycles -> Mobius Strip (Non-orientable)

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Check_Button.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <cmath>
#include <vector>
#include <string>
#include <iostream>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Mathematical State (The Sheaf Data)
// -----------------------------------------------------------------------------
struct AppState {
    float rotX = 30.0f, rotY = 45.0f;
    float zoom = 1.0f;
    
    // The Transition Functions (The Cech Cocycle)
    // Overlap 1 (Left side, ~180 deg): Always Identity for simplicity.
    // Overlap 2 (Right side, ~0 deg): User selectable.
    
    // Transition Type at Overlap 2:
    // 0: Identity (v_new = v_old)      -> Makes Cylinder
    // 1: Reflection (v_new = -v_old)   -> Makes Mobius Strip
    // 2: Twist (v_new = v_old + 0.5)   -> Makes Twisted Torus (if closed)
    int transitionType = 0; 
    
    bool showExploded = true; // Separate the charts visually?
    bool showArrows = true;   // Draw vector field
};
AppState g_state;

// -----------------------------------------------------------------------------
// Geometric Helper: The Local Section
// -----------------------------------------------------------------------------
// A "Chart" is a map from the Base Space (Angle theta) to the Total Space (3D world)
// parameterized by local coordinates (u, v).
// u: Angle along the strip
// v: Height (-1 to 1)

void drawArrow(float x, float y, float z, float nx, float ny, float nz, float scale) {
    // Draw a vector at (x,y,z) with direction (nx,ny,nz)
    float ex = x + nx*scale;
    float ey = y + ny*scale;
    float ez = z + nz*scale;
    
    glBegin(GL_LINES);
    glVertex3f(x, y, z);
    glVertex3f(ex, ey, ez);
    glEnd();
    
    // Simple arrowhead?
    glPointSize(4.0f);
    glBegin(GL_POINTS);
    glVertex3f(ex, ey, ez);
    glEnd();
}

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
class ManifoldView : public Fl_Gl_Window {
    int lastX, lastY;
public:
    ManifoldView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
    }

    void draw() override {
        if(!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_NORMALIZE);
        }
        glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45.0, (float)w()/h(), 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 0, 12, 0, 0, 0, 0, 1, 0);
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        // Chart U: The "Left/Top" semi-circle (Red)
        // Covers 10 deg to 350 deg? No, let's do:
        // U: 20 deg to 200 deg (Top/Left)
        // V: 200 deg to 380 deg (Bottom/Right)
        // Overlap 1: near 200 deg. Overlap 2: near 360/0 deg.
        
        // Offset for "Exploded View"
        float r = 3.5f;
        float explode = g_state.showExploded ? 0.4f : 0.0f;
        
        // --- DRAW CHART U (The Reference Chart) ---
        // Range: 10 degrees to 190 degrees
        glPushMatrix();
        glTranslatef(0, explode, 0); // Move Up slightly
        drawStripSection(10, 190, 1.0f, 0.0f, 0.0f); // Red, Transition Identity implies 'Up' is 'Up'
        glPopMatrix();
        
        // --- DRAW CHART V (The Dependent Chart) ---
        // Range: 170 degrees to 350 degrees
        // We must calculate the geometry of V based on the transition function relative to U.
        // At the "Start" (170, which is overlap 1), we enforce Identity gluing (g_UV = 1).
        // At the "End" (350), we approach 370 (10), where the Global Transition happens.
        
        glPushMatrix();
        glTranslatef(0, -explode, 0); // Move Down slightly
        
        // What is the local orientation of V?
        // We define V's local "Up" simply. 
        // The *Gluing* happens visually by checking if the arrows align.
        
        // However, to make the *Manifold*, we twist the geometry of V to match the transition?
        // No, in the Bundle view, V is just V. The *glue* is the identification.
        // But to visualize the *result*, we usually deform V.
        
        // Strategy: Draw V as a flat strip, but flip its TEXTURE/ARROWS if the transition demands it.
        // Wait, if we want to show a Mobius strip, the strip itself must twist.
        // Let's implement the twist linearly across Chart V so they meet.
        
        float twistStart = 0.0f; // Matches U at 180 deg
        float twistEnd = 0.0f;
        if(g_state.transitionType == 1) twistEnd = 180.0f; // Mobius flip (PI)
        
        drawTwistedStrip(170, 350, 0.0f, 0.5f, 1.0f, twistStart, twistEnd);
        
        glPopMatrix();
        
        // --- DRAW GLUING ZONES ---
        drawGlueRegion(180, explode); // Left glue (Identity anchor)
        drawGlueRegion(0, explode);   // Right glue (The Cohomology check)
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
    // Draws a standard strip section with constant orientation
    void drawStripSection(float startDeg, float endDeg, float r, float g, float b) {
        float rad = 3.0f;
        int steps = 30;
        
        for(int i=0; i<steps; ++i) {
            float t1 = (float)i / steps;
            float t2 = (float)(i+1) / steps;
            float a1 = (startDeg + t1*(endDeg-startDeg)) * PI / 180.0f;
            float a2 = (startDeg + t2*(endDeg-startDeg)) * PI / 180.0f;
            
            // Geometric Strip
            glBegin(GL_QUAD_STRIP);
            glColor4f(r, g, b, 0.7f);
            
            // "Up" is Y-axis relative to the strip center? 
            // Let's say strip is in XZ plane, width in Y.
            // NO, strip is circle in XZ, width in Y (Cylinder wall)
            
            // Top edge (y=1)
            glNormal3f(cos(a1), 0, sin(a1)); // Normal points out
            glVertex3f(rad*cos(a1), 1.0f, rad*sin(a1));
            glVertex3f(rad*cos(a1), -1.0f, rad*sin(a1));
            
            glVertex3f(rad*cos(a2), 1.0f, rad*sin(a2));
            glVertex3f(rad*cos(a2), -1.0f, rad*sin(a2));
            glEnd();
            
            // Draw Arrows (The Section)
            if(g_state.showArrows && i % 5 == 0) {
                glColor3f(1, 1, 1);
                // Arrow points UP (Y+)
                float cx = rad*cos(a1);
                float cz = rad*sin(a1);
                drawArrow(cx, 0, cz, 0, 1, 0, 0.8f);
            }
        }
    }

    // Draws a strip that twists its local "Up" vector
    // twistAng: 0 = Up, 180 = Down
    void drawTwistedStrip(float startDeg, float endDeg, float r, float g, float b, float tStart, float tEnd) {
        float rad = 3.0f;
        int steps = 30;
        
        for(int i=0; i<steps; ++i) {
            float param1 = (float)i / steps;
            float param2 = (float)(i+1) / steps;
            
            float a1 = (startDeg + param1*(endDeg-startDeg)) * PI / 180.0f;
            float a2 = (startDeg + param2*(endDeg-startDeg)) * PI / 180.0f;
            
            float twist1 = (tStart + param1*(tEnd-tStart)) * PI / 180.0f;
            float twist2 = (tStart + param2*(tEnd-tStart)) * PI / 180.0f;
            
            // Calculate edge positions based on twist
            // The "Height" vector rotates around the radial vector
            // Center C = (R cos a, 0, R sin a)
            // Radial R = (cos a, 0, sin a)
            // Tangent T = (-sin a, 0, cos a)
            // Vertical Y = (0, 1, 0)
            
            // The strip width vector w:
            // w = Y * cos(twist) + R * sin(twist) ?? 
            // Standard Mobius twist is around the Tangent.
            // So the "Vertical" vector tilts towards the "Radial" vector.
            
            auto getEdge = [&](float angle, float tw, float sign) {
                float ca = cos(angle), sa = sin(angle);
                float ct = cos(tw), st = sin(tw);
                // Rotated vertical vector
                float vx = st * ca; // Project onto radial x
                float vy = ct;      // Project onto y
                float vz = st * sa; // Project onto radial z
                
                return std::vector<float>{
                    rad*ca + sign*vx, sign*vy, rad*sa + sign*vz
                };
            };
            
            auto p1_top = getEdge(a1, twist1, 1.0f);
            auto p1_bot = getEdge(a1, twist1, -1.0f);
            auto p2_top = getEdge(a2, twist2, 1.0f);
            auto p2_bot = getEdge(a2, twist2, -1.0f);

            glBegin(GL_QUAD_STRIP);
            glColor4f(r, g, b, 0.7f);
            glVertex3f(p1_top[0], p1_top[1], p1_top[2]);
            glVertex3f(p1_bot[0], p1_bot[1], p1_bot[2]);
            glVertex3f(p2_top[0], p2_top[1], p2_top[2]);
            glVertex3f(p2_bot[0], p2_bot[1], p2_bot[2]);
            glEnd();
            
            // Draw Arrows
            if(g_state.showArrows && i % 5 == 0) {
                glColor3f(1, 1, 1);
                // Calculate mid orientation
                float cm_a = a1; float cm_t = twist1;
                auto top = getEdge(cm_a, cm_t, 1.0f);
                float cx = rad*cos(cm_a), cz = rad*sin(cm_a);
                // Vector from center to top edge
                drawArrow(cx, 0, cz, top[0]-cx, top[1], top[2], 0.8f);
            }
        }
    }

    void drawGlueRegion(float angleDeg, float explode) {
        float rad = 3.0f;
        float a = angleDeg * PI / 180.0f;
        float cx = rad * cos(a);
        float cz = rad * sin(a);
        
        // Draw a transparent box indicating the overlap zone
        glPushMatrix();
        glTranslatef(cx, 0, cz);
        glRotatef(-angleDeg, 0, 1, 0); // Align with tangent
        
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        glLineWidth(2.0f);
        
        // Visual connector
        if (g_state.showExploded) {
            glColor3f(1, 1, 0);
            glBegin(GL_LINES);
            glVertex3f(0, explode - 1.0f, 0);
            glVertex3f(0, -explode + 1.0f, 0);
            glEnd();
        }
        
        glDisable(GL_LINE_STIPPLE);
        glPopMatrix();
    }
};

// -----------------------------------------------------------------------------
// UI Construction
// -----------------------------------------------------------------------------
ManifoldView* glWin = nullptr;

void update_cb(Fl_Widget* w, void*) {
    if(glWin) glWin->redraw();
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1000, 600, "Grothendieck Bundle Constructor");
    
    Fl_Group* grp = new Fl_Group(10, 10, 250, 580);
    
    Fl_Box* title = new Fl_Box(10, 20, 240, 40, "Bundle Glue");
    title->labelfont(FL_BOLD); title->labelsize(20); title->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);

    Fl_Box* desc = new Fl_Box(10, 60, 240, 100, 
        "Constructing a Manifold.\n"
        "Red Chart: U (Source)\n"
        "Blue Chart: V (Target)\n\n"
        "Define the Transition Function\n"
        "on the right-side overlap:");
    desc->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);
    desc->box(FL_BORDER_BOX);

    Fl_Choice* ch = new Fl_Choice(10, 180, 240, 30, "Transition g_UV");
    ch->add("Identity (Cylinder) | g(v)=v");
    ch->add("Inversion (Mobius) | g(v)=-v");
    ch->value(0);
    ch->callback([](Fl_Widget* w, void*){
        g_state.transitionType = ((Fl_Choice*)w)->value();
        update_cb(nullptr, nullptr);
    });

    Fl_Check_Button* bExp = new Fl_Check_Button(10, 230, 240, 30, "Explode Charts (Show Overlap)");
    bExp->value(1);
    bExp->callback([](Fl_Widget* w, void*){
        g_state.showExploded = ((Fl_Check_Button*)w)->value();
        update_cb(nullptr, nullptr);
    });

    Fl_Check_Button* bArr = new Fl_Check_Button(10, 260, 240, 30, "Show Local Sections (Arrows)");
    bArr->value(1);
    bArr->callback([](Fl_Widget* w, void*){
        g_state.showArrows = ((Fl_Check_Button*)w)->value();
        update_cb(nullptr, nullptr);
    });
    
    Fl_Box* expl = new Fl_Box(10, 320, 240, 200, 
        "CONCEPT CHECK:\n"
        "A Bundle is defined by\n"
        "Projection: E -> B\n"
        "Fiber F\n"
        "Structure Group G\n\n"
        "Here G = Z_2 = {1, -1}.\n"
        "The cohomology class of\n"
        "the transition functions\n"
        "[g] in H^1(S^1, Z_2)\n"
        "determines the topology.");
    expl->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);
    expl->labelsize(12);

    grp->end();

    glWin = new ManifoldView(270, 10, 720, 580);
    glWin->end();

    win->resizable(glWin);
    win->show(argc, argv);
    return Fl::run();
}
