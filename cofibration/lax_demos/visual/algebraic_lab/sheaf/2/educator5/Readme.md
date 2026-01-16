To visualize the abstract concept of a Sheaf rigorously, we need to visualize the **Espace Étale** (Etale Space).

The crucial insight of Grothendieck/Leray is that a sheaf  on a base  is equivalent to a geometric space  hovering over , where the "fibers" (stalks) vary continuously. A **Section** is just a continuous path through this space.

This code implements a **Sheaf Laboratory** with three distinct sheaf types that demonstrate key properties:

1. **The Constant Sheaf ()**: Demonstrates Triviality and Covering Spaces.
2. **The Möbius Sheaf**: Demonstrates Non-Trivial Bundles and Monodromy (Twisting).
3. **The Skyscraper Sheaf**: Demonstrates Localization and Support (zero everywhere except at a point).

### Compile Instructions

Save as `sheaf_lab.cpp`.

```bash
g++ -o sheaf_lab sheaf_lab.cpp -lfltk -lfltk_gl -lGL -lGLU

```

### The Code

```cpp
// sheaf_lab.cpp
//
// THE SHEAF LABORATORY: A Rigorous Etale Space Visualizer
//
// Concepts Visualized:
// 1. Espace Etale: The topological space of all germs.
// 2. Stalks: The fiber over a point x.
// 3. Sections: Continuous maps s: U -> E.
// 4. Restriction: Viewing a section on a smaller open set U.
// 5. Monodromy: How the stalk twists as you traverse the base.
//
// Sheaves Implemented:
// - Constant Sheaf (Disjoint Sheets)
// - Mobius Sheaf (Twisted Line Bundle)
// - Skyscraper Sheaf (Supported at a point)

#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Box.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <cmath>
#include <vector>
#include <string>
#include <iostream>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Abstract Sheaf Strategy
// -----------------------------------------------------------------------------
// Defines the topology of the Etale space for a specific sheaf type.
class SheafStrategy {
public:
    virtual ~SheafStrategy() {}
    
    // Returns the position in 3D space of a value 'val' in the fiber over angle 'theta'
    // theta: 0 to 2*PI (Base Space coordinate)
    // val: The value in the stalk (vertical coordinate)
    virtual void getPoint(float theta, float val, float out[3]) = 0;
    
    // Returns the "twist" or monodromy factor at this angle
    // Used to orient the visualization of the fiber (e.g., rotating the line)
    virtual float getFiberRotation(float theta) { return 0.0f; }
    
    // For visualization: What is the valid range of values in the stalk?
    virtual float getMinVal() { return -2.0f; }
    virtual float getMaxVal() { return 2.0f; }
    
    // Description for the UI
    virtual const char* getName() = 0;
    virtual const char* getDescription() = 0;
};

// 1. The Constant Sheaf (Z or R) over S1
// Topologically: A cylinder (S1 x R)
// Sections: Horizontal circles.
class ConstantSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3]) override {
        float r = 3.0f;
        out[0] = r * cos(theta); // X
        out[1] = val;            // Y (Height is simply the value)
        out[2] = r * sin(theta); // Z
    }
    const char* getName() override { return "Constant Sheaf (Cylinder)"; }
    const char* getDescription() override { 
        return "Trivial Topology.\n"
               "Global Sections exist (Loops).\n"
               "Fiber is constant."; 
    }
};

// 2. The Mobius Sheaf (Twisted Real Line)
// Topologically: A Mobius Strip
// The "value" axis rotates by 180 degrees as you go around the circle.
class MobiusSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3]) override {
        float r = 3.0f;
        // The fiber rotates around the tangent vector of the circle.
        // Or simpler: The "Vertical" direction tilts.
        // Let's model the standard Mobius embedding in 3D.
        
        // Twist angle: theta/2. 
        // At theta=0, twist=0 (Vertical). At theta=2PI, twist=PI (Inverted).
        float twist = theta * 0.5f;
        
        // Center of fiber
        float cx = r * cos(theta);
        float cy = 0.0f;
        float cz = r * sin(theta);
        
        // The fiber vector direction (tilting in the radial plane for visibility)
        // Standard mobius: v = val * sin(twist/2) ...
        // Let's construct explicitly:
        // We rotate the vector (0, val, 0) by 'twist' around the Radial vector.
        // Actually, Mobius is usually rotating around the tangent, so the strip lies "flat" then "vertical".
        // Let's do: Center + val * (Direction)
        
        // Direction vector rotating in the (Y, Radial) plane
        float dr = sin(twist); // Radial component
        float dy = cos(twist); // Y component
        
        out[0] = cx + val * dr * cos(theta); // Add radial component projected to X
        out[1] = cy + val * dy;              // Y component
        out[2] = cz + val * dr * sin(theta); // Add radial component projected to Z
    }
    
    float getFiberRotation(float theta) override { return theta * 0.5f; }
    
    const char* getName() override { return "Mobius Sheaf (Twisted)"; }
    const char* getDescription() override { 
        return "Non-Trivial Topology.\n"
               "No Global Section (except 0).\n"
               "Try to close the loop!"; 
    }
};

// 3. The Skyscraper Sheaf (Supported at x=0)
// Value is 0 everywhere, except at theta=PI (Back of circle) where it is R.
// Visualized as a flat line that explodes into a vertical line at one point.
class SkyscraperSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3]) override {
        float r = 3.0f;
        
        // Define the "Support" region (small angle around PI)
        float dist = std::abs(theta - PI);
        if (dist < 0.2f) {
            // Inside the skyscraper stalk
            // We scale the value to show it exists here
            out[1] = val; 
        } else {
            // Outside support: The stalk is just the point 0
            // We force visual value to 0 regardless of input 'val'
            // (Unless we want to visualize the 'zero section' explicitly)
            out[1] = 0.0f;
        }
        
        out[0] = r * cos(theta);
        out[2] = r * sin(theta);
    }
    const char* getName() override { return "Skyscraper Sheaf"; }
    const char* getDescription() override { 
        return "Supported only at point P.\n"
               "Stalk F_x = 0 if x != P.\n"
               "Stalk F_p = R."; 
    }
};

// -----------------------------------------------------------------------------
// App State
// -----------------------------------------------------------------------------
struct AppState {
    float rotX = 20.0f, rotY = -30.0f;
    float zoom = 1.0f;
    
    SheafStrategy* currentSheaf = nullptr;
    
    // The "Open Set" U (an interval on the circle)
    float u_center = PI; // Center of the open set
    float u_width = PI/2.0f; // Width of the open set
    
    // The "Section" defined on U
    // Modeled as a simple value/function s(x) = height
    float sectionValue = 1.0f; // The "height" of the section
    
    // Interactive Probe
    float probeAngle = 0.0f;
    
    bool showEtale = true;   // Show the ghost mesh
    bool showStalks = false; // Show vertical lines everywhere?
    bool showProbe = true;   // Show the specific stalk at cursor
};
AppState g_state;

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
class SheafView : public Fl_Gl_Window {
    int lastX, lastY;
public:
    SheafView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
    }

    void draw() override {
        if(!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45.0, (float)w()/h(), 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 5, 12, 0, 0, 0, 0, 1, 0);
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        if(g_state.currentSheaf) {
            drawBaseSpace();
            if(g_state.showEtale) drawEtaleSpace();
            drawSectionOnU();
            if(g_state.showProbe) drawProbe();
        }
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
    void drawBaseSpace() {
        // Draw the Circle S1 on the ground
        glColor4f(0.5f, 0.5f, 0.5f, 0.5f);
        glBegin(GL_LINE_LOOP);
        for(int i=0; i<100; ++i) {
            float t = 2.0f*PI*i/100.0f;
            glVertex3f(3.0f*cos(t), -2.5f, 3.0f*sin(t)); // Draw below the main action
        }
        glEnd();
        
        // Highlight the Open Set U on the base
        float start = g_state.u_center - g_state.u_width/2.0f;
        float end = g_state.u_center + g_state.u_width/2.0f;
        
        glLineWidth(4.0f);
        glColor3f(0.2f, 0.8f, 0.2f); // Green U
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=50; ++i) {
            float t = start + (end-start)*i/50.0f;
            glVertex3f(3.0f*cos(t), -2.5f, 3.0f*sin(t));
        }
        glEnd();
        glLineWidth(1.0f);
    }

    void drawEtaleSpace() {
        // Visualize the total space E as a ghost mesh
        // We draw "fibers" densely to create a surface look
        glColor4f(1.0f, 1.0f, 1.0f, 0.15f); // Ghostly white
        
        int segments = 80;
        float minV = g_state.currentSheaf->getMinVal();
        float maxV = g_state.currentSheaf->getMaxVal();
        
        for(int i=0; i<segments; ++i) {
            float t1 = 2.0f*PI*i/segments;
            float t2 = 2.0f*PI*(i+1)/segments;
            
            glBegin(GL_QUAD_STRIP);
            // Draw a strip for the fiber range
            float p[3];
            g_state.currentSheaf->getPoint(t1, minV, p); glVertex3fv(p);
            g_state.currentSheaf->getPoint(t1, maxV, p); glVertex3fv(p);
            g_state.currentSheaf->getPoint(t2, minV, p); glVertex3fv(p);
            g_state.currentSheaf->getPoint(t2, maxV, p); glVertex3fv(p);
            glEnd();
        }
        
        // Wireframe edges
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        glBegin(GL_LINE_LOOP); // Top edge
        for(int i=0; i<=segments; ++i) {
            float t = 2.0f*PI*i/segments;
            float p[3]; g_state.currentSheaf->getPoint(t, maxV, p); glVertex3fv(p);
        }
        glEnd();
        glBegin(GL_LINE_LOOP); // Bottom edge
        for(int i=0; i<=segments; ++i) {
            float t = 2.0f*PI*i/segments;
            float p[3]; g_state.currentSheaf->getPoint(t, minV, p); glVertex3fv(p);
        }
        glEnd();
    }

    void drawSectionOnU() {
        // A section s: U -> E
        // We evaluate the sheaf at 'sectionValue' over the interval U.
        // For Mobius, simply holding a constant 'val' parameter creates a path that might flip in 3D,
        // effectively visualizing the local section.
        
        float start = g_state.u_center - g_state.u_width/2.0f;
        float end = g_state.u_center + g_state.u_width/2.0f;
        
        glLineWidth(3.0f);
        glColor3f(1.0f, 0.2f, 0.2f); // Red Section
        
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=100; ++i) {
            float t = start + (end-start)*i/100.0f;
            float p[3];
            // Get the point in Etale space
            // Note: For Skyscraper, if t is not in support, this drops to 0 automatically.
            g_state.currentSheaf->getPoint(t, g_state.sectionValue, p);
            glVertex3fv(p);
        }
        glEnd();
        
        // Draw "Germs" (thick points) at ends to emphasize local nature
        glPointSize(8.0f);
        glBegin(GL_POINTS);
        float p1[3], p2[3];
        g_state.currentSheaf->getPoint(start, g_state.sectionValue, p1);
        g_state.currentSheaf->getPoint(end, g_state.sectionValue, p2);
        glVertex3fv(p1); glVertex3fv(p2);
        glEnd();
    }

    void drawProbe() {
        // Draw the Stalk (Fiber) at the probe angle
        float t = g_state.probeAngle;
        float minV = g_state.currentSheaf->getMinVal();
        float maxV = g_state.currentSheaf->getMaxVal();
        
        float pBot[3], pTop[3];
        g_state.currentSheaf->getPoint(t, minV, pBot);
        g_state.currentSheaf->getPoint(t, maxV, pTop);
        
        // The Stalk Line
        glLineWidth(1.0f);
        glColor3f(1.0f, 1.0f, 0.0f); // Yellow Stalk
        glEnable(GL_LINE_STIPPLE);
        glLineStipple(1, 0x00FF);
        glBegin(GL_LINES);
        glVertex3fv(pBot); glVertex3fv(pTop);
        glEnd();
        glDisable(GL_LINE_STIPPLE);
        
        // The Value Point on the Stalk
        float pVal[3];
        g_state.currentSheaf->getPoint(t, g_state.sectionValue, pVal);
        
        glPointSize(10.0f);
        glColor3f(1.0f, 0.5f, 0.0f); // Orange point
        glBegin(GL_POINTS);
        glVertex3fv(pVal);
        glEnd();
    }
};

// -----------------------------------------------------------------------------
// UI Construction
// -----------------------------------------------------------------------------
SheafView* glWin = nullptr;
Fl_Box* descBox = nullptr;

ConstantSheaf sConst;
MobiusSheaf sMobius;
SkyscraperSheaf sSky;

void update_desc() {
    if(g_state.currentSheaf && descBox) {
        descBox->label(g_state.currentSheaf->getDescription());
    }
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1280, 800, "Sheaf Laboratory");
    
    Fl_Group* grp = new Fl_Group(10, 10, 300, 780);
    
    Fl_Box* title = new Fl_Box(10, 20, 290, 40, "The Sheaf Laboratory");
    title->labelfont(FL_BOLD); title->labelsize(20);
    
    // Sheaf Selector
    Fl_Choice* chSheaf = new Fl_Choice(10, 80, 290, 30, "Select Sheaf Type");
    chSheaf->add("Constant Sheaf (Z)");
    chSheaf->add("Mobius Sheaf");
    chSheaf->add("Skyscraper Sheaf");
    chSheaf->value(0);
    g_state.currentSheaf = &sConst; // Default
    
    chSheaf->callback([](Fl_Widget* w, void*){
        int v = ((Fl_Choice*)w)->value();
        if(v==0) g_state.currentSheaf = &sConst;
        else if(v==1) g_state.currentSheaf = &sMobius;
        else if(v==2) g_state.currentSheaf = &sSky;
        update_desc();
        if(glWin) glWin->redraw();
    });

    // Description Box
    descBox = new Fl_Box(10, 120, 290, 80, "Trivial Topology.");
    descBox->box(FL_BORDER_BOX);
    descBox->align(FL_ALIGN_INSIDE | FL_ALIGN_WRAP | FL_ALIGN_LEFT);
    update_desc();

    // Section Controls
    int y = 220;
    Fl_Box* lblU = new Fl_Box(10, y, 290, 20, "Open Set U (Green Arc)");
    lblU->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE); lblU->labelfont(FL_BOLD);
    y+=25;
    
    Fl_Value_Slider* sldPos = new Fl_Value_Slider(10, y, 290, 25, "U Position");
    sldPos->type(FL_HOR_NICE_SLIDER); sldPos->bounds(0, 2*PI); sldPos->value(PI);
    sldPos->callback([](Fl_Widget* w, void*){ g_state.u_center = ((Fl_Value_Slider*)w)->value(); if(glWin) glWin->redraw(); });
    y+=45;
    
    Fl_Value_Slider* sldWidth = new Fl_Value_Slider(10, y, 290, 25, "U Size (Extend Section)");
    sldWidth->type(FL_HOR_NICE_SLIDER); sldWidth->bounds(0.1, 2*PI + 0.5); sldWidth->value(PI/2);
    sldWidth->callback([](Fl_Widget* w, void*){ g_state.u_width = ((Fl_Value_Slider*)w)->value(); if(glWin) glWin->redraw(); });
    y+=45;
    
    Fl_Value_Slider* sldVal = new Fl_Value_Slider(10, y, 290, 25, "Section Value s(x)");
    sldVal->type(FL_HOR_NICE_SLIDER); sldVal->bounds(-2, 2); sldVal->value(1.0);
    sldVal->callback([](Fl_Widget* w, void*){ g_state.sectionValue = ((Fl_Value_Slider*)w)->value(); if(glWin) glWin->redraw(); });
    y+=45;

    // Probe Control
    Fl_Value_Slider* sldProbe = new Fl_Value_Slider(10, y, 290, 25, "Probe Stalk (Yellow)");
    sldProbe->type(FL_HOR_NICE_SLIDER); sldProbe->bounds(0, 2*PI); sldProbe->value(0);
    sldProbe->callback([](Fl_Widget* w, void*){ g_state.probeAngle = ((Fl_Value_Slider*)w)->value(); if(glWin) glWin->redraw(); });
    
    // Checkboxes
    y+=50;
    Fl_Check_Button* bEtale = new Fl_Check_Button(10, y, 290, 25, "Show Etale Space (Ghost)");
    bEtale->value(1);
    bEtale->callback([](Fl_Widget* w, void*){ g_state.showEtale = ((Fl_Check_Button*)w)->value(); if(glWin) glWin->redraw(); });
    
    grp->end();

    glWin = new SheafView(320, 10, 950, 780);
    glWin->end();

    win->resizable(glWin);
    win->show(argc, argv);
    return Fl::run();
}

```

### How to use this Laboratory for Educational Value

1. **Understand "Restriction" and "Gluing"**:
* **Select "Constant Sheaf"**.
* Move the **U Position** slider. You see the red curve (the section) moving around the circle.
* Increase **U Size**. The red curve grows.
* **Experiment**: Can you make the red curve cover the entire circle? Yes. The red curve meets itself perfectly. This means a *Global Section* exists.


2. **Understand "Non-Triviality" (The Mobius Test)**:
* **Select "Mobius Sheaf"**.
* Look at the "Ghost" mesh. It looks like a twisted ribbon.
* Set **Section Value** to something non-zero (e.g., 1.0).
* Increase **U Size** slowly to cover the whole circle (towards 6.28).
* **Observe**: As the red line wraps around, it approaches its tail... but **on the wrong side**! The start is at , the end is at .
* **Conclusion**: You cannot glue the ends. The section is discontinuous. Thus,  (no global sections except the zero section).


3. **Understand "Skyscraper Sheaves" (Localization)**:
* **Select "Skyscraper Sheaf"**.
* The ghost mesh disappears almost everywhere.
* Move the **Probe Stalk** slider. The yellow line (the fiber) is a single point (zero) everywhere...
* ...Until you hit the "back" of the circle (around ). Suddenly, the yellow stalk explodes into a vertical line.
* This visualizes strictly that the data of the sheaf lives *only* at that point.



### Next Step

Would you like me to introduce **Cohomology Computation** into this code, adding a display that calculates the Cech complex values ( and ) in real-time based on your slider positions?
