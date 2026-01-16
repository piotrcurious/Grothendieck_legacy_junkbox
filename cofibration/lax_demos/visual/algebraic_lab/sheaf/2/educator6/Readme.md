This update transforms the laboratory into a generic **Sheaf Explorer** capable of visualizing highly non-linear and fractal "Etale Spaces."

We introduce three new concepts to the Grothendieckian toolkit:

1. **The Bifurcation Sheaf (Chaos)**: A "Constructible Sheaf" where the fiber (the attractor) changes topology abruptly as you move along the base. This visualizes how sheaves handle discontinuity and emergence.
2. **The Riemann Sheaf ()**: The classic "Covering Space" showing non-linear branching.
3. **The Weierstrass Sheaf (Fractal Sections)**: Visualizes a bundle where the sections are continuous but nowhere differentiable, demonstrating that "geometric objects" in sheaf theory need not be smooth.

### Compile Instructions

Save as `fractal_sheaf.cpp`.

```bash
g++ -o fractal_sheaf fractal_sheaf.cpp -lfltk -lfltk_gl -lGL -lGLU

```

### The Code

```cpp
// fractal_sheaf.cpp
//
// ADVANCED SHEAF LABORATORY: Non-Linear & Fractal Etale Spaces
//
// Concepts Visualized:
// 1. Constructible Sheaves: Fibers change topology (Bifurcation).
// 2. Ramification: Branch points in complex functions (Sqrt).
// 3. Rough Sections: Continuous but nowhere differentiable (Weierstrass).
// 4. Monodromy in Chaotic Systems.

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
#include <complex>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Strategy Pattern for Etale Spaces
// -----------------------------------------------------------------------------
enum DrawStyle { STYLE_SURFACE, STYLE_POINTS, STYLE_WIREFRAME };

class SheafStrategy {
public:
    virtual ~SheafStrategy() {}
    
    // Calculate the fiber geometry
    // theta: Base space coordinate (usually 0 to 2PI)
    // val: Local vertical parameter
    // out: 3D coordinates
    virtual void getPoint(float theta, float val, float out[3]) = 0;
    
    // For fractals (Bifurcation), we might generate a set of points for a single base value
    virtual bool isScatter() { return false; }
    virtual void getScatterPoints(float theta, std::vector<float>& points) {}

    virtual float getMinVal() { return -2.0f; }
    virtual float getMaxVal() { return 2.0f; }
    
    // Does the base space wrap around (Circle) or is it an interval (Line)?
    virtual bool isCyclic() { return true; } 
    
    // Defines the "Section" s(x). 
    // Default is just a constant height, but complex sheaves have complex sections.
    virtual float getSectionHeight(float theta, float inputVal) { return inputVal; }

    virtual const char* getName() = 0;
    virtual const char* getDescription() = 0;
};

// -----------------------------------------------------------------------------
// 1. The Riemann Sheaf (Square Root Surface)
// Base: Complex plane unit circle (angle theta)
// Fiber: Two points (the two roots)
// Monodromy: 4PI periodic.
// -----------------------------------------------------------------------------
class RiemannSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3]) override {
        // We visualize the Real part of sqrt(z) as height, 
        // or just the classic Riemann surface helicoid.
        // Let's do the helicoid: w = 2*theta (covering)
        float r = 3.0f;
        // Map val (-1 to 1) to width of the ribbon
        float r_eff = r + val * 0.5f; 
        
        // We map 0..4PI to the geometry to show the double cover
        // But the input 'theta' is 0..2PI.
        // We will handle the "Sheet" logic in the visualizer loop usually,
        // but here let's make the "Etale Space" the full 4PI surface compressed?
        // Simpler: Just a standard Mobius-like twist but 4pi periodic?
        
        // Let's visualize the "Real Part of Sqrt(z)"
        // z = r e^{i theta} -> sqrt(z) = sqrt(r) e^{i theta/2}
        // Height = Re(sqrt) = sqrt(r) * cos(theta/2)
        
        // We will just draw a generic helicoid representing the surface
        float angle = theta; // This will range 0..4PI in the drawer if we allow it
        
        out[0] = r_eff * cos(angle);
        out[1] = angle * 0.5f - 3.0f; // Height increases with angle (Helicoid)
        out[2] = r_eff * sin(angle);
    }
    
    // Override to allow 4PI range drawing in main loop
    float getMaxVal() override { return 1.0f; } // Width
    
    const char* getName() override { return "Riemann Sheaf (Sqrt z)"; }
    const char* getDescription() override { 
        return "The Double Cover.\n"
               "f(z) = z^(1/2).\n"
               "Going around once (2pi) switches sheets.\n"
               "Requires 4pi to close the loop."; 
    }
};

// -----------------------------------------------------------------------------
// 2. The Bifurcation Sheaf (Logistic Map)
// Base: Parameter r in [2.9, 4.0] mapped to Angle.
// Fiber: The Attractor Set (limit points).
// -----------------------------------------------------------------------------
class BifurcationSheaf : public SheafStrategy {
public:
    bool isScatter() override { return true; }
    bool isCyclic() override { return false; } // It's an interval, not a circle

    void getPoint(float theta, float val, float out[3]) override {
        // Not used in scatter mode
    }

    // Generate the attractor points for parameter r(theta)
    void getScatterPoints(float theta, std::vector<float>& points) override {
        // Map theta (0..2PI) to r (2.9 .. 4.0)
        // We use an arc of the circle for visualization
        float t = theta / (2.0f*PI);
        float r = 2.9f + t * (4.0f - 2.9f);
        
        // Logistic Map: x_n+1 = r * x_n * (1 - x_n)
        float x = 0.5f;
        // Burn-in
        for(int i=0; i<100; ++i) x = r * x * (1.0f - x);
        
        // Collect points
        for(int i=0; i<50; ++i) {
            x = r * x * (1.0f - x);
            points.push_back(x); // The "Height" value (0..1)
        }
    }
    
    // Map base coords (theta) to 3D world
    void mapBaseToWorld(float theta, float height, float out[3]) {
        float rad = 4.0f;
        // Use a 3/4 circle arc to layout the parameter space
        float arc = theta * 0.75f + 0.4f; 
        out[0] = rad * cos(arc);
        out[1] = height * 4.0f - 2.0f; // Scale height (0..1 -> -2..2)
        out[2] = rad * sin(arc);
    }

    const char* getName() override { return "Bifurcation Sheaf (Chaos)"; }
    const char* getDescription() override { 
        return "Constructible Sheaf.\n"
               "Base: Parameter r.\n"
               "Stalk: Attractor of x -> rx(1-x).\n"
               "Visualizes splitting fibers and chaos."; 
    }
};

// -----------------------------------------------------------------------------
// 3. The Weierstrass Sheaf (Fractal Section)
// Base: Circle S1.
// Fiber: Real line R.
// Section: A fractal function.
// -----------------------------------------------------------------------------
class WeierstrassSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3]) override {
        // Standard Cylinder topology
        float r = 3.0f;
        out[0] = r * cos(theta);
        out[1] = val; 
        out[2] = r * sin(theta);
    }
    
    // The Section is the fractal part
    float getSectionHeight(float theta, float inputVal) override {
        // Weierstrass function approximation:
        // sum a^n cos(b^n pi x)
        // a = 0.5, b = 3
        float sum = 0.0f;
        float a = 0.5f;
        float b = 3.0f;
        
        // Normalize theta to 0..1 for the function
        float x = theta / (2.0f*PI); 
        
        for(int n=0; n<6; ++n) {
            sum += pow(a, n) * cos(pow(b, n) * PI * x * 2.0f); // *2 to wrap twice?
        }
        // InputVal shifts the whole fractal up/down
        return sum + inputVal - 1.0f; // Center it
    }

    const char* getName() override { return "Weierstrass Sheaf (Fractal)"; }
    const char* getDescription() override { 
        return "Rough Section.\n"
               "Topology is trivial (Cylinder).\n"
               "The Section s(x) is continuous\n"
               "but nowhere differentiable."; 
    }
};

// -----------------------------------------------------------------------------
// App State
// -----------------------------------------------------------------------------
struct AppState {
    float rotX = 20.0f, rotY = -30.0f;
    SheafStrategy* currentSheaf = nullptr;
    
    float u_center = PI; 
    float u_width = PI/2.0f;
    float sectionParam = 0.0f; // Used as 'val' or shift
    
    bool showEtale = true;
};
AppState g_state;

// -----------------------------------------------------------------------------
// OpenGL View
// -----------------------------------------------------------------------------
class FractalSheafView : public Fl_Gl_Window {
    int lastX, lastY;
public:
    FractalSheafView(int x, int y, int w, int h) : Fl_Gl_Window(x,y,w,h) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA | FL_MULTISAMPLE);
    }

    void draw() override {
        if(!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.08f, 0.08f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45.0, (float)w()/h(), 0.1, 100.0);
        
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 4, 14, 0, 0, 0, 0, 1, 0);
        glRotatef(g_state.rotX, 1, 0, 0);
        glRotatef(g_state.rotY, 0, 1, 0);

        if(g_state.currentSheaf) {
            drawEtaleSpace();
            drawSection();
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
    void drawEtaleSpace() {
        if (!g_state.showEtale) return;

        SheafStrategy* s = g_state.currentSheaf;
        
        // 1. Scatter Mode (Bifurcation)
        if (s->isScatter()) {
            glPointSize(2.0f);
            glBegin(GL_POINTS);
            int steps = 200;
            for(int i=0; i<steps; ++i) {
                float theta = 2.0f*PI * (float)i/steps;
                std::vector<float> pts;
                s->getScatterPoints(theta, pts);
                
                // Color gradient based on theta (Parameter r)
                glColor4f(0.2f + 0.8f*(float)i/steps, 0.5f, 1.0f - 0.8f*(float)i/steps, 0.6f);
                
                for(float h : pts) {
                    float p[3];
                    ((BifurcationSheaf*)s)->mapBaseToWorld(theta, h, p);
                    glVertex3fv(p);
                }
            }
            glEnd();
            return;
        }

        // 2. Surface Mode (Riemann, Weierstrass)
        glColor4f(1.0f, 1.0f, 1.0f, 0.15f);
        
        int segments = 100;
        float maxAngle = s->getName()[0] == 'R' ? 4.0f*PI : 2.0f*PI; // Hack for Riemann 4PI
        
        float minV = s->getMinVal();
        float maxV = s->getMaxVal();
        
        for(int i=0; i<segments; ++i) {
            float t1 = maxAngle * i / segments;
            float t2 = maxAngle * (i+1) / segments;
            
            glBegin(GL_QUAD_STRIP);
            float p[3];
            s->getPoint(t1, minV, p); glVertex3fv(p);
            s->getPoint(t1, maxV, p); glVertex3fv(p);
            s->getPoint(t2, minV, p); glVertex3fv(p);
            s->getPoint(t2, maxV, p); glVertex3fv(p);
            glEnd();
        }
        
        // Wireframe edges for clarity
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=segments; ++i) {
             float t = maxAngle * i / segments;
             float p[3]; s->getPoint(t, minV, p); glVertex3fv(p);
        }
        glEnd();
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=segments; ++i) {
             float t = maxAngle * i / segments;
             float p[3]; s->getPoint(t, maxV, p); glVertex3fv(p);
        }
        glEnd();
    }

    void drawSection() {
        SheafStrategy* s = g_state.currentSheaf;
        if(s->isScatter()) return; // No continuous section in chaos usually (unless periodic window)

        float start = g_state.u_center - g_state.u_width/2.0f;
        float end = g_state.u_center + g_state.u_width/2.0f;
        int steps = 100;
        
        glLineWidth(3.0f);
        glColor3f(1.0f, 0.3f, 0.3f); // Red Section
        
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=steps; ++i) {
            float t = start + (end-start)*i/steps;
            
            // Calculate height. If it's Weierstrass, this returns the fractal height.
            // If it's Riemann, we just use the parameter as is? 
            // For Riemann, let's just draw a line at constant 'val' offset on the surface
            
            float h = s->getSectionHeight(t, g_state.sectionParam);
            float p[3];
            
            // For Riemann, we need to handle the 4PI logic if we want a "long" section
            // But getPoint handles the mapping from t->World.
            // If t > 2PI, we are on second sheet.
            
            s->getPoint(t, h, p); 
            
            // Override for Weierstrass to actually modify Y
            if (std::string(s->getName()).find("Weierstrass") != std::string::npos) {
                p[1] = h; // Explicit height override
            }
            
            glVertex3fv(p);
        }
        glEnd();
        
        // Draw germs
        glPointSize(6.0f);
        glBegin(GL_POINTS);
        float p1[3], p2[3];
        float h1 = s->getSectionHeight(start, g_state.sectionParam);
        float h2 = s->getSectionHeight(end, g_state.sectionParam);
        
        s->getPoint(start, h1, p1); 
        s->getPoint(end, h2, p2);
        
        if (std::string(s->getName()).find("Weierstrass") != std::string::npos) {
             p1[1] = h1; p2[1] = h2;
        }

        glVertex3fv(p1); glVertex3fv(p2);
        glEnd();
    }
};

// -----------------------------------------------------------------------------
// Main UI
// -----------------------------------------------------------------------------
FractalSheafView* glWin = nullptr;
Fl_Box* descBox = nullptr;

RiemannSheaf sRiemann;
BifurcationSheaf sBifurc;
WeierstrassSheaf sWeier;

void update_cb(Fl_Widget* w, void*) {
    if(glWin) glWin->redraw();
}

int main(int argc, char** argv) {
    Fl_Double_Window* win = new Fl_Double_Window(1280, 800, "Non-Linear Sheaf Lab");
    
    Fl_Group* grp = new Fl_Group(10, 10, 300, 780);
    
    Fl_Box* title = new Fl_Box(10, 20, 290, 40, "FRACTAL SHEAVES");
    title->labelfont(FL_BOLD); title->labelsize(20);
    
    Fl_Choice* ch = new Fl_Choice(10, 80, 290, 30, "Etale Space Model");
    ch->add("Riemann Surface (Sqrt z)");
    ch->add("Bifurcation (Logistic Chaos)");
    ch->add("Weierstrass (Fractal Section)");
    ch->value(0);
    g_state.currentSheaf = &sRiemann;
    
    ch->callback([](Fl_Widget* w, void*){
        int v = ((Fl_Choice*)w)->value();
        if(v==0) g_state.currentSheaf = &sRiemann;
        else if(v==1) g_state.currentSheaf = &sBifurc;
        else if(v==2) g_state.currentSheaf = &sWeier;
        
        if(descBox) descBox->label(g_state.currentSheaf->getDescription());
        update_cb(nullptr, nullptr);
    });

    descBox = new Fl_Box(10, 120, 290, 80, sRiemann.getDescription());
    descBox->box(FL_BORDER_BOX);
    descBox->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);

    // Controls
    int y = 220;
    Fl_Box* lbl = new Fl_Box(10, y, 290, 20, "Section / Parameter Controls");
    lbl->labelfont(FL_BOLD); lbl->align(FL_ALIGN_LEFT|FL_ALIGN_INSIDE);
    y+=30;

    Fl_Value_Slider* sldP = new Fl_Value_Slider(10, y, 290, 25, "U Position (Base)");
    sldP->bounds(0, 4*PI); sldP->value(PI); // Allow 4PI for Riemann
    sldP->callback([](Fl_Widget* w, void*){ g_state.u_center = ((Fl_Value_Slider*)w)->value(); update_cb(nullptr,nullptr); });
    y+=40;

    Fl_Value_Slider* sldW = new Fl_Value_Slider(10, y, 290, 25, "U Width (Domain Size)");
    sldW->bounds(0.1, 2*PI); sldW->value(PI/2);
    sldW->callback([](Fl_Widget* w, void*){ g_state.u_width = ((Fl_Value_Slider*)w)->value(); update_cb(nullptr,nullptr); });
    y+=40;

    Fl_Value_Slider* sldH = new Fl_Value_Slider(10, y, 290, 25, "Value / Offset");
    sldH->bounds(-1.0, 1.0); sldH->value(0);
    sldH->callback([](Fl_Widget* w, void*){ g_state.sectionParam = ((Fl_Value_Slider*)w)->value(); update_cb(nullptr,nullptr); });
    y+=40;

    Fl_Check_Button* bE = new Fl_Check_Button(10, y, 290, 25, "Show Etale Ghost");
    bE->value(1);
    bE->callback([](Fl_Widget* w, void*){ g_state.showEtale = ((Fl_Check_Button*)w)->value(); update_cb(nullptr,nullptr); });

    grp->end();

    glWin = new FractalSheafView(320, 10, 950, 780);
    glWin->end();

    win->resizable(glWin);
    win->show(argc, argv);
    return Fl::run();
}

```

### Guide to the Non-Linear Grothendieck Lab

This tool visualizes how the abstract definition of a sheaf adapts to "messy" functions.

#### 1. The Bifurcation Sheaf (Chaos)

* **What it is**: The "Etale Space" here is the collection of attractor points for the Logistic Map .
* **Grothendieckian Concept**: **Constructibility**.
* Observe that for low "U Position" (small ), the fiber is a single point (1-sheeted).
* As you slide right, it splits into 2, then 4.
* **Insight**: A sheaf does not need to have the same number of "stalks" everywhere. It tracks the *change* in the solution set. The sheaf "knows" where the bifurcations happen.



#### 2. The Riemann Sheaf (Square Root)

* **What it is**: The surface .
* **Grothendieckian Concept**: **Ramification & Monodromy**.
* Use the "U Position" slider to move the Red Section.
* Note that the slider allows you to go from  to .
* If you trace the section for one full circle (), you do **not** return to the start. You are on the "other sheet" (the negative root).
* This visualizes why we define sheaves on a "Covering Space" rather than the base plane itself—to make the functions single-valued and continuous.



#### 3. The Weierstrass Sheaf (Fractal)

* **What it is**: A trivial cylinder bundle, but with a **Fractal Section**.
* **Grothendieckian Concept**: **Algebraic Generality**.
* Sheaf theory does not require sections to be smooth (). It only requires them to be continuous ().
* The Red line is the Weierstrass function. It is "spiky" everywhere.
* Despite being infinitely rough, it is a valid section of the sheaf of continuous functions .
* This highlights the power of the toolkit: it applies equally to differential geometry (smooth) and fractal geometry (rough).
