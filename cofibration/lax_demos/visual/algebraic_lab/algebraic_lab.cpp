#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Button.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <vector>
#include <complex>
#include <cmath>
#include <iostream>
#include <functional>
#include <algorithm>

// -------------------------------------------------------------------------
// Constants & Math Helpers
// -------------------------------------------------------------------------
const float PI = 3.14159265359f;

struct Vec3 {
    float x, y, z;
    Vec3 operator+(const Vec3& v) const { return {x+v.x, y+v.y, z+v.z}; }
    Vec3 operator-(const Vec3& v) const { return {x-v.x, y-v.y, z-v.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
    
    Vec3 normalize() {
        float l = length();
        if(l > 0) return {x/l, y/l, z/l};
        return *this;
    }
    
    static Vec3 cross(const Vec3& a, const Vec3& b) {
        return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
    }
};

struct Color { float r, g, b, a; };

// HSV to RGB for complex phase coloring
Color complexToColor(float angle, float mag) {
    float h = angle / (2.0f * PI); 
    if(h < 0) h += 1.0f;
    float s = 1.0f;
    float v = 1.0f - 1.0f/(1.0f + mag); // Map magnitude to brightness saturation
    
    float c = v * s;
    float x = c * (1.0f - std::abs(std::fmod(h * 6.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    float r=0, g=0, b=0;
    if(h < 1/6.0f) { r=c; g=x; }
    else if(h < 2/6.0f) { r=x; g=c; }
    else if(h < 3/6.0f) { g=c; b=x; }
    else if(h < 4/6.0f) { g=x; b=c; }
    else if(h < 5/6.0f) { r=x; b=c; }
    else { r=c; b=x; }
    
    return {r+m, g+m, b+m, 1.0f};
}

// -------------------------------------------------------------------------
// Global Simulation State
// -------------------------------------------------------------------------
struct SimState {
    int mode = 0;           // 0: Riemann Surface, 1: Algebraic Surface, 2: Vector Field
    float paramA = 1.0f;    // Coefficient A
    float paramB = 0.0f;    // Coefficient B
    float paramC = 0.0f;    // Coefficient C
    float resolution = 60.0f;
    bool wireframe = false;
    bool rotate = true;
};

SimState g_state; // Global state accessible by UI and Renderer

// -------------------------------------------------------------------------
// OpenGL Renderer
// -------------------------------------------------------------------------
class AlgebraicGLWindow : public Fl_Gl_Window {
    float time;
    float rotX, rotY;

public:
    AlgebraicGLWindow(int x, int y, int w, int h) 
        : Fl_Gl_Window(x, y, w, h), time(0), rotX(20), rotY(45) {
        mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_MULTISAMPLE);
    }

    void update() {
        if(g_state.rotate) time += 0.01f;
        redraw();
    }

    void handle_input(int event) {
        // Simple mouse interaction could go here
    }

private:
    void draw() override {
        if (!valid()) {
            glViewport(0, 0, w(), h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_LIGHTING);
            glEnable(GL_LIGHT0);
            glEnable(GL_COLOR_MATERIAL);
            glEnable(GL_NORMALIZE);
            
            GLfloat light_pos[] = { 5.0, 10.0, 10.0, 1.0 };
            glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
            GLfloat ambient[] = { 0.2f, 0.2f, 0.2f, 1.0f };
            glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
            
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            gluPerspective(45.0, (float)w()/(float)h(), 0.1, 100.0);
            glMatrixMode(GL_MODELVIEW);
        }

        glClearColor(0.15f, 0.15f, 0.18f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glLoadIdentity();
        gluLookAt(0, 2, 8, 0, 0, 0, 0, 1, 0);

        glRotatef(rotX + time * 5.0f * (g_state.rotate ? 1 : 0), 1, 0, 0);
        glRotatef(rotY + time * 10.0f * (g_state.rotate ? 1 : 0), 0, 1, 0);

        if(g_state.wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        else glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        // Dispatch based on mode
        switch(g_state.mode) {
            case 0: drawRiemannSurface(); break;
            case 1: drawAlgebraicSurface(); break;
            case 2: drawVectorField(); break;
        }
    }

    // ----------------------------------------------------------------
    // Mode 0: Riemann Surface Visualization
    // Visualizes f(z) = z^k + A*z + B
    // Height = |f(z)|, Color = arg(f(z))
    // ----------------------------------------------------------------
    void drawRiemannSurface() {
        int res = (int)g_state.resolution;
        float limit = 3.0f;
        
        glBegin(GL_QUADS);
        for(int i = 0; i < res; ++i) {
            for(int j = 0; j < res; ++j) {
                float u = (float)i/res * 2*limit - limit;
                float v = (float)j/res * 2*limit - limit;
                float step = (2*limit)/res;

                auto func = [&](std::complex<float> z) {
                    // f(z) = z^A + B*z + C
                    // Since z^Float is complex, we approximate integer powers or use std::pow
                    return std::pow(z, std::complex<float>(std::floor(g_state.paramA), 0)) 
                           + g_state.paramB * z 
                           + std::complex<float>(g_state.paramC, 0.0f);
                };

                auto evalVertex = [&](float x, float y) {
                    std::complex<float> z(x, y);
                    std::complex<float> val = func(z);
                    float mag = std::abs(val);
                    float arg = std::arg(val);
                    
                    // Clamp height for visualization
                    float h = std::log(mag + 1.0f) * 1.5f; 
                    Color c = complexToColor(arg, mag);
                    glColor3f(c.r, c.g, c.b);
                    glVertex3f(x, h - 2.0f, y);
                };

                evalVertex(u, v);
                evalVertex(u+step, v);
                evalVertex(u+step, v+step);
                evalVertex(u, v+step);
            }
        }
        glEnd();
    }

    // ----------------------------------------------------------------
    // Mode 1: Implicit Real Algebraic Surface
    // Steiner Surface / Roman Surface variants: x^2*y^2 + y^2*z^2 + z^2*x^2 + xyz = 0
    // Or Superquadrics
    // ----------------------------------------------------------------
    void drawAlgebraicSurface() {
        int res = (int)g_state.resolution;
        float r = 2.5f;
        
        // Parametric visualization of a specialized surface (e.g. Klein Bottle or Dini)
        // Dini's Surface:
        // x = a cos(u) sin(v)
        // y = a sin(u) sin(v)
        // z = a (cos(v) + log(tan(v/2))) + b*u
        
        float a = g_state.paramA * 0.5f + 0.1f;
        float b = g_state.paramB * 0.5f;

        glBegin(GL_QUADS);
        for(int i = 0; i < res; ++i) {
            for(int j = 1; j < res-1; ++j) { // avoid poles for Dini
                float u1 = (float)i/res * 4 * PI;
                float v1 = (float)j/res * 2.0f + 0.1f; // Avoid 0
                float u2 = (float)(i+1)/res * 4 * PI;
                float v2 = (float)(j+1)/res * 2.0f + 0.1f;

                auto dini = [&](float u, float v) {
                    float x = a * cos(u) * sin(v);
                    float y = a * sin(u) * sin(v);
                    float z = a * (cos(v) + log(tan(v/2.0f))) + b * u;
                    
                    // Simple color based on curvature/position
                    glColor3f(0.5f + 0.5f*cos(u), 0.5f + 0.5f*sin(v), 0.7f);
                    glNormal3f(cos(u), sin(u), 0); // Approx normal
                    glVertex3f(x, y - b*2*PI, z); // Center it roughly
                };

                dini(u1, v1);
                dini(u2, v1);
                dini(u2, v2);
                dini(u1, v2);
            }
        }
        glEnd();
    }

    // ----------------------------------------------------------------
    // Mode 2: Vector Fields on Tangent Bundle (Hairy Ball Theorem)
    // Base: Sphere. Vectors: Tangent vectors.
    // ----------------------------------------------------------------
    void drawVectorField() {
        int res = 20; // Lower res for clarity
        float R = 2.0f;
        
        // Draw Base Sphere
        glColor4f(0.3f, 0.3f, 0.3f, 0.8f);
        GLUquadric* quad = gluNewQuadric();
        gluSphere(quad, R-0.05f, res, res);
        gluDeleteQuadric(quad);

        glBegin(GL_LINES);
        for(int i = 0; i < res; ++i) {
            for(int j = 0; j < res; ++j) {
                float theta = (float)i/res * 2 * PI;
                float phi = (float)j/res * PI;

                float x = R * sin(phi) * cos(theta);
                float y = R * cos(phi);
                float z = R * sin(phi) * sin(theta);
                
                // Vector Field Definition V(p)
                // Let's create a field that rotates around Y axis + some disturbance
                // Tangent vector T = (-z, 0, x) is rotation around Y.
                // Parameter A adjusts vertical flow.
                
                Vec3 p = {x, y, z};
                
                // Rotational component
                Vec3 rot = {-z, 0, x};
                
                // Pole-to-pole flow (Param A)
                Vec3 flow = {0, -1, 0}; 
                
                // Combine
                Vec3 v = rot + flow * g_state.paramA;
                
                // Project v onto tangent plane at p (Gram-Schmidt)
                // v_tangent = v - (v . n) * n, where n is normal (p normalized)
                Vec3 n = p.normalize();
                float dot = v.x*n.x + v.y*n.y + v.z*n.z;
                Vec3 tan = v - n*dot;
                tan = tan.normalize() * (0.5f + g_state.paramB);

                glColor3f(1.0f, 0.8f, 0.2f);
                glVertex3f(x, y, z);
                glVertex3f(x + tan.x, y + tan.y, z + tan.z);
            }
        }
        glEnd();
    }
};

// -------------------------------------------------------------------------
// User Interface Building
// -------------------------------------------------------------------------

AlgebraicGLWindow* g_glWindow = nullptr;

void update_cb(Fl_Widget* w, void*) {
    if(!g_glWindow) return;
    g_glWindow->redraw();
}

void mode_cb(Fl_Widget* w, void*) {
    Fl_Choice* c = (Fl_Choice*)w;
    g_state.mode = c->value();
    g_glWindow->redraw();
}

void slider_a_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.paramA = s->value();
    g_glWindow->redraw();
}

void slider_b_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.paramB = s->value();
    g_glWindow->redraw();
}

void slider_c_cb(Fl_Widget* w, void*) {
    Fl_Value_Slider* s = (Fl_Value_Slider*)w;
    g_state.paramC = s->value();
    g_glWindow->redraw();
}

void check_rot_cb(Fl_Widget* w, void*) {
    Fl_Check_Button* b = (Fl_Check_Button*)w;
    g_state.rotate = b->value();
}

void timer_cb(void*) {
    if(g_glWindow) g_glWindow->update();
    Fl::repeat_timeout(1.0/60.0, timer_cb);
}

int main(int argc, char **argv) {
    // Main Window Layout
    Fl_Double_Window* win = new Fl_Double_Window(1000, 600, "Algebraic Geometry Lab");
    
    // OpenGL Viewport
    g_glWindow = new AlgebraicGLWindow(10, 10, 700, 580);
    
    // Control Panel
    Fl_Group* controls = new Fl_Group(720, 10, 270, 580, "Parameters");
    controls->box(FL_DOWN_BOX);
    
    // 1. Mode Selection
    Fl_Choice* modeChoice = new Fl_Choice(740, 40, 230, 25, "Visualization Mode");
    modeChoice->add("Riemann Surface (Complex Analysis)");
    modeChoice->add("Dini Surface (Differential Geo)");
    modeChoice->add("Vector Fields (Topology)");
    modeChoice->value(0);
    modeChoice->callback(mode_cb);
    
    // 2. Sliders
    int y = 100;
    Fl_Value_Slider* sliderA = new Fl_Value_Slider(740, y, 230, 20, "Parameter A (Power / Radius)");
    sliderA->type(FL_HOR_NICE_SLIDER);
    sliderA->bounds(1.0, 6.0);
    sliderA->value(g_state.paramA);
    sliderA->callback(slider_a_cb);
    
    y += 50;
    Fl_Value_Slider* sliderB = new Fl_Value_Slider(740, y, 230, 20, "Parameter B (Linear Coeff / Twist)");
    sliderB->type(FL_HOR_NICE_SLIDER);
    sliderB->bounds(-2.0, 2.0);
    sliderB->value(g_state.paramB);
    sliderB->callback(slider_b_cb);
    
    y += 50;
    Fl_Value_Slider* sliderC = new Fl_Value_Slider(740, y, 230, 20, "Parameter C (Constant)");
    sliderC->type(FL_HOR_NICE_SLIDER);
    sliderC->bounds(-5.0, 5.0);
    sliderC->value(g_state.paramC);
    sliderC->callback(slider_c_cb);
    
    // 3. Toggles
    y += 50;
    Fl_Check_Button* rotCheck = new Fl_Check_Button(740, y, 100, 25, "Auto Rotate");
    rotCheck->value(1);
    rotCheck->callback(check_rot_cb);
    
    controls->end();
    
    win->end();
    win->resizable(g_glWindow);
    win->show(argc, argv);
    
    // Start loop
    Fl::add_timeout(1.0/60.0, timer_cb);
    return Fl::run();
}
