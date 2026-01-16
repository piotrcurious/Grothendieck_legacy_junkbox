#include <FL/Fl.H>
#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Gl_Window.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Check_Button.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/gl.h>
#include <GL/glu.h>

#include <cmath>
#include <vector>
#include <memory>

static constexpr float PI = 3.14159265359f;

// -----------------------------------------------------------------------------
// Core Symplectic & Sheaf State
// -----------------------------------------------------------------------------
struct StalkParams {
    float alpha = 1.0f; 
    float beta  = 0.5f; 
    float morph = 0.0f; 
};

class SheafStrategy;

struct AppState {
    float rotX = 25.0f, rotY = -45.0f;
    float u_pos = 0.0f, u_width = PI, section_val = 0.0f;
    StalkParams params;
    SheafStrategy* current = nullptr; 
    bool showEtale = true;
    bool showDifferentials = true; 
    bool animate = false;
    float anim_t = 0.0f;
};

// -----------------------------------------------------------------------------
// High-Dimensional Optical Sheaf Strategies
// -----------------------------------------------------------------------------
class SheafStrategy {
public:
    virtual ~SheafStrategy() {}
    virtual void getPoint(float u, float v, float out[3], const StalkParams& p) = 0;
    
    virtual void getTangent(float u, float v, float out[3], const StalkParams& p) {
        float p1[3], p2[3], du = 0.005f;
        getPoint(u - du, v, p1, p);
        getPoint(u + du, v, p2, p);
        for(int i=0; i<3; i++) out[i] = (p2[i] - p1[i]) / (2.0f * du);
    }

    virtual float getMinV() { return -2.0f; }
    virtual float getMaxV() { return 2.0f; }
    virtual const char* getName() = 0;
    virtual const char* getDesc() = 0;
    virtual const char* getParamALabel() { return "Param A"; }
    virtual const char* getParamBLabel() { return "Param B"; }
};

// 1. Gouy-Phase Symplectic Sheaf
// Models the rotation of the phase-space Lagrangian manifold through a focal point.
class GouySheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // Base coordinate u: z-axis (propagation)
        // Fiber coordinate v: x-axis (transverse)
        float z = u;
        float x = v * p.alpha;
        
        // Rayleigh range zR
        float zR = p.beta + 0.1f;
        // The Gouy Phase: arctan(z / zR)
        float psi = atan2f(z, zR);
        
        // The phase-space "sheet" evolves as a rotation in (x, p)
        // We visualize the wavefront S(x,z) = x^2 / 2R(z) - psi(z)
        float Rz = z + (zR*zR)/(z+1e-6f);
        float phase = (x*x) / (2.0f * (Rz + 1e-6f)) - psi + p.morph;
        
        out[0] = x * 4.0f;
        out[1] = phase * 2.0f;
        out[2] = z * 4.0f;
    }
    const char* getName() override { return "Gouy Phase (Symplectic)"; }
    const char* getDesc() override { return "Models the phase-space rotation\nof a Gaussian beam. The Gouy phase\ncauses a π/2 'jump' in the section."; }
    const char* getParamALabel() override { return "Beam Waist"; }
    const char* getParamBLabel() override { return "Rayleigh Range"; }
};

// 2. Poincaré Spinor Sheaf
// Visualizes the Pancharatnam-Berry phase as a sheaf over the sphere of directions.
class PoincareSheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // u, v are spherical coordinates
        float theta = u; // Base: latitude
        float phi = v;   // Fiber: longitude/phase
        
        float r = 5.0f;
        // We model a spin-1/2 state |psi> = [cos(theta/2), e^(i*phi)sin(theta/2)]
        // The geometric phase is the area enclosed on the sphere.
        float geo_phase = p.alpha * (1.0f - cosf(theta)) * phi;
        float h = p.beta * sinf(geo_phase + p.morph);
        
        out[0] = (r + h) * sinf(theta) * cosf(phi);
        out[1] = (r + h) * cosf(theta);
        out[2] = (r + h) * sinf(theta) * sinf(phi);
    }
    const char* getName() override { return "Poincaré Spinor Sheaf"; }
    const char* getDesc() override { return "A sheaf of Jones Vectors over S2.\nGeometric phase accumulates as\na 'twist' in the local section."; }
    const char* getParamALabel() override { return "Curvature Scale"; }
    const char* getParamBLabel() override { return "Phase Amplitude"; }
    float getMinV() override { return 0; }
    float getMaxV() override { return 2.0f * PI; }
};

// 3. Abbe Sine Sheaf (Isoplanatism)
// Represents the mapping from spatial coordinates to angular frequencies.
class AbbeSheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // u: Spatial coordinate x
        // v: Pupil coordinate sin(alpha)
        float x = u;
        float sin_alpha = v * p.alpha;
        
        // The Sine Condition: y = M * x
        // We visualize the mapping error (Abberation sheaf)
        float m = 1.0f; // Magnification
        float error = (sin_alpha - m * sinf(x)) * p.beta;
        float phase = cosf(x * p.alpha + p.morph) * error;
        
        out[0] = x * 3.0f;
        out[1] = phase * 5.0f;
        out[2] = v * 5.0f;
    }
    const char* getName() override { return "Abbe Isoplanatic Sheaf"; }
    const char* getDesc() override { return "Models the invariant mapping\nbetween spatial and frequency stalks.\nBreaks down for non-isoplanatic fields."; }
    const char* getParamALabel() override { return "Numerical Aperture"; }
    const char* getParamBLabel() override { return "Mapping Error"; }
};

// -----------------------------------------------------------------------------
// UI & Visualization logic
// -----------------------------------------------------------------------------
struct UIContext {
    AppState* state;
    Fl_Box* desc_box;
    Fl_Value_Slider *s_alpha, *s_beta;
    std::vector<std::unique_ptr<SheafStrategy>> strategies;

    UIContext(AppState* s) : state(s) {
        strategies.push_back(std::make_unique<GouySheaf>());
        strategies.push_back(std::make_unique<PoincareSheaf>());
        strategies.push_back(std::make_unique<AbbeSheaf>());
        state->current = strategies[0].get();
    }
};

class SheafView : public Fl_Gl_Window {
    AppState* state;
public:
    SheafView(int x, int y, int w, int h, AppState* s) : Fl_Gl_Window(x,y,w,h), state(s) {}

    void draw() override {
        if (!valid()) {
            glViewport(0,0,w(),h());
            glEnable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(45, (float)w()/h(), 0.1, 200);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(20, 20, 30, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if (!state->current) return;

        if (state->showEtale) {
            int u_steps = 60, v_steps = 40;
            float u_min = -PI, u_max = PI;
            float v_min = state->current->getMinV(), v_max = state->current->getMaxV();

            glBegin(GL_QUADS);
            for(int i=0; i<u_steps; i++) {
                float u1 = u_min + (u_max-u_min)*i/u_steps;
                float u2 = u_min + (u_max-u_min)*(i+1)/u_steps;
                for(int j=0; j<v_steps; j++) {
                    float v1 = v_min + (v_max-v_min)*j/v_steps;
                    float v2 = v_min + (v_max-v_min)*(j+1)/v_steps;
                    float p1[3], p2[3], p3[3], p4[3];
                    state->current->getPoint(u1, v1, p1, state->params);
                    state->current->getPoint(u2, v1, p2, state->params);
                    state->current->getPoint(u2, v2, p3, state->params);
                    state->current->getPoint(u1, v2, p4, state->params);
                    
                    float lum = 0.2f + 0.1f * sinf(u1*2.0f);
                    glColor4f(0.2f, 0.5f, 1.0f, lum);
                    glVertex3fv(p1); glVertex3fv(p2); glVertex3fv(p3); glVertex3fv(p4);
                }
            }
            glEnd();
        }

        // Section visualization
        float u_start = state->u_pos - state->u_width/2.0f;
        float u_end = state->u_pos + state->u_width/2.0f;
        
        if (state->showDifferentials) {
            glColor4f(0.3f, 1.0f, 0.7f, 0.7f);
            glBegin(GL_LINES);
            for(int i=0; i<=40; i++) {
                float u = u_start + (u_end-u_start)*i/40.0f;
                float p[3], tan[3];
                state->current->getPoint(u, state->section_val, p, state->params);
                state->current->getTangent(u, state->section_val, tan, state->params);
                glVertex3fv(p);
                glVertex3f(p[0]+tan[0]*0.5f, p[1]+tan[1]*0.5f, p[2]+tan[2]*0.5f);
            }
            glEnd();
        }

        glColor3f(1.0f, 0.4f, 0.0f); glLineWidth(3.0f);
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=200; i++) {
            float u = u_start + (u_end-u_start)*i/200.0f;
            float p[3]; state->current->getPoint(u, state->section_val, p, state->params);
            glVertex3fv(p);
        }
        glEnd(); glLineWidth(1.0f);
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            state->rotY += (float)(Fl::event_x()-lx) * 0.5f;
            state->rotX += (float)(Fl::event_y()-ly) * 0.5f;
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1200, 800, "Sheaf Dynamics in Optical Phase Space");
    static UIContext ctx(&g_state);

    int sw = 320;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 800);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(12, 14, 20));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "SYMPLECTIC SHEAF");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(22);

    Fl_Choice* menu = new Fl_Choice(20, 90, sw-40, 30, "Operator Model");
    menu->align(FL_ALIGN_TOP_LEFT); menu->labelcolor(FL_WHITE);
    for(auto& s : ctx.strategies) menu->add(s->getName());
    menu->value(0);

    ctx.desc_box = new Fl_Box(20, 140, sw-40, 100, ctx.strategies[0]->getDesc());
    ctx.desc_box->box(FL_BORDER_BOX); ctx.desc_box->color(fl_rgb_color(25, 30, 40));
    ctx.desc_box->labelcolor(fl_rgb_color(170, 190, 240));
    ctx.desc_box->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);

    int y = 260;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        y += 70; return s;
    };

    auto s_pos = create_s("Base Position (u)", -PI, PI, 0);
    auto s_wid = create_s("Domain Width (du)", 0.1, 2*PI, PI);
    auto s_val = create_s("Fiber Value (v)", -2.0, 2.0, 0.0);
    
    y += 10;
    ctx.s_alpha = create_s(ctx.strategies[0]->getParamALabel(), 0.1, 5.0, 1.0);
    ctx.s_beta = create_s(ctx.strategies[0]->getParamBLabel(), 0.1, 5.0, 1.0);
    auto s_morph = create_s("Temporal Phase (t)", -10.0, 10.0, 0.0);

    auto b_mesh = new Fl_Check_Button(20, y, 140, 25, "Total Space");
    b_mesh->labelcolor(FL_WHITE); b_mesh->value(1);
    auto b_diff = new Fl_Check_Button(170, y, 140, 25, "Show Flows");
    b_diff->labelcolor(FL_WHITE); b_diff->value(1);
    y += 40;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Animate Operators");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);
    sidebar->end();

    SheafView* view = new SheafView(sw, 0, 1200-sw, 800, &g_state);
    
    menu->callback([](Fl_Widget* w, void* d){
        UIContext* c = (UIContext*)d;
        int idx = ((Fl_Choice*)w)->value();
        c->state->current = c->strategies[idx].get();
        c->desc_box->label(c->state->current->getDesc());
        c->s_alpha->label(c->state->current->getParamALabel());
        c->s_beta->label(c->state->current->getParamBLabel());
        c->desc_box->window()->redraw();
    }, &ctx);

    s_pos->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->u_pos = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    s_wid->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->u_width = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    s_val->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->section_val = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    ctx.s_alpha->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->params.alpha = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    ctx.s_beta->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->params.beta = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    s_morph->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->params.morph = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    b_mesh->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showEtale = ((Fl_Check_Button*)w)->value(); }, &g_state);
    b_diff->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->showDifferentials = ((Fl_Check_Button*)w)->value(); }, &g_state);
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.anim_t += 0.02f;
            g_state.params.morph = g_state.anim_t;
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
