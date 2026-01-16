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
// Mathematical Framework: State & Parameters
// -----------------------------------------------------------------------------
struct StalkParams {
    float alpha = 1.0f; 
    float beta  = 0.5f; 
    float morph = 0.0f; 
};

class SheafStrategy;

struct AppState {
    float rotX = 25.0f, rotY = -45.0f;
    float u_pos = PI, u_width = 2.0f * PI, section_val = 0.0f;
    StalkParams params;
    SheafStrategy* current = nullptr; 
    bool showEtale = true;
    bool showDifferentials = true; 
    bool animate = false;
    float anim_t = 0.0f;
};

// -----------------------------------------------------------------------------
// Optics-Inspired Sheaf Strategies
// -----------------------------------------------------------------------------
class SheafStrategy {
public:
    virtual ~SheafStrategy() {}
    virtual void getPoint(float theta, float val, float out[3], const StalkParams& p) = 0;
    
    virtual void getTangent(float theta, float val, float out[3], const StalkParams& p) {
        float p1[3], p2[3], dt = 0.005f;
        getPoint(theta - dt, val, p1, p);
        getPoint(theta + dt, val, p2, p);
        for(int i=0; i<3; i++) out[i] = (p2[i] - p1[i]) / (2.0f * dt);
    }

    virtual float getMinVal() { return -2.5f; }
    virtual float getMaxVal() { return 2.5f; }
    virtual const char* getName() = 0;
    virtual const char* getDesc() = 0;
    virtual const char* getParamALabel() { return "Param A"; }
    virtual const char* getParamBLabel() { return "Param B"; }
    virtual bool isMultiSheeted() { return false; }
};

// 1. Gabor-Logon Sheaf (Time-Frequency Optics)
// Represents the sheaf of local spectral components of a signal
class GaborSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3], const StalkParams& p) override {
        float r = 4.5f;
        // Local frequency oscillation modulated by a Gaussian window
        float window = expf(-powf(sinf(theta*0.5f), 2.0f) / (0.1f * p.beta));
        float freq = p.alpha * sinf(theta + p.morph);
        float h = val + window * sinf(theta * 10.0f * freq);
        
        out[0] = r * cosf(theta);
        out[1] = h;
        out[2] = r * sinf(theta);
    }
    const char* getName() override { return "Gabor Phase Sheaf"; }
    const char* getDesc() override { return "Models local frequency as a sheaf.\nSections represent the 'Logon' or\ninformation atom in optical signals."; }
    const char* getParamALabel() override { return "Carrier Freq"; }
    const char* getParamBLabel() override { return "Window Width"; }
};

// 2. Zernike Aberration Sheaf (Wavefront Sensing)
// Models the sheaf of phase errors (Coma / Astigmatism)
class ZernikeSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3], const StalkParams& p) override {
        float r = 4.5f + val;
        // Z_3^-1 (Coma): (3rho^3 - 2rho)sin(theta)
        // We project the aberration onto the etale cylinder
        float rho = 1.0f; 
        float coma = (3.0f*powf(rho,3) - 2.0f*rho) * sinf(theta * p.alpha);
        float astig = powf(rho,2) * cosf(2.0f*theta + p.morph);
        
        float h = p.beta * (coma + astig);
        
        out[0] = r * cosf(theta);
        out[1] = h;
        out[2] = r * sinf(theta);
    }
    const char* getName() override { return "Zernike Aberration"; }
    const char* getDesc() override { return "Sheaf of wavefront errors.\nSections represent local phase\ndeformations in an optical system."; }
    const char* getParamALabel() override { return "Azimuthal Mode"; }
    const char* getParamBLabel() override { return "RMS Amplitude"; }
};

// 3. Pearcey Caustic Sheaf (Catastrophe Optics)
// Models the folding of ray-paths near a caustic point
class CausticSheaf : public SheafStrategy {
public:
    void getPoint(float theta, float val, float out[3], const StalkParams& p) override {
        float R = 4.5f;
        // A Pearcey-type fold: x^4 + ax^2 + bx
        // We treat the fiber as the control space of the catastrophe
        float x = val;
        float a = p.alpha * cosf(theta);
        float b = p.beta * sinf(theta + p.morph);
        
        // The etale surface is the 'equilibrium' manifold: 4x^3 + 2ax + b = 0
        // We visualize the potential surface itself
        float potential = powf(x, 4.0f) + a*powf(x, 2.0f) + b*x;
        
        out[0] = (R + x) * cosf(theta);
        out[1] = potential * 0.5f;
        out[2] = (R + x) * sinf(theta);
    }
    const char* getName() override { return "Caustic Catastrophe"; }
    const char* getDesc() override { return "Models ray-path folding.\nThe sheaf branches where multiple\nrays interfere (caustics)."; }
    const char* getParamALabel() override { return "Fold Tension"; }
    const char* getParamBLabel() override { return "Bifurcation"; }
};

// -----------------------------------------------------------------------------
// Graphics Engine
// -----------------------------------------------------------------------------
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
        glClearColor(0.002f, 0.005f, 0.01f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION); glLoadIdentity();
        gluPerspective(42, (float)w()/h(), 0.1, 150);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(0, 18, 38, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if (!state->current) return;

        // Base Reference
        glColor4f(0.3f, 0.4f, 0.6f, 0.1f);
        glBegin(GL_LINE_LOOP);
        for(int i=0; i<100; i++) glVertex3f(4.5f*cosf(2*PI*i/100.0f), -10.0f, 4.5f*sinf(2*PI*i/100.0f));
        glEnd();

        // Etale Total Space (Field Distribution)
        if (state->showEtale) {
            glColor4f(0.0f, 0.6f, 0.9f, 0.08f);
            int t_steps = 180, v_steps = 12;
            float range = 2.0f * PI; 
            for(int i=0; i<t_steps; i++) {
                float t1 = range * i/t_steps, t2 = range * (i+1)/t_steps;
                glBegin(GL_QUAD_STRIP);
                for(int j=0; j<=v_steps; j++) {
                    float v = state->current->getMinVal() + (state->current->getMaxVal()-state->current->getMinVal())*j/v_steps;
                    float p1[3], p2[3];
                    state->current->getPoint(t1, v, p1, state->params); glVertex3fv(p1);
                    state->current->getPoint(t2, v, p2, state->params); glVertex3fv(p2);
                }
                glEnd();
            }
        }

        // Section (Wavefront Fragment)
        float start = state->u_pos - state->u_width/2.0f;
        float end = state->u_pos + state->u_width/2.0f;

        if (state->showDifferentials) {
            glColor4f(0.0f, 1.0f, 0.8f, 0.5f);
            glBegin(GL_LINES);
            for(int i=0; i<=50; i++) {
                float t = start + (end-start)*i/50.0f;
                float p[3], tan[3];
                state->current->getPoint(t, state->section_val, p, state->params);
                state->current->getTangent(t, state->section_val, tan, state->params);
                glVertex3fv(p);
                glVertex3f(p[0] + tan[0]*0.15f, p[1] + tan[1]*0.15f, p[2] + tan[2]*0.15f);
            }
            glEnd();
        }

        glColor3f(1.0f, 0.4f, 0.1f);
        glLineWidth(2.5f);
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=300; i++) {
            float t = start + (end-start)*i/300.0f;
            float p[3]; state->current->getPoint(t, state->section_val, p, state->params);
            glVertex3fv(p);
        }
        glEnd();
        glLineWidth(1.0f);
    }

    int handle(int e) override {
        static int lx, ly;
        if(e == FL_PUSH) { lx = Fl::event_x(); ly = Fl::event_y(); return 1; }
        if(e == FL_DRAG) {
            state->rotY += (float)(Fl::event_x()-lx) * 0.35f;
            state->rotX += (float)(Fl::event_y()-ly) * 0.35f;
            lx = Fl::event_x(); ly = Fl::event_y(); redraw(); return 1;
        }
        return Fl_Gl_Window::handle(e);
    }
};

// -----------------------------------------------------------------------------
// UI Construction
// -----------------------------------------------------------------------------
struct UIContext {
    AppState* state;
    Fl_Box* desc_box;
    Fl_Value_Slider* s_pos, *s_alpha, *s_beta;
    std::vector<std::unique_ptr<SheafStrategy>> strategies;

    UIContext(AppState* s) : state(s) {
        strategies.push_back(std::make_unique<GaborSheaf>());
        strategies.push_back(std::make_unique<ZernikeSheaf>());
        strategies.push_back(std::make_unique<CausticSheaf>());
        state->current = strategies[0].get();
    }
};

int main() {
    static AppState g_state;
    Fl_Double_Window* win = new Fl_Double_Window(1280, 850, "Optics Sheaf Laboratory");
    static UIContext ctx(&g_state);

    int sw = 340;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 850);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(10, 12, 18));

    Fl_Box* head = new Fl_Box(20, 25, sw-40, 40, "WAVEFRONT ETALE");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(26);

    Fl_Choice* menu = new Fl_Choice(20, 100, sw-40, 30, "Optical Mapping");
    menu->align(FL_ALIGN_TOP_LEFT); menu->labelcolor(FL_WHITE);
    for(auto& s : ctx.strategies) menu->add(s->getName());
    menu->value(0);

    ctx.desc_box = new Fl_Box(20, 150, sw-40, 110, ctx.strategies[0]->getDesc());
    ctx.desc_box->box(FL_BORDER_BOX); ctx.desc_box->color(fl_rgb_color(25, 28, 35));
    ctx.desc_box->labelcolor(fl_rgb_color(160, 180, 220));
    ctx.desc_box->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);

    int y = 290;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(12); s->align(FL_ALIGN_TOP_LEFT);
        y += 75; return s;
    };

    ctx.s_pos = create_s("Aperture Position (u)", -2*PI, 6*PI, PI);
    auto s_width = create_s("Section Aperture (du)", 0.1, 4*PI, 2*PI);
    auto s_val = create_s("Radial / Control Value", -2.5, 2.5, 0.0);

    y += 10;
    Fl_Box* sep = new Fl_Box(20, y, sw-40, 1); sep->box(FL_FLAT_BOX); sep->color(FL_DARK3);
    y += 20;

    ctx.s_alpha = create_s(ctx.strategies[0]->getParamALabel(), 0.1, 8.0, 1.0);
    ctx.s_beta = create_s(ctx.strategies[0]->getParamBLabel(), 0.0, 2.0, 0.5);
    auto s_morph = create_s("Time / Phase Shift", -5.0, 5.0, 0.0);

    auto b_mesh = new Fl_Check_Button(20, y, 140, 25, "Render Field");
    b_mesh->labelcolor(FL_WHITE); b_mesh->value(1);
    auto b_diff = new Fl_Check_Button(180, y, 140, 25, "Wavefront Tilt");
    b_diff->labelcolor(FL_WHITE); b_diff->value(1);
    y += 45;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Live Wave Propagation");
    b_anim->labelcolor(FL_WHITE); b_anim->value(0);

    sidebar->end();
    SheafView* view = new SheafView(sw, 0, 1280-sw, 850, &g_state);

    menu->callback([](Fl_Widget* w, void* d){
        UIContext* c = (UIContext*)d;
        int idx = ((Fl_Choice*)w)->value();
        c->state->current = c->strategies[idx].get();
        c->desc_box->label(c->state->current->getDesc());
        c->s_alpha->label(c->state->current->getParamALabel());
        c->s_beta->label(c->state->current->getParamBLabel());
        c->desc_box->window()->redraw();
    }, &ctx);

    ctx.s_pos->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->u_pos = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
    s_width->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->u_width = (float)((Fl_Value_Slider*)w)->value(); }, &g_state);
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
            g_state.anim_t += 0.015f;
            g_state.params.morph = sinf(g_state.anim_t) * 2.0f;
            ctx.s_pos->value(g_state.u_pos);
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
