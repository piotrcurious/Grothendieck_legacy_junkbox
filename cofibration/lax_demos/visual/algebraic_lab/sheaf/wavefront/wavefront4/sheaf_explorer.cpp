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

class SheafStrategy {
public:
    virtual ~SheafStrategy() {}
    // Now returns a coordinate in the 1-jet space (x, p, z)
    virtual void getPoint(float u, float v, float out[3], const StalkParams& p) = 0;
    
    virtual float getMinV() { return -2.0f; }
    virtual float getMaxV() { return 2.0f; }
    virtual const char* getName() = 0;
    virtual const char* getDesc() = 0;
    virtual const char* getParamALabel() { return "Param A"; }
    virtual const char* getParamBLabel() { return "Param B"; }
};

// 1. Legendrian Swallowtail (Catastrophe Sheaf)
// Rigorous mapping of the 1-jet space J^1(R) where sections are local wavefronts.
class SwallowtailSheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // u acts as the coordinate 'x', v as the momentum 'p'
        // We use the generating function for the A3 singularity: f(x,v) = v^4 + x*v^2 + y*v
        // Legendrian surface is (x, p, z) where p = df/dv=0 and z = f
        float x = u * 4.0f;
        float v_val = v * p.alpha;
        
        // Catastrophe manifold: 4*v^3 + 2*x*v + beta = 0 (solving for beta as fiber)
        float z = powf(v_val, 4.0f) + x * powf(v_val, 2.0f) + p.beta * v_val;
        float slope = 4.0f * powf(v_val, 3.0f) + 2.0f * x * v_val;

        out[0] = x;     // Base Coordinate
        out[1] = z;     // Action (Sheaf Value)
        out[2] = v_val * 5.0f; // Momentum/Fiber
    }
    const char* getName() override { return "Legendrian Swallowtail"; }
    const char* getDesc() override { return "Visualizes the 1-jet space of a wavefront.\nThe multi-valued sheets represent\ncaustics (A3 singularity) where local\nsections cannot be glued globally."; }
    const char* getParamALabel() override { return "Wave Curvature"; }
    const char* getParamBLabel() override { return "Asymmetry"; }
};

// 2. Optical Vortex (U(1) Fibration)
// Models the phase sheaf of a Laguerre-Gaussian beam.
class VortexSheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // u: Radius, v: Angle (The Base is the Transverse Plane)
        float r = (u + PI + 0.1f) * 1.5f;
        float phi = v; 
        
        // Topological charge L
        float L = floorf(p.alpha);
        float phase = L * phi + p.morph;
        
        // The etale space is a helicoid representing the phase winding.
        out[0] = r * cosf(phi);
        out[1] = phase * p.beta; 
        out[2] = r * sinf(phi);
    }
    const char* getName() override { return "Optical Vortex Sheaf"; }
    const char* getDesc() override { return "A U(1) sheaf over the punctured plane.\nThe winding number (topological charge)\ndictates the connectivity of the sheets."; }
    const char* getParamALabel() override { return "Charge (L)"; }
    const char* getParamBLabel() override { return "Pitch Scale"; }
    float getMinV() override { return 0; }
    float getMaxV() override { return 2.0f * PI; }
};

// 3. Analyticity (Kramers-Kronig)
// Real and Imaginary parts of the refractive index as a coherent sheaf.
class AnalyticSheaf : public SheafStrategy {
public:
    void getPoint(float u, float v, float out[3], const StalkParams& p) override {
        // u: Frequency omega, v: Stalk parameter
        float omega = u;
        float res_omega = p.alpha; // Resonance frequency
        float gamma = p.beta;     // Damping
        
        // Lorentz Oscillator Model
        // n^2 - 1 = f / (omega_0^2 - omega^2 - i*gamma*omega)
        float denom_re = res_omega*res_omega - omega*omega;
        float denom_im = -gamma * omega;
        float mag_sq = denom_re*denom_re + denom_im*denom_im + 1e-6f;
        
        float n_re = 1.0f + (denom_re / mag_sq) * 2.0f;
        float n_im = ((-denom_im) / mag_sq) * 2.0f;
        
        // Etale space shows the dispersion and absorption relation
        out[0] = omega * 4.0f;
        out[1] = n_re * 3.0f;
        out[2] = n_im * 5.0f;
    }
    const char* getName() override { return "Kramers-Kronig Sheaf"; }
    const char* getDesc() override { return "The complex refractive index sheaf.\nDispersion (Re) and Absorption (Im)\nare coupled by the Hilbert transform,\nensuring causality in the field."; }
    const char* getParamALabel() override { return "Resonance Freq"; }
    const char* getParamBLabel() override { return "Damping (Gamma)"; }
};

// -----------------------------------------------------------------------------
// UI Logic
// -----------------------------------------------------------------------------
struct UIContext {
    AppState* state;
    Fl_Box* desc_box;
    Fl_Value_Slider *s_alpha, *s_beta;
    std::vector<std::unique_ptr<SheafStrategy>> strategies;

    UIContext(AppState* s) : state(s) {
        strategies.push_back(std::make_unique<SwallowtailSheaf>());
        strategies.push_back(std::make_unique<VortexSheaf>());
        strategies.push_back(std::make_unique<AnalyticSheaf>());
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
        gluPerspective(45, (float)w()/h(), 0.1, 250);
        glMatrixMode(GL_MODELVIEW); glLoadIdentity();
        gluLookAt(25, 25, 35, 0, 0, 0, 0, 1, 0);
        glRotatef(state->rotX, 1, 0, 0);
        glRotatef(state->rotY, 0, 1, 0);

        if (!state->current) return;

        if (state->showEtale) {
            int u_steps = 80, v_steps = 50;
            float u_min = -PI, u_max = PI;
            float v_min = state->current->getMinV(), v_max = state->current->getMaxV();

            glBegin(GL_TRIANGLES);
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
                    
                    glColor4f(0.3f, 0.6f, 1.0f, 0.12f);
                    glVertex3fv(p1); glVertex3fv(p2); glVertex3fv(p3);
                    glVertex3fv(p1); glVertex3fv(p3); glVertex3fv(p4);
                }
            }
            glEnd();
        }

        float u_start = state->u_pos - state->u_width/2.0f;
        float u_end = state->u_pos + state->u_width/2.0f;

        glColor3f(1.0f, 0.5f, 0.0f); glLineWidth(4.0f);
        glBegin(GL_LINE_STRIP);
        for(int i=0; i<=300; i++) {
            float u = u_start + (u_end-u_start)*i/300.0f;
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
    Fl_Double_Window* win = new Fl_Double_Window(1200, 800, "Rigorous Optical Sheaf Analysis");
    static UIContext ctx(&g_state);

    int sw = 320;
    Fl_Group* sidebar = new Fl_Group(0, 0, sw, 800);
    sidebar->box(FL_FLAT_BOX); sidebar->color(fl_rgb_color(10, 12, 18));

    Fl_Box* head = new Fl_Box(20, 20, sw-40, 40, "FIELD ETALE SPACE");
    head->labelcolor(FL_WHITE); head->labelfont(FL_BOLD); head->labelsize(20);

    Fl_Choice* menu = new Fl_Choice(20, 90, sw-40, 30, "Theoretical Model");
    menu->align(FL_ALIGN_TOP_LEFT); menu->labelcolor(FL_WHITE);
    for(auto& s : ctx.strategies) menu->add(s->getName());
    menu->value(0);

    ctx.desc_box = new Fl_Box(20, 140, sw-40, 110, ctx.strategies[0]->getDesc());
    ctx.desc_box->box(FL_BORDER_BOX); ctx.desc_box->color(fl_rgb_color(20, 25, 35));
    ctx.desc_box->labelcolor(fl_rgb_color(180, 200, 250));
    ctx.desc_box->align(FL_ALIGN_WRAP | FL_ALIGN_INSIDE | FL_ALIGN_LEFT);

    int y = 270;
    auto create_s = [&](const char* lbl, float minv, float maxv, float curv) {
        auto s = new Fl_Value_Slider(20, y+25, sw-40, 20, lbl);
        s->type(FL_HOR_NICE_SLIDER); s->bounds(minv, maxv); s->value(curv);
        s->labelcolor(FL_WHITE); s->labelsize(11); s->align(FL_ALIGN_TOP_LEFT);
        y += 70; return s;
    };

    auto s_pos = create_s("Domain Offset (u)", -PI, PI, 0);
    auto s_wid = create_s("Open Cover Size (du)", 0.1, 2*PI, PI);
    auto s_val = create_s("Fiber Selection (v)", -2.0, 2.0, 0.0);
    
    y += 10;
    ctx.s_alpha = create_s(ctx.strategies[0]->getParamALabel(), 0.1, 5.0, 1.0);
    ctx.s_beta = create_s(ctx.strategies[0]->getParamBLabel(), -5.0, 5.0, 0.5);
    auto s_morph = create_s("Field Evolution", -10.0, 10.0, 0.0);

    auto b_mesh = new Fl_Check_Button(20, y, 140, 25, "Etale Mesh");
    b_mesh->labelcolor(FL_WHITE); b_mesh->value(1);
    y += 35;
    auto b_anim = new Fl_Check_Button(20, y, 200, 25, "Animate Propagation");
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
    b_anim->callback([](Fl_Widget* w, void* s){ ((AppState*)s)->animate = ((Fl_Check_Button*)w)->value(); }, &g_state);

    Fl::add_idle([](void* v){
        SheafView* sv = (SheafView*)v;
        if(g_state.animate) {
            g_state.anim_t += 0.03f;
            g_state.params.morph = g_state.anim_t;
        }
        sv->redraw();
    }, view);

    win->resizable(view);
    win->show();
    return Fl::run();
}
