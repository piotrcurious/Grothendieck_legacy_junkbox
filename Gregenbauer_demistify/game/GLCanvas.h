#ifndef GL_CANVAS_H
#define GL_CANVAS_H

#include <FL/Fl_Gl_Window.H>
#include <GL/gl.h>
#include <GL/glu.h>
#include "GegenbauerCore.h"

class GLCanvas : public Fl_Gl_Window {
public:
    GegenbauerCore* core = nullptr;

    LayerType current_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    LayerType target_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    double transition_progress = 0.0; // T in [0, 1]
    bool is_animating = false;
    double anim_speed = 0.02;

    BackendType active_backend = BackendType::FLOAT64;

    // Camera state
    float rot_x = 25.0f;
    float rot_y = -35.0f;
    float zoom = 4.5f;
    float pan_x = 0.0f;
    float pan_y = 0.0f;

    // Mouse interaction tracking
    int mouse_last_x = 0;
    int mouse_last_y = 0;
    int mouse_down_button = 0;

    GLCanvas(int x, int y, int w, int h, const char* label = nullptr);

    void set_core(GegenbauerCore* c) { core = c; }
    void set_target_layer(LayerType target);
    void trigger_transition_animation();

    void draw() override;
    int handle(int event) override;

    static void anim_callback(void* userdata);

private:
    void init_gl();
    void setup_lighting();
    void render_hud();
    void draw_string_3d(float x, float y, float z, const char* str);

    // Layer renderers
    void render_layer_geometry(LayerType layer, float alpha);

    void render_layer_i(float alpha);
    void render_layer_ii(float alpha);
    void render_layer_iii(float alpha);
    void render_layer_iv(float alpha);
    void render_layer_v(float alpha);
    void render_layer_vi(float alpha);
    void render_layer_vii(float alpha);
    void render_layer_viii(float alpha);

    // Helper math & color interpolation
    static void get_phi_color(double val, float& r, float& g, float& b);
    static double smoothstep(double t);
};

#endif // GL_CANVAS_H
