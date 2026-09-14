#ifndef GL_CANVAS_H
#define GL_CANVAS_H

#include <FL/Fl_Gl_Window.H>
#include <GL/gl.h>
#include <GL/glu.h>
#include "GegenbauerCore.h"

// Standardized LayerFrame visual model for morphing transitions
struct Marker {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float r = 1.0f, g = 1.0f, b = 1.0f;
    float size = 5.0f;
};

struct LayerFrame {
    LayerType layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    float geometry_scale = 1.0f;
    float alpha = 1.0f;
    float wave_amplitude = 1.0f;
    float potential_wall_height = 1.0f;
    float tower_spacing = 0.35f;
    float wheel_radius = 1.4f;

    std::vector<Marker> markers;
};

class GLCanvas : public Fl_Gl_Window {
public:
    GegenbauerCore core;
    RepresentationSnapshot snapshot;

    // Visualization toggles
    bool show_live_overlays = true;

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

    void set_snapshot(const RepresentationSnapshot& snap) { snapshot = snap; redraw(); }

    void draw() override;
    int handle(int event) override;

    static LayerFrame make_layer_frame(LayerType layer, const RepresentationSnapshot& snap);
    static LayerFrame interpolate_layer_frame(const LayerFrame& f1, const LayerFrame& f2, double t);

private:
    void init_gl();
    void setup_lighting();
    void render_hud();

    // Layer renderers
    void render_layer_frame(const LayerFrame& frame, const RepresentationSnapshot& snap);

    void render_layer_i(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_ii(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_iii(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_iv(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_v(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_vi(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_vii(const LayerFrame& frame, const RepresentationSnapshot& snap);
    void render_layer_viii(const LayerFrame& frame, const RepresentationSnapshot& snap);

    // Helper math & color interpolation
    static void get_phi_color(double val, float& r, float& g, float& b);
    static double smoothstep(double t);
};

#endif // GL_CANVAS_H
