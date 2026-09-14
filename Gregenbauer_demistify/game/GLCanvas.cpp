#include "GLCanvas.h"
#include <FL/Fl.H>
#include <FL/gl.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

GLCanvas::GLCanvas(int x, int y, int w, int h, const char* label)
    : Fl_Gl_Window(x, y, w, h, label) {
    mode(FL_RGB | FL_DOUBLE | FL_DEPTH);
}

void GLCanvas::update_snapshot() {
    if (core) {
        snapshot = core->get_snapshot(target_layer, active_backend, auto_router);
        if (auto_router) {
            target_layer = snapshot.layer;
            active_backend = snapshot.backend;
        }
    }
}

void GLCanvas::set_target_layer(LayerType target) {
    target_layer = target;
    update_snapshot();
    transition_progress = 0.0;
    is_animating = true;
    Fl::remove_timeout(anim_callback, this);
    Fl::add_timeout(1.0 / 60.0, anim_callback, this);
    redraw();
}

void GLCanvas::trigger_transition_animation() {
    update_snapshot();
    transition_progress = 0.0;
    is_animating = true;
    Fl::remove_timeout(anim_callback, this);
    Fl::add_timeout(1.0 / 60.0, anim_callback, this);
    redraw();
}

void GLCanvas::anim_callback(void* userdata) {
    GLCanvas* canvas = static_cast<GLCanvas*>(userdata);
    if (!canvas->is_animating) return;

    canvas->transition_progress += canvas->anim_speed;
    if (canvas->transition_progress >= 1.0) {
        canvas->transition_progress = 1.0;
        canvas->current_layer = canvas->target_layer;
        canvas->is_animating = false;
    } else {
        Fl::repeat_timeout(1.0 / 60.0, anim_callback, canvas);
    }
    canvas->redraw();
}

double GLCanvas::smoothstep(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

void GLCanvas::get_phi_color(double val, float& r, float& g, float& b) {
    double norm = (val + 1.0) * 0.5;
    norm = std::clamp(norm, 0.0, 1.0);
    if (norm < 0.5) {
        float t = static_cast<float>(norm * 2.0);
        r = 0.1f * (1.0f - t) + 0.2f * t;
        g = 0.4f * (1.0f - t) + 0.6f * t;
        b = 0.9f * (1.0f - t) + 0.8f * t;
    } else {
        float t = static_cast<float>((norm - 0.5) * 2.0);
        r = 0.2f * (1.0f - t) + 1.0f * t;
        g = 0.6f * (1.0f - t) + 0.8f * t;
        b = 0.8f * (1.0f - t) + 0.2f * t;
    }
}

void GLCanvas::init_gl() {
    glViewport(0, 0, w(), h());
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, static_cast<double>(w()) / static_cast<double>(h() ? h() : 1), 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glClearColor(0.04f, 0.06f, 0.10f, 1.0f);
}

void GLCanvas::setup_lighting() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

    GLfloat light_pos[] = { 5.0f, 10.0f, 8.0f, 1.0f };
    GLfloat light_ambient[] = { 0.2f, 0.25f, 0.35f, 1.0f };
    GLfloat light_diffuse[] = { 0.9f, 0.9f, 0.95f, 1.0f };
    GLfloat light_specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };

    glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
}

void GLCanvas::draw() {
    if (!valid()) {
        init_gl();
        valid(1);
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(pan_x, pan_y, -zoom);
    glRotatef(rot_x, 1.0f, 0.0f, 0.0f);
    glRotatef(rot_y, 0.0f, 1.0f, 0.0f);

    setup_lighting();

    if (!core) return;
    update_snapshot();

    double s = smoothstep(transition_progress);
    float alpha_curr = static_cast<float>(1.0 - s);
    float alpha_targ = static_cast<float>(s);

    if (current_layer == target_layer || transition_progress >= 1.0) {
        render_layer_geometry(target_layer, 1.0f);
    } else {
        render_layer_geometry(current_layer, alpha_curr);
        render_layer_geometry(target_layer, alpha_targ);
    }

    render_hud();
}

int GLCanvas::handle(int event) {
    switch (event) {
        case FL_PUSH:
            mouse_last_x = Fl::event_x();
            mouse_last_y = Fl::event_y();
            mouse_down_button = Fl::event_button();
            return 1;
        case FL_DRAG: {
            int dx = Fl::event_x() - mouse_last_x;
            int dy = Fl::event_y() - mouse_last_y;
            if (mouse_down_button == FL_LEFT_MOUSE) {
                rot_y += dx * 0.5f;
                rot_x += dy * 0.5f;
            } else if (mouse_down_button == FL_RIGHT_MOUSE || mouse_down_button == FL_MIDDLE_MOUSE) {
                pan_x += dx * 0.005f * zoom;
                pan_y -= dy * 0.005f * zoom;
            }
            mouse_last_x = Fl::event_x();
            mouse_last_y = Fl::event_y();
            redraw();
            return 1;
        }
        case FL_MOUSEWHEEL:
            zoom += Fl::event_dy() * 0.2f;
            zoom = std::clamp(zoom, 1.0f, 20.0f);
            redraw();
            return 1;
        default:
            return Fl_Gl_Window::handle(event);
    }
}

void GLCanvas::render_layer_geometry(LayerType layer, float alpha) {
    switch (layer) {
        case LayerType::LAYER_I_HARMONIC_GEOMETRY: render_layer_i(alpha); break;
        case LayerType::LAYER_II_PROJECTOR_TEMPLE: render_layer_ii(alpha); break;
        case LayerType::LAYER_III_DIFFERENTIAL_WAVE: render_layer_iii(alpha); break;
        case LayerType::LAYER_IV_JACOBI_CITY: render_layer_iv(alpha); break;
        case LayerType::LAYER_V_TWO_POLE_COORDS: render_layer_v(alpha); break;
        case LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS: render_layer_vi(alpha); break;
        case LayerType::LAYER_VII_ARITHMETIC_FACTORY: render_layer_vii(alpha); break;
        case LayerType::LAYER_VIII_CERTIFICATION_CHAMBER: render_layer_viii(alpha); break;
    }
}

// -------------------------------------------------------------
// Layer Renderers
// -------------------------------------------------------------

void GLCanvas::render_layer_i(float alpha) {
    int lat_steps = 40;
    int lon_steps = 60;
    float radius = 1.5f;

    glEnable(GL_LIGHTING);
    for (int i = 0; i < lat_steps; ++i) {
        float lat0 = M_PI * (-0.5f + static_cast<float>(i) / lat_steps);
        float z0_val = std::sin(lat0);
        float r0_val = std::cos(lat0);

        float lat1 = M_PI * (-0.5f + static_cast<float>(i + 1) / lat_steps);
        float z1_val = std::sin(lat1);
        float r1_val = std::cos(lat1);

        double th0 = M_PI * 0.5 - lat0;
        double th1 = M_PI * 0.5 - lat1;
        double phi_val0 = core->eval_phi(snapshot.n, std::cos(th0));
        double phi_val1 = core->eval_phi(snapshot.n, std::cos(th1));

        float r_c0, g_c0, b_c0, r_c1, g_c1, b_c1;
        get_phi_color(phi_val0, r_c0, g_c0, b_c0);
        get_phi_color(phi_val1, r_c1, g_c1, b_c1);

        glBegin(GL_TRIANGLE_STRIP);
        for (int j = 0; j <= lon_steps; ++j) {
            float lng = 2.0f * M_PI * static_cast<float>(j) / lon_steps;
            float x0_pos = std::cos(lng) * r0_val;
            float y0_pos = std::sin(lng) * r0_val;
            float x1_pos = std::cos(lng) * r1_val;
            float y1_pos = std::sin(lng) * r1_val;

            float disp0 = 1.0f + 0.15f * static_cast<float>(phi_val0);
            float disp1 = 1.0f + 0.15f * static_cast<float>(phi_val1);

            glColor4f(r_c0, g_c0, b_c0, alpha * 0.9f);
            glNormal3f(x0_pos, y0_pos, z0_val);
            glVertex3f(x0_pos * radius * disp0, z0_val * radius * disp0, y0_pos * radius * disp0);

            glColor4f(r_c1, g_c1, b_c1, alpha * 0.9f);
            glNormal3f(x1_pos, y1_pos, z1_val);
            glVertex3f(x1_pos * radius * disp1, z1_val * radius * disp1, y1_pos * radius * disp1);
        }
        glEnd();
    }

    glDisable(GL_LIGHTING);
    glLineWidth(2.5f);
    glColor4f(1.0f, 0.9f, 0.2f, alpha);
    int th_samples = 200;
    for (int k = 0; k < th_samples; ++k) {
        double th_curr = M_PI * static_cast<double>(k) / th_samples;
        double th_next = M_PI * static_cast<double>(k + 1) / th_samples;
        double p_c = core->eval_phi(snapshot.n, std::cos(th_curr));
        double p_n = core->eval_phi(snapshot.n, std::cos(th_next));

        if (p_c * p_n <= 0.0) {
            double th_zero = 0.5 * (th_curr + th_next);
            float lat_z = M_PI * 0.5f - static_cast<float>(th_zero);
            float r_z = radius * std::cos(lat_z);
            float z_z = radius * std::sin(lat_z);

            glBegin(GL_LINE_LOOP);
            for (int j = 0; j < 60; ++j) {
                float lng = 2.0f * M_PI * static_cast<float>(j) / 60;
                glVertex3f(std::cos(lng) * r_z, z_z, std::sin(lng) * r_z);
            }
            glEnd();
        }
    }
}

void GLCanvas::render_layer_ii(float alpha) {
    glDisable(GL_LIGHTING);

    // Luminous central K-fixed ray
    glLineWidth(4.0f);
    glColor4f(0.2f, 0.9f, 1.0f, alpha);
    glBegin(GL_LINES);
    glVertex3f(0.0f, -2.5f, 0.0f);
    glVertex3f(0.0f, 2.5f, 0.0f);
    glEnd();

    // Rank-one projector ray v_n \otimes v_n^* cloud
    int num_particles = 120;
    glPointSize(5.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < num_particles; ++i) {
        float angle_p = static_cast<float>(i) * 0.35f;
        float rad_p = 0.5f + 1.2f * std::sin(angle_p * 2.3f);
        float y_p = -2.0f + 4.0f * (static_cast<float>(i) / num_particles);
        float x_p = std::cos(angle_p) * rad_p;
        float z_p = std::sin(angle_p) * rad_p;

        glColor4f(0.5f, 0.4f, 0.9f, alpha * 0.6f);
        glVertex3f(x_p, y_p, z_p);
    }
    glEnd();

    glLineWidth(1.2f);
    glColor4f(1.0f, 0.6f, 0.2f, alpha * 0.4f);
    glBegin(GL_LINES);
    for (int i = 0; i < num_particles; i += 3) {
        float angle_p = static_cast<float>(i) * 0.35f;
        float rad_p = 0.5f + 1.2f * std::sin(angle_p * 2.3f);
        float y_p = -2.0f + 4.0f * (static_cast<float>(i) / num_particles);
        float x_p = std::cos(angle_p) * rad_p;
        float z_p = std::sin(angle_p) * rad_p;

        glVertex3f(x_p, y_p, z_p);
        glVertex3f(0.0f, y_p, 0.0f);
    }
    glEnd();

    float v_y = static_cast<float>(snapshot.phi) * 2.0f;
    glColor4f(1.0f, 1.0f, 0.3f, alpha);
    glPointSize(10.0f);
    glBegin(GL_POINTS);
    glVertex3f(0.0f, v_y, 0.0f);
    glEnd();
}

void GLCanvas::render_layer_iii(float alpha) {
    glDisable(GL_LIGHTING);

    int samples = 200;
    float x_start = -2.2f;
    float x_end = 2.2f;

    // Potential wall surface V(theta)
    glLineWidth(2.5f);
    glColor4f(1.0f, 0.3f, 0.3f, alpha * 0.8f);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = 0.005 + (M_PI - 0.01) * (static_cast<double>(i) / samples);
        double v_val = core->eval_potential_v(th_val);
        float pos_x = x_start + (x_end - x_start) * (static_cast<float>(i) / samples);
        float pos_y = std::min(3.0f, static_cast<float>(v_val * 0.2));
        glVertex3f(pos_x, pos_y - 1.0f, 0.0f);
    }
    glEnd();

    // Schrödinger wave u_n(theta)
    glLineWidth(3.5f);
    glColor4f(0.2f, 1.0f, 0.5f, alpha);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = 0.005 + (M_PI - 0.01) * (static_cast<double>(i) / samples);
        double u_val = core->eval_schrodinger_u(snapshot.n, th_val);
        float pos_x = x_start + (x_end - x_start) * (static_cast<float>(i) / samples);
        float pos_y = static_cast<float>(u_val * 1.5);
        glVertex3f(pos_x, pos_y, 0.2f);
    }
    glEnd();

    // Energy baseline N_n^2
    float e_y = std::min(2.8f, static_cast<float>(snapshot.N * 0.1));
    glLineWidth(1.5f);
    glColor4f(0.4f, 0.8f, 1.0f, alpha * 0.5f);
    glBegin(GL_LINES);
    glVertex3f(x_start, e_y, 0.0f);
    glVertex3f(x_end, e_y, 0.0f);
    glEnd();
}

void GLCanvas::render_layer_iv(float alpha) {
    glEnable(GL_LIGHTING);

    int m_nodes = std::min(15, snapshot.n + 3);
    GolubWelschResult gw = core->compute_golub_welsch(m_nodes);
    float spacing = 0.35f;
    float start_x = -0.5f * (m_nodes - 1) * spacing;

    for (int k = 0; k < m_nodes; ++k) {
        float pos_x = start_x + k * spacing;
        double alpha_k = (k < m_nodes - 1) ? GegenbauerCore::get_jacobi_alpha(k, snapshot.lambda) : 0.0;

        float tower_height = 0.5f + static_cast<float>(alpha_k) * 2.0f;
        glColor4f(0.3f, 0.7f, 0.9f, alpha * 0.8f);

        glPushMatrix();
        glTranslatef(pos_x, tower_height * 0.5f - 1.0f, 0.0f);
        glScalef(0.12f, tower_height, 0.12f);
        glBegin(GL_QUADS);
        glNormal3f(0, 0, 1); glVertex3f(-1, -1, 1); glVertex3f(1, -1, 1); glVertex3f(1, 1, 1); glVertex3f(-1, 1, 1);
        glNormal3f(0, 0, -1); glVertex3f(-1, -1, -1); glVertex3f(-1, 1, -1); glVertex3f(1, 1, -1); glVertex3f(1, -1, -1);
        glEnd();
        glPopMatrix();

        if (k < m_nodes - 1) {
            glDisable(GL_LIGHTING);
            glLineWidth(3.0f * static_cast<float>(alpha_k * 2.0));
            glColor4f(1.0f, 0.8f, 0.2f, alpha);
            glBegin(GL_LINES);
            glVertex3f(pos_x, tower_height - 1.0f, 0.0f);
            glVertex3f(pos_x + spacing, (0.5f + static_cast<float>(GegenbauerCore::get_jacobi_alpha(k + 1, snapshot.lambda)) * 2.0f) - 1.0f, 0.0f);
            glEnd();
            glEnable(GL_LIGHTING);
        }
    }

    glDisable(GL_LIGHTING);
    glLineWidth(2.0f);
    for (int k = 0; k < m_nodes; ++k) {
        float x_val = static_cast<float>(gw.eigenvalues[k]);
        float line_x = x_val * 2.0f;
        glColor4f(0.2f, 1.0f, 0.4f, alpha * 0.7f);
        glBegin(GL_LINES);
        glVertex3f(line_x, -1.8f, -0.5f);
        glVertex3f(line_x, -1.2f, -0.5f);
        glEnd();
    }
}

void GLCanvas::render_layer_v(float alpha) {
    glDisable(GL_LIGHTING);

    int samples = 100;
    float w = 1.3f;

    // 1. North Pole
    glLineWidth(2.5f);
    glColor4f(0.2f, 0.9f, 1.0f, alpha);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = 0.0001 + 0.3 * (static_cast<double>(i) / samples);
        double bessel_val = core->eval_north_bessel(th_val);
        float pos_x = -2.2f + w * (static_cast<float>(i) / samples);
        float pos_y = static_cast<float>(bessel_val * 1.2);
        glVertex3f(pos_x, pos_y, 0.0f);
    }
    glEnd();

    // 2. Center WKB
    glColor4f(0.4f, 1.0f, 0.5f, alpha);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = 0.3 + (M_PI - 0.6) * (static_cast<double>(i) / samples);
        double wkb_val = core->eval_wkb_interior(th_val);
        float pos_x = -0.65f + w * (static_cast<float>(i) / samples);
        float pos_y = static_cast<float>(wkb_val * 1.2);
        glVertex3f(pos_x, pos_y, 0.0f);
    }
    glEnd();

    // 3. South Pole
    glColor4f(1.0f, 0.7f, 0.2f, alpha);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = M_PI - 0.3 + 0.3 * (static_cast<double>(i) / samples);
        double bessel_val = core->eval_south_bessel(th_val);
        float pos_x = 0.9f + w * (static_cast<float>(i) / samples);
        float pos_y = static_cast<float>(bessel_val * 1.2);
        glVertex3f(pos_x, pos_y, 0.0f);
    }
    glEnd();

    glLineWidth(1.0f);
    glColor4f(0.6f, 0.6f, 0.7f, alpha * 0.4f);
    float box_xs[] = { -2.2f, -0.65f, 0.9f };
    for (float bx : box_xs) {
        glBegin(GL_LINE_LOOP);
        glVertex3f(bx, -1.5f, 0.0f);
        glVertex3f(bx + w, -1.5f, 0.0f);
        glVertex3f(bx + w, 1.5f, 0.0f);
        glVertex3f(bx, 1.5f, 0.0f);
        glEnd();
    }
}

void GLCanvas::render_layer_vi(float alpha) {
    glDisable(GL_LIGHTING);

    int samples = 200;
    float x_start = -2.2f;
    float x_end = 2.2f;

    // Domain masks
    glBegin(GL_QUADS);
    glColor4f(0.2f, 0.6f, 1.0f, alpha * 0.15f);
    glVertex3f(x_start, -1.8f, -0.1f); glVertex3f(x_start + 1.2f, -1.8f, -0.1f);
    glVertex3f(x_start + 1.2f, 1.8f, -0.1f); glVertex3f(x_start, 1.8f, -0.1f);

    glColor4f(0.8f, 0.4f, 1.0f, alpha * 0.25f);
    glVertex3f(x_start + 0.9f, -1.8f, -0.05f); glVertex3f(x_start + 1.5f, -1.8f, -0.05f);
    glVertex3f(x_start + 1.5f, 1.8f, -0.05f); glVertex3f(x_start + 0.9f, 1.8f, -0.05f);

    glColor4f(1.0f, 0.6f, 0.2f, alpha * 0.15f);
    glVertex3f(x_end - 1.2f, -1.8f, -0.1f); glVertex3f(x_end, -1.8f, -0.1f);
    glVertex3f(x_end, 1.8f, -0.1f); glVertex3f(x_end - 1.2f, 1.8f, -0.1f);
    glEnd();

    // Composite matched wave F_comp(theta)
    glLineWidth(3.5f);
    glColor4f(1.0f, 1.0f, 0.3f, alpha);
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= samples; ++i) {
        double th_val = 0.001 + (M_PI - 0.002) * (static_cast<double>(i) / samples);
        double f_comp = core->eval_composite_asymptotics(th_val);
        float pos_x = x_start + (x_end - x_start) * (static_cast<float>(i) / samples);
        float pos_y = static_cast<float>(f_comp * 1.5);
        glVertex3f(pos_x, pos_y, 0.1f);
    }
    glEnd();

    // Exact curve comparison
    if (show_composite_live) {
        glLineWidth(1.5f);
        glColor4f(0.2f, 1.0f, 0.8f, alpha * 0.7f);
        glBegin(GL_LINE_STRIP);
        for (int i = 0; i <= samples; ++i) {
            double th_val = 0.001 + (M_PI - 0.002) * (static_cast<double>(i) / samples);
            double f_exact = core->eval_phi(snapshot.n, std::cos(th_val));
            float pos_x = x_start + (x_end - x_start) * (static_cast<float>(i) / samples);
            float pos_y = static_cast<float>(f_exact * 1.5);
            glVertex3f(pos_x, pos_y, 0.12f);
        }
        glEnd();
    }
}

void GLCanvas::render_layer_vii(float alpha) {
    glDisable(GL_LIGHTING);

    const char* backend_names[] = {
        "FLOAT32", "FLOAT64", "LONGDOUBLE", "Q16.16", "LNS", "EXACT_RATIONAL", "MODULAR_RNS"
    };

    float start_y = 1.8f;
    float track_height = 0.5f;

    for (int i = 0; i < 7; ++i) {
        BackendType bt = static_cast<BackendType>(i);
        (void)backend_names;
        double backend_phi = core->eval_backend_phi(bt, snapshot.n, snapshot.x);
        double exact_phi = core->eval_phi();
        double err = std::abs(backend_phi - exact_phi);

        float ty = start_y - i * track_height;

        glLineWidth(2.0f);
        if (bt == snapshot.backend) {
            glColor4f(0.3f, 1.0f, 0.4f, alpha);
        } else {
            glColor4f(0.5f, 0.6f, 0.8f, alpha * 0.5f);
        }

        glBegin(GL_LINES);
        glVertex3f(-2.2f, ty, 0.0f);
        glVertex3f(2.2f, ty, 0.0f);
        glEnd();

        if (err > 1e-12) {
            int num_noise = std::min(20, static_cast<int>(err * 1e6) + 2);
            glPointSize(4.0f);
            glColor4f(1.0f, 0.3f, 0.3f, alpha * 0.7f);
            glBegin(GL_POINTS);
            for (int k = 0; k < num_noise; ++k) {
                float nx = -2.0f + 4.0f * (static_cast<float>(k) / num_noise);
                float ny = ty + 0.05f * std::sin(k * 3.7f);
                glVertex3f(nx, ny, 0.05f);
            }
            glEnd();
        }
    }
}

void GLCanvas::render_layer_viii(float alpha) {
    glDisable(GL_LIGHTING);

    // 4-Axis Verification Wheel
    float radius = 1.4f;
    glLineWidth(3.0f);
    glColor4f(0.3f, 0.8f, 1.0f, alpha * 0.7f);
    glBegin(GL_LINE_LOOP);
    for (int i = 0; i < 60; ++i) {
        float ang = 2.0f * M_PI * static_cast<float>(i) / 60;
        glVertex3f(std::cos(ang) * radius, std::sin(ang) * radius, 0.0f);
    }
    glEnd();

    // 4 Independent Axes
    glLineWidth(2.0f);
    // Axis 1: Algebraic
    glColor4f(snapshot.cert.algebraic_exact ? 0.2f : 0.8f, snapshot.cert.algebraic_exact ? 1.0f : 0.3f, 0.3f, alpha);
    glBegin(GL_LINES); glVertex3f(-radius, 0.0f, 0.0f); glVertex3f(radius, 0.0f, 0.0f); glEnd();

    // Axis 2: Arithmetic
    glColor4f(snapshot.cert.arithmetic_exact ? 0.2f : 0.8f, snapshot.cert.arithmetic_exact ? 1.0f : 0.3f, 0.3f, alpha);
    glBegin(GL_LINES); glVertex3f(0.0f, -radius, 0.0f); glVertex3f(0.0f, radius, 0.0f); glEnd();

    // Axis 3: Analytic
    glColor4f(snapshot.cert.analytic_certified ? 0.2f : 0.8f, snapshot.cert.analytic_certified ? 1.0f : 0.3f, 0.3f, alpha);
    glBegin(GL_LINES); glVertex3f(-radius * 0.7f, -radius * 0.7f, 0.0f); glVertex3f(radius * 0.7f, radius * 0.7f, 0.0f); glEnd();

    // Axis 4: Numerical
    glColor4f(snapshot.cert.numerical_valid ? 0.2f : 0.8f, snapshot.cert.numerical_valid ? 1.0f : 0.3f, 0.3f, alpha);
    glBegin(GL_LINES); glVertex3f(-radius * 0.7f, radius * 0.7f, 0.0f); glVertex3f(radius * 0.7f, -radius * 0.7f, 0.0f); glEnd();

    // Luminous Certification Badge if valid
    if (snapshot.cert.algebraic_exact || snapshot.cert.arithmetic_exact || snapshot.cert.analytic_certified) {
        glLineWidth(4.0f);
        glColor4f(0.2f, 1.0f, 0.4f, alpha);
        glBegin(GL_LINE_LOOP);
        for (int i = 0; i < 8; ++i) {
            float ang = 2.0f * M_PI * static_cast<float>(i) / 8;
            glVertex3f(std::cos(ang) * (radius * 0.5f), std::sin(ang) * (radius * 0.5f), 0.1f);
        }
        glEnd();
    }
}

void GLCanvas::render_hud() {
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    gluOrtho2D(0, w(), 0, h());

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);

    char buf[256];
    const char* layer_names[] = {
        "Layer I: Harmonic Geometry & Fischer Decomposition",
        "Layer II: Projector Temple & K-Fixed Ray v_n \\otimes v_n^*",
        "Layer III: Differential & Schrodinger Wave Equations",
        "Layer IV: Jacobi Spectral City & Golub-Welsch Tower J_m",
        "Layer V: Two-Pole Boundary Layer Coordinates (z_+, z_-)",
        "Layer VI: Composite Matched Asymptotics F_comp",
        "Layer VII: Multi-Backend Arithmetic Execution Factory",
        "Layer VIII: Verification & Certification Chamber"
    };

    glColor3f(0.9f, 0.95f, 1.0f);
    std::snprintf(buf, sizeof(buf), "[%s]", layer_names[static_cast<int>(current_layer)]);
    gl_font(FL_HELVETICA_BOLD, 14);
    gl_draw(buf, 15, h() - 25);

    if (is_animating) {
        glColor3f(1.0f, 0.8f, 0.2f);
        std::snprintf(buf, sizeof(buf), "Morphing Transition T = %.2f -> [%s]",
                      transition_progress, layer_names[static_cast<int>(target_layer)]);
        gl_draw(buf, 15, h() - 45);
    }

    glColor3f(0.7f, 0.85f, 0.95f);
    std::snprintf(buf, sizeof(buf),
                  "d=%d (\\lambda=%.1f)  n=%d  \\theta=%.4f (x=%.4f)  N=%.1f  z_+=%.3f  z_-=%.3f  [Regime: %s]",
                  snapshot.d, snapshot.lambda, snapshot.n, snapshot.theta, snapshot.x,
                  snapshot.N, snapshot.z_plus, snapshot.z_minus, snapshot.regime_name.c_str());
    gl_font(FL_HELVETICA, 12);
    gl_draw(buf, 15, 20);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

    glEnable(GL_DEPTH_TEST);
}
