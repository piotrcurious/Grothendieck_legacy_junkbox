#include "GameUI.h"
#include <FL/Fl.H>
#include <FL/fl_ask.H>
#include <cstdio>
#include <sstream>

GameUI::GameUI(int width, int height) : core(3, 5, 0.5) {
    main_win = std::make_unique<Fl_Double_Window>(width, height, "EIGHTH LAYER: The Harmonic Representation Engine");

    int canvas_h = height - 260;
    gl_canvas = new GLCanvas(10, 10, width - 20, canvas_h, "GL Canvas");
    gl_canvas->set_core(&core);

    // Lower Control Panel Group
    Fl_Group* ctrl_grp = new Fl_Group(10, canvas_h + 15, width - 20, 230);
    ctrl_grp->box(FL_FLAT_BOX);
    ctrl_grp->color(fl_rgb_color(20, 26, 36));

    int col1_x = 70;
    int col2_x = 390;
    int col3_x = 710;

    // Column 1: Core Parameters (d, n, theta)
    slider_d = new Fl_Value_Slider(col1_x, canvas_h + 25, 220, 25, "Dim (d):");
    slider_d->type(FL_HOR_SLIDER);
    slider_d->bounds(3, 20);
    slider_d->step(1);
    slider_d->value(core.d);
    slider_d->labelcolor(FL_WHITE);
    slider_d->textcolor(FL_WHITE);
    slider_d->callback(cb_slider_d, this);

    slider_n = new Fl_Value_Slider(col1_x, canvas_h + 60, 220, 25, "Deg (n):");
    slider_n->type(FL_HOR_SLIDER);
    slider_n->bounds(0, 500);
    slider_n->step(1);
    slider_n->value(core.n);
    slider_n->labelcolor(FL_WHITE);
    slider_n->textcolor(FL_WHITE);
    slider_n->callback(cb_slider_n, this);

    slider_theta = new Fl_Value_Slider(col1_x, canvas_h + 95, 220, 25, "Angle (\u03B8):");
    slider_theta->type(FL_HOR_SLIDER);
    slider_theta->bounds(0.001, 3.1415);
    slider_theta->step(0.001);
    slider_theta->value(core.theta);
    slider_theta->labelcolor(FL_WHITE);
    slider_theta->textcolor(FL_WHITE);
    slider_theta->callback(cb_slider_theta, this);

    btn_auto_router = new Fl_Button(col1_x, canvas_h + 135, 220, 25, "Router AI: MANUAL");
    btn_auto_router->color(fl_rgb_color(50, 70, 90));
    btn_auto_router->labelcolor(FL_WHITE);
    btn_auto_router->callback(cb_btn_auto_router, this);

    // Column 2: Representation Layer & Morphing Controls
    choice_layer = new Fl_Choice(col2_x, canvas_h + 25, 240, 25, "Layer:");
    choice_layer->add("Layer I: Harmonic Geometry (S^{d-1})");
    choice_layer->add("Layer II: Projector Temple (v_n \u2297 v_n*)");
    choice_layer->add("Layer III: Differential & Schr\u00F6dinger Wave");
    choice_layer->add("Layer IV: Jacobi Spectral City (J_m)");
    choice_layer->add("Layer V: Two-Pole Boundary Coordinates");
    choice_layer->add("Layer VI: Composite Matched Asymptotics");
    choice_layer->add("Layer VII: Multi-Backend Arithmetic Factory");
    choice_layer->add("Layer VIII: Certification Chamber");
    choice_layer->value(0);
    choice_layer->labelcolor(FL_WHITE);
    choice_layer->callback(cb_choice_layer, this);

    slider_transition = new Fl_Value_Slider(col2_x, canvas_h + 60, 240, 25, "Morph T:");
    slider_transition->type(FL_HOR_SLIDER);
    slider_transition->bounds(0.0, 1.0);
    slider_transition->step(0.01);
    slider_transition->value(0.0);
    slider_transition->labelcolor(FL_WHITE);
    slider_transition->textcolor(FL_WHITE);
    slider_transition->callback(cb_slider_transition, this);

    btn_anim_morph = new Fl_Button(col2_x, canvas_h + 95, 115, 25, "Animate Morph");
    btn_anim_morph->color(fl_rgb_color(40, 120, 200));
    btn_anim_morph->labelcolor(FL_WHITE);
    btn_anim_morph->callback(cb_btn_anim_morph, this);

    btn_info = new Fl_Button(col2_x + 125, canvas_h + 95, 115, 25, "Layer Info");
    btn_info->color(fl_rgb_color(60, 160, 100));
    btn_info->labelcolor(FL_WHITE);
    btn_info->callback(cb_btn_info, this);

    choice_backend = new Fl_Choice(col2_x, canvas_h + 135, 240, 25, "Backend:");
    choice_backend->add("FLOAT32 (Single Precision)");
    choice_backend->add("FLOAT64 (Double Precision)");
    choice_backend->add("LONGDOUBLE (80-bit Extended)");
    choice_backend->add("Q16.16 (Fixed Point)");
    choice_backend->add("LNS (Logarithmic System)");
    choice_backend->add("EXACT_RATIONAL (\u221A Symbolic)");
    choice_backend->add("MODULAR_RNS (Residue CRT)");
    choice_backend->value(1);
    choice_backend->labelcolor(FL_WHITE);
    choice_backend->callback(cb_choice_backend, this);

    // Column 3: Rich Mathematical Telemetry Display
    text_buffer = new Fl_Text_Buffer();
    text_telemetry = new Fl_Text_Display(col3_x, canvas_h + 20, width - col3_x - 20, 180);
    text_telemetry->buffer(text_buffer);
    text_telemetry->color(fl_rgb_color(12, 16, 24));
    text_telemetry->textcolor(fl_rgb_color(120, 240, 160));
    text_telemetry->textfont(FL_COURIER);
    text_telemetry->textsize(12);

    ctrl_grp->end();

    main_win->end();
    main_win->resizable(gl_canvas);

    on_state_changed();
}

GameUI::~GameUI() {
    delete text_buffer;
}

void GameUI::show() {
    main_win->show();
}

void GameUI::on_state_changed() {
    gl_canvas->update_snapshot();
    const RepresentationSnapshot& snap = gl_canvas->snapshot;

    std::ostringstream oss;
    oss << "=== MATHEMATICAL TELEMETRY HUD ===\n";
    oss << "[Regime]: " << snap.regime_name << "\n";
    oss << "  d=" << snap.d << " (\u03BB=" << snap.lambda << ") | n=" << snap.n
        << " | N=" << snap.N << "\n";
    oss << "  z_+ = " << snap.z_plus << " | z_- = " << snap.z_minus << "\n";
    oss << "[Conditioning & Bounds]:\n";
    oss << "  \u03BA=" << snap.conditioning << " | Forward Err=" << snap.forward_error
        << " | B_K=" << snap.analytic_bound << "\n";
    oss << "[Residuals]:\n";
    oss << "  R_rec=" << snap.r_rec << " | R_ODE=" << snap.r_ode
        << " | R_Schr=" << snap.r_schr << "\n";
    oss << "  Backend Discrepancy = " << snap.backend_error << "\n";
    oss << "[Representation Router AI]:\n";
    oss << "  " << snap.router_reason << "\n";
    oss << "[Layer VIII Certification]:\n";
    oss << "  ALG: " << (snap.cert.algebraic_exact ? "[\u2713]" : "[\u2717]")
        << "  ARITH: " << (snap.cert.arithmetic_exact ? "[\u2713]" : "[\u2717]")
        << "  ANALYTIC: " << (snap.cert.analytic_certified ? "[\u2713]" : "[\u2717]")
        << "  NUMERICAL: " << (snap.cert.numerical_valid ? "[\u2713]" : "[\u2717]");

    text_buffer->text(oss.str().c_str());

    if (gl_canvas->auto_router) {
        choice_layer->value(static_cast<int>(snap.layer));
        choice_backend->value(static_cast<int>(snap.backend));
    }

    gl_canvas->redraw();
}

void GameUI::cb_slider_d(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(static_cast<int>(ui->slider_d->value()), ui->core.n, ui->core.theta);
    ui->on_state_changed();
}

void GameUI::cb_slider_n(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(ui->core.d, static_cast<int>(ui->slider_n->value()), ui->core.theta);
    ui->on_state_changed();
}

void GameUI::cb_slider_theta(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(ui->core.d, ui->core.n, ui->slider_theta->value());
    ui->on_state_changed();
}

void GameUI::cb_slider_transition(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->transition_progress = ui->slider_transition->value();
    ui->gl_canvas->redraw();
}

void GameUI::cb_choice_layer(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    LayerType target = static_cast<LayerType>(ui->choice_layer->value());
    ui->gl_canvas->set_target_layer(target);
    ui->slider_transition->value(0.0);
    ui->on_state_changed();
}

void GameUI::cb_choice_backend(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->active_backend = static_cast<BackendType>(ui->choice_backend->value());
    ui->on_state_changed();
}

void GameUI::cb_btn_anim_morph(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->trigger_transition_animation();
}

void GameUI::cb_btn_auto_router(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->auto_router = !ui->gl_canvas->auto_router;
    if (ui->gl_canvas->auto_router) {
        ui->btn_auto_router->label("Router AI: AUTO");
        ui->btn_auto_router->color(fl_rgb_color(40, 160, 100));
    } else {
        ui->btn_auto_router->label("Router AI: MANUAL");
        ui->btn_auto_router->color(fl_rgb_color(50, 70, 90));
    }
    ui->on_state_changed();
}

void GameUI::cb_btn_info(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    int layer_idx = static_cast<int>(ui->gl_canvas->current_layer);
    const char* info_texts[] = {
        "Layer I: Harmonic Geometry & Fischer Decomposition\n\nModels S^{d-1} spherical manifold with degree-n zonal harmonic amplitude field, nodal hypersurfaces, and metric curvature scaling with dimension d.",
        "Layer II: Projector Temple & K-Fixed Ray\n\nVisualizes dimensionality reduction from high-dimensional representation space V_n onto the 1D K-fixed axis v_n via rank-one projector P_{K,n} = v_n \u2297 v_n*.",
        "Layer III: Differential Waves & Schr\u00F6dinger Potential\n\nDisplays Gegenbauer Sturm-Liouville radial wave equation (1-x^2)\u03C6'' - (2\u03BB+1)x \u03C6' + n(n+2\u03BB)\u03C6 = 0 and quantum potential landscape V(\u03B8) = \u03BB(\u03BB-1)csc^2(\u03B8).",
        "Layer IV: Jacobi Spectral City & Golub-Welsch Tower\n\nTridiagonal matrix operator J_m tower with subdiagonal couplings \u03B1_k and Golub-Welsch spectral lines x_k in (-1, 1).",
        "Layer V: Two-Pole Boundary Layer Coordinates\n\nDisplays North pole Bessel scaling z_+ = N_n \u03B8, South pole Bessel scaling z_- = N_n (\u03C0-\u03B8), and interior WKB wave.",
        "Layer VI: Composite Matched Asymptotics\n\nUnifies endpoint Bessel expansions and interior WKB oscillations into the two-overlap composite matched wave F_comp.",
        "Layer VII: Multi-Backend Arithmetic Factory\n\nEvaluates Gegenbauer wave across 7 execution backends (FLOAT32 to MODULAR_RNS), displaying precision mantissa lattices and noise particles.",
        "Layer VIII: Verification & Certification Chamber\n\nMulti-axis verification wheel classifying truth into ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, or NUMERICAL_VALID."
    };
    fl_message("%s", info_texts[layer_idx]);
}
