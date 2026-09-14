#include "GameUI.h"
#include <FL/Fl.H>
#include <FL/fl_ask.H>
#include <cstdio>
#include <sstream>
#include <iomanip>
#include <numbers>

GameUI::GameUI(int width, int height) {
    main_win = std::make_unique<Fl_Double_Window>(width, height, "EIGHTH LAYER: The Harmonic Representation Engine");
    main_win->size_range(900, 650);

    state.params.d = 3;
    state.params.n = 5;
    state.params.theta = 0.5;
    state.params.error_target = 1e-8;
    state.params.asymptotic_K = 1;
    state.params.jacobi_m = 10;
    state.current_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    state.target_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    state.backend = BackendType::FLOAT64;
    state.auto_router = false;
    state.transition = 0.0;

    int canvas_h = height - 280;
    gl_canvas = new GLCanvas(10, 10, width - 20, canvas_h, "GL Canvas");

    Fl_Group* ctrl_grp = new Fl_Group(10, canvas_h + 15, width - 20, 250);
    ctrl_grp->box(FL_FLAT_BOX);
    ctrl_grp->color(fl_rgb_color(20, 26, 36));

    int col1_x = 70;
    int col2_x = 390;
    int col3_x = 710;

    // Column 1: Core Parameters & Endpoint Sensitive Angle
    slider_d = new Fl_Value_Slider(col1_x, canvas_h + 20, 220, 22, "Dim (d):");
    slider_d->type(FL_HOR_SLIDER);
    slider_d->bounds(3, 20);
    slider_d->step(1);
    slider_d->value(state.params.d);
    slider_d->labelcolor(FL_WHITE);
    slider_d->textcolor(FL_WHITE);
    slider_d->callback(cb_slider_d, this);

    slider_n = new Fl_Value_Slider(col1_x, canvas_h + 47, 220, 22, "Deg (n):");
    slider_n->type(FL_HOR_SLIDER);
    slider_n->bounds(0, 500);
    slider_n->step(1);
    slider_n->value(state.params.n);
    slider_n->labelcolor(FL_WHITE);
    slider_n->textcolor(FL_WHITE);
    slider_n->callback(cb_slider_n, this);

    slider_theta = new Fl_Value_Slider(col1_x, canvas_h + 74, 220, 22, "Angle (\u03B8):");
    slider_theta->type(FL_HOR_SLIDER);
    slider_theta->bounds(0.0001, std::numbers::pi - 0.0001);
    slider_theta->step(0.0001);
    slider_theta->value(state.params.theta);
    slider_theta->labelcolor(FL_WHITE);
    slider_theta->textcolor(FL_WHITE);
    slider_theta->callback(cb_slider_theta, this);

    slider_asymptotic_k = new Fl_Value_Slider(col1_x, canvas_h + 101, 220, 22, "Order K:");
    slider_asymptotic_k->type(FL_HOR_SLIDER);
    slider_asymptotic_k->bounds(1, 5);
    slider_asymptotic_k->step(1);
    slider_asymptotic_k->value(state.params.asymptotic_K);
    slider_asymptotic_k->labelcolor(FL_WHITE);
    slider_asymptotic_k->textcolor(FL_WHITE);
    slider_asymptotic_k->callback(cb_slider_asymptotic_k, this);

    slider_jacobi_m = new Fl_Value_Slider(col1_x, canvas_h + 128, 220, 22, "Jacobi m:");
    slider_jacobi_m->type(FL_HOR_SLIDER);
    slider_jacobi_m->bounds(4, 30);
    slider_jacobi_m->step(1);
    slider_jacobi_m->value(state.params.jacobi_m);
    slider_jacobi_m->labelcolor(FL_WHITE);
    slider_jacobi_m->textcolor(FL_WHITE);
    slider_jacobi_m->callback(cb_slider_jacobi_m, this);

    btn_auto_router = new Fl_Button(col1_x, canvas_h + 160, 220, 25, "Router AI: MANUAL");
    btn_auto_router->color(fl_rgb_color(50, 70, 90));
    btn_auto_router->labelcolor(FL_WHITE);
    btn_auto_router->callback(cb_btn_auto_router, this);

    // Column 2: Layer, Backend & Morph Controls
    choice_layer = new Fl_Choice(col2_x, canvas_h + 20, 240, 22, "Layer:");
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

    slider_transition = new Fl_Value_Slider(col2_x, canvas_h + 47, 240, 22, "Morph T:");
    slider_transition->type(FL_HOR_SLIDER);
    slider_transition->bounds(0.0, 1.0);
    slider_transition->step(0.01);
    slider_transition->value(0.0);
    slider_transition->labelcolor(FL_WHITE);
    slider_transition->textcolor(FL_WHITE);
    slider_transition->callback(cb_slider_transition, this);

    btn_anim_morph = new Fl_Button(col2_x, canvas_h + 74, 115, 25, "Animate Morph");
    btn_anim_morph->color(fl_rgb_color(40, 120, 200));
    btn_anim_morph->labelcolor(FL_WHITE);
    btn_anim_morph->callback(cb_btn_anim_morph, this);

    btn_info = new Fl_Button(col2_x + 125, canvas_h + 74, 115, 25, "Layer Info");
    btn_info->color(fl_rgb_color(60, 160, 100));
    btn_info->labelcolor(FL_WHITE);
    btn_info->callback(cb_btn_info, this);

    choice_backend = new Fl_Choice(col2_x, canvas_h + 105, 240, 22, "Backend:");
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

    choice_error_target = new Fl_Choice(col2_x, canvas_h + 132, 240, 22, "\u03B5_target:");
    choice_error_target->add("1e-4 (Coarse)");
    choice_error_target->add("1e-8 (Balanced)");
    choice_error_target->add("1e-12 (Strict)");
    choice_error_target->add("1e-15 (Exact)");
    choice_error_target->value(1);
    choice_error_target->labelcolor(FL_WHITE);
    choice_error_target->callback(cb_choice_error_target, this);

    // Column 3: Structured 7-Section Telemetry Display
    text_buffer = new Fl_Text_Buffer();
    text_telemetry = new Fl_Text_Display(col3_x, canvas_h + 15, width - col3_x - 20, 225);
    text_telemetry->buffer(text_buffer);
    text_telemetry->color(fl_rgb_color(12, 16, 24));
    text_telemetry->textcolor(fl_rgb_color(120, 240, 160));
    text_telemetry->textfont(FL_COURIER);
    text_telemetry->textsize(11);

    ctrl_grp->end();

    main_win->end();
    main_win->resizable(gl_canvas);

    publish_snapshot();
}

GameUI::~GameUI() {
    Fl::remove_timeout(timer_update_cb, this);
    Fl::remove_timeout(timer_morph_cb, this);
    delete text_buffer;
}

void GameUI::show() {
    main_win->show();
}

void GameUI::mark_dirty_and_schedule() {
    is_dirty = true;
    Fl::remove_timeout(timer_update_cb, this);
    Fl::add_timeout(0.016, timer_update_cb, this);
}

void GameUI::timer_update_cb(void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui && ui->is_dirty) {
        ui->publish_snapshot();
        ui->is_dirty = false;
    }
}

void GameUI::timer_morph_cb(void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (!ui || !ui->is_morph_animating) return;

    ui->state.transition += 0.025;
    if (ui->state.transition >= 1.0) {
        ui->state.transition = 1.0;
        ui->state.current_layer = ui->state.target_layer;
        ui->is_morph_animating = false;
    } else {
        Fl::repeat_timeout(0.016, timer_morph_cb, ui);
    }
    ui->slider_transition->value(ui->state.transition);
    ui->publish_snapshot();
}

void GameUI::publish_snapshot() {
    RepresentationSnapshot snap = core.evaluate(state);
    gl_canvas->set_snapshot(snap);

    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2);

    oss << "========================================\n";
    oss << "1. STATE: d=" << snap.params.d << " (\u03BB=" << snap.lambda
        << ") | n=" << snap.params.n << " | N=" << snap.N << "\n";
    oss << "   \u03B8=" << std::fixed << std::setprecision(4) << snap.params.theta
        << " (x=" << snap.x << ") | \u03C6_n=" << snap.phi << "\n";
    oss << "----------------------------------------\n";
    oss << "2. REGIME & DOMAIN VALIDITY: " << snap.regime_name << "\n";
    oss << "   z_+=" << std::fixed << std::setprecision(3) << snap.z_plus
        << " [" << (snap.north_valid ? "VALID" : "OUT") << "] | "
        << "z_-=" << snap.z_minus << " [" << (snap.south_valid ? "VALID" : "OUT") << "]\n";
    oss << "----------------------------------------\n";
    oss << "3. REPRESENTATION & BACKEND:\n";
    oss << "   Layer: " << static_cast<int>(snap.current_layer)
        << " -> " << static_cast<int>(snap.target_layer)
        << " (T=" << std::fixed << std::setprecision(2) << snap.transition << ")\n";
    oss << "   Effective: " << snap.effective_layer_name << " | " << snap.effective_backend_name << "\n";
    oss << "----------------------------------------\n";
    oss << "4. NUMERICS & CONDITIONING:\n";
    oss << std::scientific << std::setprecision(2);
    oss << "   \u03BA=" << snap.conditioning << " | Fwd Err=" << snap.forward_error
        << " | B_K=" << snap.analytic_bound << "\n";
    oss << "----------------------------------------\n";
    oss << "5. STRUCTURAL RESIDUALS:\n";
    oss << "   R_rec=" << snap.r_rec << " | R_ODE=" << snap.r_ode << "\n";
    oss << "   R_Schr=" << snap.r_schr << " | R_J=" << snap.r_jacobi << "\n";
    oss << "   Backend Discrepancy = " << snap.backend_error << "\n";
    oss << "----------------------------------------\n";
    oss << "6. CERTIFICATION (5-TIER):\n";
    oss << "   ALG: " << (snap.cert.algebraic_exact ? "[PASS]" : "[FAIL]") << " - " << snap.cert.algebraic_reason << "\n";
    oss << "   ARITH: " << (snap.cert.arithmetic_exact ? "[PASS]" : "[FAIL]") << " - " << snap.cert.arithmetic_reason << "\n";
    oss << "   ANALYTIC: " << (snap.cert.analytic_certified ? "[PASS]" : "[FAIL]") << " - " << snap.cert.analytic_reason << "\n";
    oss << "   NUMERICAL: " << (snap.cert.numerical_approx ? "[PASS]" : "[FAIL]") << " - " << snap.cert.numerical_reason << "\n";
    oss << "----------------------------------------\n";
    oss << "7. ROUTER DECISION:\n";
    oss << "   " << snap.router_decision.reason << "\n";
    oss << "========================================";

    const std::string text = oss.str();
    text_buffer->text(text.c_str());

    if (state.auto_router && !is_updating_widgets) {
        is_updating_widgets = true;
        choice_layer->value(static_cast<int>(snap.effective_layer));
        choice_backend->value(static_cast<int>(snap.effective_backend));
        is_updating_widgets = false;
    }
}

void GameUI::cb_slider_d(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.params.d = static_cast<int>(ui->slider_d->value());
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_slider_n(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.params.n = static_cast<int>(ui->slider_n->value());
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_slider_theta(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.params.theta = ui->slider_theta->value();
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_slider_asymptotic_k(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.params.asymptotic_K = static_cast<int>(ui->slider_asymptotic_k->value());
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_slider_jacobi_m(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.params.jacobi_m = static_cast<int>(ui->slider_jacobi_m->value());
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_choice_error_target(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    int idx = ui->choice_error_target->value();
    double targets[] = { 1e-4, 1e-8, 1e-12, 1e-15 };
    ui->state.params.error_target = targets[std::clamp(idx, 0, 3)];
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_slider_transition(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.transition = ui->slider_transition->value();
    if (ui->state.transition >= 1.0) {
        ui->state.current_layer = ui->state.target_layer;
    }
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_choice_layer(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->is_updating_widgets) return;
    ui->state.target_layer = static_cast<LayerType>(ui->choice_layer->value());
    ui->state.transition = 0.0;
    ui->slider_transition->value(0.0);
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_choice_backend(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->is_updating_widgets) return;
    ui->state.backend = static_cast<BackendType>(ui->choice_backend->value());
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_btn_anim_morph(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.transition = 0.0;
    ui->is_morph_animating = true;
    Fl::remove_timeout(timer_morph_cb, ui);
    Fl::add_timeout(0.016, timer_morph_cb, ui);
}

void GameUI::cb_btn_auto_router(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->state.auto_router = !ui->state.auto_router;
    if (ui->state.auto_router) {
        ui->btn_auto_router->label("Router AI: AUTO");
        ui->btn_auto_router->color(fl_rgb_color(40, 160, 100));
    } else {
        ui->btn_auto_router->label("Router AI: MANUAL");
        ui->btn_auto_router->color(fl_rgb_color(50, 70, 90));
    }
    ui->mark_dirty_and_schedule();
}

void GameUI::cb_btn_info(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    int layer_idx = static_cast<int>(ui->state.current_layer);
    int target_idx = static_cast<int>(ui->state.target_layer);

    const char* info_texts[] = {
        "Layer I: 3D Projection/Slice of S^{d-1} Hyper-Sphere\n\nModels a 3D projected slice of the (d-1)-dimensional hyper-sphere S^{d-1} with degree-n zonal harmonic wave field.",
        "Layer II: Projector Temple & K-Fixed Ray\n\nVisualizes representation space V_n, K-fixed ray v_n, and rank-one projector P_{K,n} = v_n \u2297 v_n*.",
        "Layer III: Differential Waves & Schr\u00F6dinger Potential\n\nDisplays Gegenbauer Sturm-Liouville radial wave equation (1-x^2)\u03C6'' - (2\u03BB+1)x \u03C6' + n(n+2\u03BB)\u03C6 = 0 and quantum potential landscape V(\u03B8) = \u03BB(\u03BB-1)csc^2(\u03B8).",
        "Layer IV: Jacobi Spectral City & Golub-Welsch Tower\n\nTridiagonal matrix operator J_m tower with subdiagonal couplings \u03B1_k, eigenvalues x_k in (-1, 1), and quadrature weights w_k.",
        "Layer V: Two-Pole Boundary Layer Coordinates\n\nDisplays North pole Bessel scaling z_+ = N_n \u03B8, South pole Bessel scaling z_- = N_n (\u03C0-\u03B8), and interior WKB wave.",
        "Layer VI: Composite Matched Asymptotics\n\nUnifies endpoint Bessel expansions and interior WKB oscillations into two-overlap composite matched wave F_comp.",
        "Layer VII: Multi-Backend Arithmetic Factory\n\nEvaluates Gegenbauer wave across 7 execution backends (FLOAT32 to MODULAR_RNS), displaying precision mantissa lattices and noise particles.",
        "Layer VIII: Verification & Certification Chamber\n\nMulti-axis verification wheel classifying truth into ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, or NUMERICAL_APPROX."
    };

    if (ui->state.transition > 0.0 && layer_idx != target_idx) {
        fl_message("Morph Transition in Progress:\n[Current]: %s\n\n[Target]: %s", info_texts[layer_idx], info_texts[target_idx]);
    } else {
        fl_message("%s", info_texts[layer_idx]);
    }
}
