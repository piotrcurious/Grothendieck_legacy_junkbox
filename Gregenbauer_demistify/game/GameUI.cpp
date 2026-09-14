#include "GameUI.h"
#include <FL/Fl.H>
#include <FL/fl_ask.H>
#include <sstream>
#include <iomanip>
#include <numbers>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <array>

static constexpr std::array<double, 4> kErrorTargets = { 1e-4, 1e-8, 1e-12, 1e-15 };

static constexpr std::array<LayerInfo, kNumLayers> g_layer_infos{{
    {"Layer I: 3D Projection/Slice of S^{d-1} Hyper-Sphere",
     "Models a 3D projected slice of the (d-1)-dimensional hyper-sphere S^{d-1} with degree-n zonal harmonic wave field."},
    {"Layer II: Projector Temple & K-Fixed Ray",
     "Visualizes representation space V_n, K-fixed ray v_n, and rank-one projector P_{K,n} = v_n \u2297 v_n*."},
    {"Layer III: Differential Waves & Schr\u00F6dinger Potential",
     "Displays Gegenbauer Sturm-Liouville radial wave equation (1-x^2)\u03C6'' - (2\u03BB+1)x \u03C6' + n(n+2\u03BB)\u03C6 = 0 and quantum potential landscape V(\u03B8) = \u03BB(\u03BB-1)csc^2(\u03B8)."},
    {"Layer IV: Jacobi Spectral City & Golub-Welsch Tower",
     "Tridiagonal matrix operator J_m tower with subdiagonal couplings \u03B1_k, eigenvalues x_k in (-1, 1), and quadrature weights w_k."},
    {"Layer V: Two-Pole Boundary Layer Coordinates",
     "Displays North pole Bessel scaling z_+ = N_n \u03B8, South pole Bessel scaling z_- = N_n (\u03C0-\u03B8), and interior WKB wave."},
    {"Layer VI: Composite Matched Asymptotics",
     "Unifies endpoint Bessel expansions and interior WKB oscillations into two-overlap composite matched wave F_comp."},
    {"Layer VII: Multi-Backend Arithmetic Factory",
     "Evaluates Gegenbauer wave across 7 execution backends (FLOAT32 to MODULAR_RNS), displaying precision mantissa lattices and noise particles."},
    {"Layer VIII: Verification & Certification Chamber",
     "Multi-axis verification wheel classifying verification status into ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, or NUMERICAL_APPROX."}
}};

GameUI::GameUI(int width, int height)
    : is_dirty(false), is_morph_animating(false), is_updating_widgets(false) {
    main_win = std::make_unique<Fl_Double_Window>(width, height, "EIGHTH LAYER: The Harmonic Representation Engine");
    main_win->size_range(900, 720);

    state.params.d = 3;
    state.params.n = 5;
    state.params.theta = 0.5;
    state.params.error_target = kErrorTargets[1];
    state.params.error_target_idx = 1;
    state.params.asymptotic_K = 1;
    state.params.jacobi_m = 10;
    state.current_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    state.target_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    state.backend = BackendType::FLOAT64;
    state.auto_router = false;
    state.transition = 0.0;

    constexpr int margin = 10;
    constexpr int ctrl_height = 220;

    int canvas_h = std::max(height - ctrl_height - margin * 3, 200);
    gl_canvas = new GLCanvas(margin, margin, width - margin * 2, canvas_h, "GL Canvas");

    Fl_Group* ctrl_grp = new Fl_Group(margin, canvas_h + margin * 2, width - margin * 2, ctrl_height);
    ctrl_grp->box(FL_FLAT_BOX);
    ctrl_grp->color(fl_rgb_color(20, 26, 36));

    int col1_x = 70;
    int col2_x = 70 + static_cast<int>((width - 90) * 0.35);
    int col3_x = 70 + static_cast<int>((width - 90) * 0.68);

    // Column 1: Core Parameters
    slider_d = new Fl_Value_Slider(col1_x, canvas_h + 25, 220, 22, "Dim (d):");
    slider_d->type(FL_HOR_SLIDER);
    slider_d->bounds(3, 20);
    slider_d->step(1);
    slider_d->value(state.params.d);
    slider_d->labelcolor(FL_WHITE);
    slider_d->textcolor(FL_WHITE);
    slider_d->callback(cb_slider_d, this);

    slider_n = new Fl_Value_Slider(col1_x, canvas_h + 52, 220, 22, "Deg (n):");
    slider_n->type(FL_HOR_SLIDER);
    slider_n->bounds(0, 500);
    slider_n->step(1);
    slider_n->value(state.params.n);
    slider_n->labelcolor(FL_WHITE);
    slider_n->textcolor(FL_WHITE);
    slider_n->callback(cb_slider_n, this);

    slider_theta = new Fl_Value_Slider(col1_x, canvas_h + 79, 220, 22, "Angle (\u03B8):");
    slider_theta->type(FL_HOR_SLIDER);
    slider_theta->bounds(0.0001, std::numbers::pi - 0.0001);
    slider_theta->step(0.0001);
    slider_theta->value(state.params.theta);
    slider_theta->labelcolor(FL_WHITE);
    slider_theta->textcolor(FL_WHITE);
    slider_theta->callback(cb_slider_theta, this);

    slider_asymptotic_k = new Fl_Value_Slider(col1_x, canvas_h + 106, 220, 22, "Order K:");
    slider_asymptotic_k->type(FL_HOR_SLIDER);
    slider_asymptotic_k->bounds(1, 5);
    slider_asymptotic_k->step(1);
    slider_asymptotic_k->value(state.params.asymptotic_K);
    slider_asymptotic_k->labelcolor(FL_WHITE);
    slider_asymptotic_k->textcolor(FL_WHITE);
    slider_asymptotic_k->callback(cb_slider_asymptotic_k, this);

    slider_jacobi_m = new Fl_Value_Slider(col1_x, canvas_h + 133, 220, 22, "Jacobi m:");
    slider_jacobi_m->type(FL_HOR_SLIDER);
    slider_jacobi_m->bounds(4, 30);
    slider_jacobi_m->step(1);
    slider_jacobi_m->value(state.params.jacobi_m);
    slider_jacobi_m->labelcolor(FL_WHITE);
    slider_jacobi_m->textcolor(FL_WHITE);
    slider_jacobi_m->callback(cb_slider_jacobi_m, this);

    btn_auto_router = new Fl_Button(col1_x, canvas_h + 165, 220, 25, "Router AI: MANUAL");
    btn_auto_router->color(fl_rgb_color(50, 70, 90));
    btn_auto_router->labelcolor(FL_WHITE);
    btn_auto_router->callback(cb_btn_auto_router, this);

    // Column 2: Layer, Backend & Morph Controls
    choice_layer = new Fl_Choice(col2_x, canvas_h + 25, 240, 22, "Layer:");
    choice_layer->add("Layer I: Harmonic Geometry (S^{d-1})");
    choice_layer->add("Layer II: Projector Temple (v_n \u2297 v_n*)");
    choice_layer->add("Layer III: Differential & Schr\u00F6dinger Wave");
    choice_layer->add("Layer IV: Jacobi Spectral City (J_m)");
    choice_layer->add("Layer V: Two-Pole Boundary Coordinates");
    choice_layer->add("Layer VI: Composite Matched Asymptotics");
    choice_layer->add("Layer VII: Multi-Backend Arithmetic Factory");
    choice_layer->add("Layer VIII: Verification & Certification Chamber");
    choice_layer->value(0);
    choice_layer->labelcolor(FL_WHITE);
    choice_layer->callback(cb_choice_layer, this);

    slider_transition = new Fl_Value_Slider(col2_x, canvas_h + 52, 240, 22, "Morph T:");
    slider_transition->type(FL_HOR_SLIDER);
    slider_transition->bounds(0.0, 1.0);
    slider_transition->step(0.01);
    slider_transition->value(0.0);
    slider_transition->labelcolor(FL_WHITE);
    slider_transition->textcolor(FL_WHITE);
    slider_transition->callback(cb_slider_transition, this);

    btn_anim_morph = new Fl_Button(col2_x, canvas_h + 79, 115, 25, "Animate Morph");
    btn_anim_morph->color(fl_rgb_color(40, 120, 200));
    btn_anim_morph->labelcolor(FL_WHITE);
    btn_anim_morph->callback(cb_btn_anim_morph, this);

    btn_info = new Fl_Button(col2_x + 125, canvas_h + 79, 115, 25, "Layer Info");
    btn_info->color(fl_rgb_color(60, 160, 100));
    btn_info->labelcolor(FL_WHITE);
    btn_info->callback(cb_btn_info, this);

    choice_backend = new Fl_Choice(col2_x, canvas_h + 110, 240, 22, "Backend:");
    choice_backend->add("FLOAT32 (Single Precision)");
    choice_backend->add("FLOAT64 (Double Precision)");
    choice_backend->add("LONGDOUBLE (80-bit Extended)");
    choice_backend->add("Q16.16 (Fixed Point)");
    choice_backend->add("LNS (Logarithmic System)");
    choice_backend->add("EXACT_RATIONAL / SYMBOLIC");
    choice_backend->add("MODULAR_RNS (Residue CRT)");
    choice_backend->value(1);
    choice_backend->labelcolor(FL_WHITE);
    choice_backend->callback(cb_choice_backend, this);

    choice_error_target = new Fl_Choice(col2_x, canvas_h + 137, 240, 22, "\u03B5_target:");
    choice_error_target->add("1e-4 (Coarse)");
    choice_error_target->add("1e-8 (Balanced)");
    choice_error_target->add("1e-12 (Strict)");
    choice_error_target->add("1e-15 (Very Strict)");
    choice_error_target->value(1);
    choice_error_target->labelcolor(FL_WHITE);
    choice_error_target->callback(cb_choice_error_target, this);

    // Column 3: Telemetry Display
    text_buffer = new Fl_Text_Buffer();
    text_telemetry = new Fl_Text_Display(col3_x, canvas_h + 20, width - col3_x - margin * 2, 195);
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
    main_win.reset();
    delete text_buffer;
    text_buffer = nullptr;
}

void GameUI::show() {
    main_win->show();
}

void GameUI::normalize_state(GameState& st) {
    st.params.d = std::clamp(st.params.d, 3, 20);
    st.params.n = std::clamp(st.params.n, 0, 500);
    st.params.theta = std::clamp(st.params.theta, 0.0001, std::numbers::pi - 0.0001);
    st.params.asymptotic_K = std::clamp(st.params.asymptotic_K, 1, 5);
    st.params.jacobi_m = std::clamp(st.params.jacobi_m, 4, 30);

    st.params.error_target_idx = std::clamp(st.params.error_target_idx, 0, static_cast<int>(kErrorTargets.size()) - 1);
    st.params.error_target = kErrorTargets[st.params.error_target_idx];

    st.transition = std::clamp(st.transition, 0.0, 1.0);

    int l_curr = std::clamp(static_cast<int>(st.current_layer), 0, kNumLayers - 1);
    st.current_layer = static_cast<LayerType>(l_curr);

    int l_targ = std::clamp(static_cast<int>(st.target_layer), 0, kNumLayers - 1);
    st.target_layer = static_cast<LayerType>(l_targ);

    int b_val = std::clamp(static_cast<int>(st.backend), 0, kNumBackends - 1);
    st.backend = static_cast<BackendType>(b_val);
}

void GameUI::apply_state_change(const GameState& new_state) {
    cancel_layer_transition();
    current_valid = false;
    target_valid = false;
    state = new_state;
    normalize_state(state);
    mark_dirty_and_schedule();
}

void GameUI::set_transition(double t) {
    cancel_layer_transition();
    state.transition = std::clamp(t, 0.0, 1.0);
    if (state.transition >= 1.0) {
        commit_layer_transition();
    } else {
        mark_dirty_and_schedule();
    }
}

void GameUI::advance_transition(double dt) {
    state.transition += dt / kMorphDuration;
    state.transition = std::clamp(state.transition, 0.0, 1.0);

    if (state.transition >= 1.0) {
        commit_layer_transition();
    } else {
        render_snapshot = GegenbauerCore::morph_snapshots(current_snapshot, target_snapshot, state.transition);
        gl_canvas->set_snapshot(render_snapshot);
        if (slider_transition) slider_transition->value(state.transition);
    }
}

void GameUI::begin_layer_transition(LayerType target) {
    int target_idx = std::clamp(static_cast<int>(target), 0, kNumLayers - 1);
    target = static_cast<LayerType>(target_idx);

    if (target == state.current_layer) {
        cancel_layer_transition();
        state.target_layer = target;
        state.transition = 1.0;
        sync_widgets_from_state();
        mark_dirty_and_schedule();
        return;
    }
    cancel_layer_transition();
    state.target_layer = target;
    state.transition = 0.0;
    normalize_state(state);
    sync_widgets_from_state();

    // Cache fresh snapshots immediately before morph timer
    publish_snapshot();

    is_morph_animating = true;
    last_anim_time = std::chrono::steady_clock::now();
    Fl::add_timeout(kFrameInterval, timer_morph_cb, this);
}

void GameUI::cancel_layer_transition() {
    is_morph_animating = false;
    Fl::remove_timeout(timer_morph_cb, this);
}

void GameUI::commit_layer_transition() {
    cancel_layer_transition();
    if (!target_valid) {
        target_snapshot = core.evaluate(state, state.target_layer);
        target_key = SnapshotKey{state.params.d, state.params.n, state.params.theta,
                                 state.params.asymptotic_K, state.params.jacobi_m,
                                 state.params.error_target_idx, state.backend,
                                 state.target_layer, state.auto_router};
        target_valid = true;
    }
    current_snapshot = target_snapshot;
    current_key = target_key;
    current_valid = true;
    state.current_layer = state.target_layer;
    state.transition = 1.0;
    normalize_state(state);
    sync_widgets_from_state();
    mark_dirty_and_schedule();
}

void GameUI::sync_widgets_from_state() {
    if (is_updating_widgets) return;
    is_updating_widgets = true;

    slider_d->value(state.params.d);
    slider_n->value(state.params.n);
    slider_theta->value(state.params.theta);
    slider_asymptotic_k->value(state.params.asymptotic_K);
    slider_jacobi_m->value(state.params.jacobi_m);
    slider_transition->value(state.transition);

    choice_layer->value(static_cast<int>(state.target_layer));
    choice_backend->value(static_cast<int>(state.backend));
    choice_error_target->value(state.params.error_target_idx);

    if (state.auto_router) {
        btn_auto_router->label("Router AI: AUTO");
        btn_auto_router->color(fl_rgb_color(40, 160, 100));
    } else {
        btn_auto_router->label("Router AI: MANUAL");
        btn_auto_router->color(fl_rgb_color(50, 70, 90));
    }

    is_updating_widgets = false;
}

void GameUI::mark_dirty_and_schedule() {
    is_dirty = true;
    Fl::remove_timeout(timer_update_cb, this);
    Fl::add_timeout(kFrameInterval, timer_update_cb, this);
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

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - ui->last_anim_time).count();
    ui->last_anim_time = now;

    ui->advance_transition(dt);
    if (ui->is_morph_animating) {
        Fl::repeat_timeout(kFrameInterval, timer_morph_cb, ui);
    }
}

RepresentationSnapshot GameUI::get_or_evaluate_snapshot(const SnapshotKey& key,
                                                         const GameState& eval_state,
                                                         LayerType eval_layer) {
    if (current_valid && current_key == key) return current_snapshot;
    if (target_valid && target_key == key) return target_snapshot;

    RepresentationSnapshot evaluated = core.evaluate(eval_state, eval_layer);
    if (eval_layer == eval_state.current_layer) {
        current_snapshot = evaluated;
        current_key = key;
        current_valid = true;
    } else {
        target_snapshot = evaluated;
        target_key = key;
        target_valid = true;
    }
    return evaluated;
}

void GameUI::publish_snapshot() {
    // Two-snapshot evaluation model with explicit evaluation layers and cache helper
    SnapshotKey req_curr_key = SnapshotKey{state.params.d, state.params.n, state.params.theta,
                                           state.params.asymptotic_K, state.params.jacobi_m,
                                           state.params.error_target_idx, state.backend,
                                           state.current_layer, state.auto_router};

    current_snapshot = get_or_evaluate_snapshot(req_curr_key, state, state.current_layer);

    if (state.current_layer == state.target_layer) {
        target_snapshot = current_snapshot;
        target_key = current_key;
        target_valid = true;
    } else {
        SnapshotKey req_targ_key = SnapshotKey{state.params.d, state.params.n, state.params.theta,
                                               state.params.asymptotic_K, state.params.jacobi_m,
                                               state.params.error_target_idx, state.backend,
                                               state.target_layer, state.auto_router};

        target_snapshot = get_or_evaluate_snapshot(req_targ_key, state, state.target_layer);
    }

    render_snapshot = GegenbauerCore::morph_snapshots(current_snapshot, target_snapshot, state.transition);
    gl_canvas->set_snapshot(render_snapshot);

    // Authoritative telemetry HUD
    std::ostringstream oss;
    oss << std::scientific << std::setprecision(2);

    oss << "========================================\n";
    oss << "1. STATE: d=" << current_snapshot.params.d << " (\u03BB=" << current_snapshot.lambda
        << ") | n=" << current_snapshot.params.n << " | N=" << current_snapshot.N << "\n";
    oss << "   \u03B8=" << std::fixed << std::setprecision(4) << current_snapshot.params.theta
        << " (x=" << current_snapshot.x << ") | \u03C6_n=" << current_snapshot.phi << "\n";
    oss << "----------------------------------------\n";
    oss << "2. REGIME & DOMAIN VALIDITY: " << current_snapshot.regime_name << "\n";
    oss << "   z_+=" << std::fixed << std::setprecision(3) << current_snapshot.z_plus
        << " [" << (current_snapshot.north_valid ? "VALID" : "OUT") << "] | "
        << "z_-=" << current_snapshot.z_minus << " [" << (current_snapshot.south_valid ? "VALID" : "OUT") << "]\n";
    oss << "----------------------------------------\n";
    oss << "3. REPRESENTATION & BACKEND:\n";
    oss << "   Layer: " << static_cast<int>(render_snapshot.current_layer)
        << " -> " << static_cast<int>(render_snapshot.target_layer)
        << " (T=" << std::fixed << std::setprecision(2) << render_snapshot.transition << ")\n";
    oss << "   Effective: [CURR] " << current_snapshot.effective_layer_name
        << " -> [TARG] " << target_snapshot.effective_layer_name << "\n";
    oss << "   Backend: " << current_snapshot.effective_backend_name;
    if (current_snapshot.effective_backend_name != target_snapshot.effective_backend_name) {
        oss << " -> " << target_snapshot.effective_backend_name;
    }
    oss << "\n----------------------------------------\n";
    oss << "4. NUMERICS & CONDITIONING:\n";
    oss << std::scientific << std::setprecision(2);
    oss << "   \u03BA=" << current_snapshot.conditioning << " | Fwd Err=" << current_snapshot.forward_error
        << " | B_K=" << current_snapshot.analytic_bound << "\n";
    oss << "----------------------------------------\n";
    oss << "5. STRUCTURAL RESIDUALS:\n";
    oss << "   R_rec=" << current_snapshot.r_rec << " | R_ODE=" << current_snapshot.r_ode << "\n";
    oss << "   R_Schr=" << current_snapshot.r_schr << " | R_J=" << current_snapshot.r_jacobi << "\n";
    oss << "   Backend Discrepancy = " << current_snapshot.backend_error << "\n";
    oss << "----------------------------------------\n";
    oss << "6. CERTIFICATION (4-AXIS) [Diagnostics: CURRENT Representation]:\n";
    oss << "   ALG: " << (current_snapshot.cert.algebraic_exact ? "[PASS]" : "[FAIL]") << " - " << current_snapshot.cert.algebraic_reason << "\n";
    oss << "   ARITH: " << (current_snapshot.cert.arithmetic_exact ? "[PASS]" : "[FAIL]") << " - " << current_snapshot.cert.arithmetic_reason << "\n";
    oss << "   ANALYTIC: " << (current_snapshot.cert.analytic_certified ? "[PASS]" : "[FAIL]") << " - " << current_snapshot.cert.analytic_reason << "\n";
    oss << "   NUMERICAL: " << (current_snapshot.cert.numerical_approx ? "[PASS]" : "[FAIL]") << " - " << current_snapshot.cert.numerical_reason << "\n";
    oss << "----------------------------------------\n";
    oss << "7. ROUTER DECISION:\n";
    oss << "   Requested: " << GegenbauerCore::get_layer_name(state.target_layer)
        << " | Effective: " << current_snapshot.effective_layer_name << "\n";
    oss << "   " << current_snapshot.router_decision.reason << "\n";
    oss << "========================================";

    const std::string text = oss.str();
    text_buffer->text(text.c_str());

    sync_widgets_from_state();
}

void GameUI::cb_slider_d(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.params.d = static_cast<int>(ui->slider_d->value());
    ui->apply_state_change(next_state);
}

void GameUI::cb_slider_n(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.params.n = static_cast<int>(ui->slider_n->value());
    ui->apply_state_change(next_state);
}

void GameUI::cb_slider_theta(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.params.theta = ui->slider_theta->value();
    ui->apply_state_change(next_state);
}

void GameUI::cb_slider_asymptotic_k(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.params.asymptotic_K = static_cast<int>(ui->slider_asymptotic_k->value());
    ui->apply_state_change(next_state);
}

void GameUI::cb_slider_jacobi_m(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.params.jacobi_m = static_cast<int>(ui->slider_jacobi_m->value());
    ui->apply_state_change(next_state);
}

void GameUI::cb_choice_error_target(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->is_updating_widgets) return;
    int idx = std::clamp(ui->choice_error_target->value(), 0, static_cast<int>(kErrorTargets.size()) - 1);
    GameState next_state = ui->state;
    next_state.params.error_target_idx = idx;
    next_state.params.error_target = kErrorTargets[idx];
    ui->apply_state_change(next_state);
}

void GameUI::cb_slider_transition(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->set_transition(ui->slider_transition->value());
}

void GameUI::cb_choice_layer(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->is_updating_widgets) return;
    int val = std::clamp(ui->choice_layer->value(), 0, kNumLayers - 1);
    LayerType target = static_cast<LayerType>(val);
    ui->begin_layer_transition(target);
}

void GameUI::cb_choice_backend(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->is_updating_widgets) return;
    int val = std::clamp(ui->choice_backend->value(), 0, kNumBackends - 1);
    GameState next_state = ui->state;
    next_state.backend = static_cast<BackendType>(val);
    ui->apply_state_change(next_state);
}

void GameUI::cb_btn_anim_morph(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    if (ui->state.current_layer == ui->state.target_layer && ui->state.transition >= 1.0) {
        return;
    }
    ui->begin_layer_transition(ui->state.target_layer);
}

void GameUI::cb_btn_auto_router(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    GameState next_state = ui->state;
    next_state.auto_router = !next_state.auto_router;
    ui->apply_state_change(next_state);
}

void GameUI::cb_btn_info(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    int layer_idx = std::clamp(static_cast<int>(ui->state.current_layer), 0, kNumLayers - 1);
    int target_idx = std::clamp(static_cast<int>(ui->state.target_layer), 0, kNumLayers - 1);

    if (ui->state.transition > 0.0 && layer_idx != target_idx) {
        fl_message("Morph Transition in Progress:\n[Current]: %s\n%s\n\n[Target]: %s\n%s",
                   g_layer_infos[layer_idx].title, g_layer_infos[layer_idx].description,
                   g_layer_infos[target_idx].title, g_layer_infos[target_idx].description);
    } else {
        fl_message("%s\n\n%s", g_layer_infos[layer_idx].title, g_layer_infos[layer_idx].description);
    }
}
