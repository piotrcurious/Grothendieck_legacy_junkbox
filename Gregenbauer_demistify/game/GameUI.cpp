#include "GameUI.h"
#include <FL/Fl.H>
#include <FL/fl_ask.H>
#include <cstdio>
#include <sstream>

GameUI::GameUI(int width, int height) : core(3, 5, 0.5) {
    main_win = new Fl_Double_Window(width, height, "EIGHTH LAYER: The Harmonic Representation Engine");

    // 1. OpenGL Rendering Canvas (Top area)
    int canvas_h = height - 220;
    gl_canvas = new GLCanvas(10, 10, width - 20, canvas_h, "GL Canvas");
    gl_canvas->set_core(&core);

    // 2. Lower Control Group Panel
    Fl_Group* ctrl_grp = new Fl_Group(10, canvas_h + 20, width - 20, 190);
    ctrl_grp->box(FL_FLAT_BOX);
    ctrl_grp->color(fl_rgb_color(22, 28, 38));

    // Sliders Column 1 (Parameters: d, n, theta)
    slider_d = new Fl_Value_Slider(70, canvas_h + 30, 220, 25, "Dim (d):");
    slider_d->type(FL_HOR_SLIDER);
    slider_d->bounds(3, 20);
    slider_d->step(1);
    slider_d->value(core.d);
    slider_d->labelcolor(FL_WHITE);
    slider_d->textcolor(FL_WHITE);
    slider_d->callback(cb_slider_d, this);

    slider_n = new Fl_Value_Slider(70, canvas_h + 65, 220, 25, "Deg (n):");
    slider_n->type(FL_HOR_SLIDER);
    slider_n->bounds(0, 500);
    slider_n->step(1);
    slider_n->value(core.n);
    slider_n->labelcolor(FL_WHITE);
    slider_n->textcolor(FL_WHITE);
    slider_n->callback(cb_slider_n, this);

    slider_theta = new Fl_Value_Slider(70, canvas_h + 100, 220, 25, "Angle (th):");
    slider_theta->type(FL_HOR_SLIDER);
    slider_theta->bounds(0.001, 3.1415);
    slider_theta->step(0.001);
    slider_theta->value(core.theta);
    slider_theta->labelcolor(FL_WHITE);
    slider_theta->textcolor(FL_WHITE);
    slider_theta->callback(cb_slider_theta, this);

    // Column 2 (Layer selection & Transition Morphing)
    choice_layer = new Fl_Choice(400, canvas_h + 30, 240, 25, "Layer:");
    choice_layer->add("Layer I: Harmonic Geometry (S^{d-1})");
    choice_layer->add("Layer II: Projector Temple (v_n (x) v_n^*)");
    choice_layer->add("Layer III: Differential & Schrodinger Wave");
    choice_layer->add("Layer IV: Jacobi Spectral City (J_m)");
    choice_layer->add("Layer V: Two-Pole Boundary Coordinates");
    choice_layer->add("Layer VI: Composite Matched Asymptotics");
    choice_layer->add("Layer VII: Multi-Backend Arithmetic Factory");
    choice_layer->add("Layer VIII: Certification Chamber");
    choice_layer->value(0);
    choice_layer->labelcolor(FL_WHITE);
    choice_layer->callback(cb_choice_layer, this);

    slider_transition = new Fl_Value_Slider(400, canvas_h + 65, 240, 25, "Morph T:");
    slider_transition->type(FL_HOR_SLIDER);
    slider_transition->bounds(0.0, 1.0);
    slider_transition->step(0.01);
    slider_transition->value(0.0);
    slider_transition->labelcolor(FL_WHITE);
    slider_transition->textcolor(FL_WHITE);
    slider_transition->callback(cb_slider_transition, this);

    btn_anim_morph = new Fl_Button(400, canvas_h + 100, 115, 25, "Animate Morph");
    btn_anim_morph->color(fl_rgb_color(40, 120, 200));
    btn_anim_morph->labelcolor(FL_WHITE);
    btn_anim_morph->callback(cb_btn_anim_morph, this);

    btn_info = new Fl_Button(525, canvas_h + 100, 115, 25, "Layer Info");
    btn_info->color(fl_rgb_color(60, 160, 100));
    btn_info->labelcolor(FL_WHITE);
    btn_info->callback(cb_btn_info, this);

    // Column 3 (Backend & Residual HUD)
    choice_backend = new Fl_Choice(740, canvas_h + 30, 220, 25, "Backend:");
    choice_backend->add("FLOAT32 (Single Precision)");
    choice_backend->add("FLOAT64 (Double Precision)");
    choice_backend->add("LONGDOUBLE (80-bit Extended)");
    choice_backend->add("Q16.16 (Fixed Point)");
    choice_backend->add("LNS (Logarithmic Number System)");
    choice_backend->add("EXACT_RATIONAL (Q[lambda, x])");
    choice_backend->add("MODULAR_RNS (Residue System)");
    choice_backend->value(1);
    choice_backend->labelcolor(FL_WHITE);
    choice_backend->callback(cb_choice_backend, this);

    out_residuals = new Fl_Output(740, canvas_h + 65, 330, 60, "Telemetry:");
    out_residuals->labelcolor(FL_WHITE);
    out_residuals->color(fl_rgb_color(15, 20, 30));
    out_residuals->textcolor(fl_rgb_color(100, 230, 150));

    ctrl_grp->end();

    main_win->end();
    main_win->resizable(gl_canvas);

    update_ui_from_core();
}

GameUI::~GameUI() {
    delete main_win;
}

void GameUI::show() {
    main_win->show();
}

void GameUI::update_ui_from_core() {
    ResidualState res = core.compute_residuals(gl_canvas->active_backend);

    char buf[512];
    const char* truth_str = "NUMERICAL_APPROX";
    if (res.truth_class == TruthClass::ALGEBRAIC_EXACT) truth_str = "ALGEBRAIC_EXACT";
    else if (res.truth_class == TruthClass::ARITHMETIC_EXACT) truth_str = "ARITHMETIC_EXACT";
    else if (res.truth_class == TruthClass::ANALYTIC_CERTIFIED) truth_str = "ANALYTIC_CERTIFIED";

    std::snprintf(buf, sizeof(buf),
                  "Truth: %s [%s]\nR_rec: %.2e | R_ODE: %.2e | R_Schr: %.2e\nR_Jacobi: %.2e | Backend Error: %.2e",
                  truth_str, res.is_certified ? "CERTIFIED" : "UNCERTIFIED",
                  res.r_rec, res.r_ode, res.r_schr, res.r_jacobi, res.e_backend);

    out_residuals->value(buf);
    gl_canvas->redraw();
}

void GameUI::cb_slider_d(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(static_cast<int>(ui->slider_d->value()), ui->core.n, ui->core.theta);
    ui->update_ui_from_core();
}

void GameUI::cb_slider_n(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(ui->core.d, static_cast<int>(ui->slider_n->value()), ui->core.theta);
    ui->update_ui_from_core();
}

void GameUI::cb_slider_theta(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->core.update_parameters(ui->core.d, ui->core.n, ui->slider_theta->value());
    ui->update_ui_from_core();
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
}

void GameUI::cb_choice_backend(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->active_backend = static_cast<BackendType>(ui->choice_backend->value());
    ui->update_ui_from_core();
}

void GameUI::cb_btn_anim_morph(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    ui->gl_canvas->trigger_transition_animation();
}

void GameUI::cb_btn_info(Fl_Widget* w, void* userdata) {
    GameUI* ui = static_cast<GameUI*>(userdata);
    int layer_idx = static_cast<int>(ui->gl_canvas->current_layer);
    const char* info_texts[] = {
        "Layer I: Harmonic Geometry & Fischer Decomposition\n\nModels S^{d-1} spherical manifold with degree-n zonal harmonic amplitude field, nodal hypersurfaces, and metric curvature scaling with dimension d.",
        "Layer II: Projector Temple & K-Fixed Ray\n\nVisualizes dimensionality reduction from high-dimensional representation space V_n onto the 1D K-fixed axis v_n via rank-one projector P_{K,n} = v_n (x) v_n^*.",
        "Layer III: Differential Waves & Schrodinger Potential\n\nDisplays Gegenbauer Sturm-Liouville radial wave equation (1-x^2)phi'' - (2lam+1)x phi' + n(n+2lam)phi = 0 and quantum potential landscape V(theta) = lam(lam-1)csc^2(theta).",
        "Layer IV: Jacobi Spectral City & Golub-Welsch Tower\n\nTridiagonal matrix operator J_m tower with subdiagonal couplings alpha_k and Golub-Welsch spectral lines x_k in (-1, 1).",
        "Layer V: Two-Pole Boundary Layer Coordinates\n\nDisplays North pole Bessel scaling z_+ = N_n theta, South pole Bessel scaling z_- = N_n (pi-theta), and interior WKB wave.",
        "Layer VI: Composite Matched Asymptotics\n\nUnifies endpoint Bessel expansions and interior WKB oscillations into the two-overlap composite matched wave F_comp.",
        "Layer VII: Multi-Backend Arithmetic Factory\n\nEvaluates Gegenbauer wave across 7 execution backends (FLOAT32 to MODULAR_RNS), displaying precision mantissa lattices and noise particles.",
        "Layer VIII: Verification & Certification Chamber\n\nMulti-axis verification wheel classifying truth into ALGEBRAIC_EXACT, ARITHMETIC_EXACT, ANALYTIC_CERTIFIED, or NUMERICAL_APPROX."
    };
    fl_message("%s", info_texts[layer_idx]);
}
