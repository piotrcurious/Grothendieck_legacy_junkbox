#ifndef GAME_UI_H
#define GAME_UI_H

#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Output.H>
#include "GegenbauerCore.h"
#include "GLCanvas.h"

class GameUI {
public:
    GegenbauerCore core;

    Fl_Double_Window* main_win = nullptr;
    GLCanvas* gl_canvas = nullptr;

    // Controls
    Fl_Value_Slider* slider_d = nullptr;
    Fl_Value_Slider* slider_n = nullptr;
    Fl_Value_Slider* slider_theta = nullptr;
    Fl_Value_Slider* slider_transition = nullptr;

    Fl_Choice* choice_layer = nullptr;
    Fl_Choice* choice_backend = nullptr;

    Fl_Button* btn_anim_morph = nullptr;
    Fl_Button* btn_info = nullptr;

    // Status boxes
    Fl_Box* box_status = nullptr;
    Fl_Output* out_residuals = nullptr;

    GameUI(int width = 1100, int height = 800);
    ~GameUI();

    void show();
    void update_ui_from_core();

    // Callbacks
    static void cb_slider_d(Fl_Widget* w, void* userdata);
    static void cb_slider_n(Fl_Widget* w, void* userdata);
    static void cb_slider_theta(Fl_Widget* w, void* userdata);
    static void cb_slider_transition(Fl_Widget* w, void* userdata);
    static void cb_choice_layer(Fl_Widget* w, void* userdata);
    static void cb_choice_backend(Fl_Widget* w, void* userdata);
    static void cb_btn_anim_morph(Fl_Widget* w, void* userdata);
    static void cb_btn_info(Fl_Widget* w, void* userdata);
};

#endif // GAME_UI_H
