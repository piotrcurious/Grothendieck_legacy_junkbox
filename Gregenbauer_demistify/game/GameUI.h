#ifndef GAME_UI_H
#define GAME_UI_H

#include <FL/Fl_Double_Window.H>
#include <FL/Fl_Value_Slider.H>
#include <FL/Fl_Button.H>
#include <FL/Fl_Choice.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Group.H>
#include <FL/Fl_Text_Display.H>
#include <FL/Fl_Text_Buffer.H>
#include <memory>
#include "GegenbauerCore.h"
#include "GLCanvas.h"

class GameUI {
public:
    GegenbauerCore core;
    GameState state;

    std::unique_ptr<Fl_Double_Window> main_win;
    GLCanvas* gl_canvas = nullptr;

    // Controls
    Fl_Value_Slider* slider_d = nullptr;
    Fl_Value_Slider* slider_n = nullptr;
    Fl_Value_Slider* slider_theta = nullptr;
    Fl_Value_Slider* slider_transition = nullptr;

    Fl_Choice* choice_layer = nullptr;
    Fl_Choice* choice_backend = nullptr;

    Fl_Button* btn_anim_morph = nullptr;
    Fl_Button* btn_auto_router = nullptr;
    Fl_Button* btn_info = nullptr;

    // Structured Telemetry Display
    Fl_Text_Display* text_telemetry = nullptr;
    Fl_Text_Buffer* text_buffer = nullptr;

    // Debouncing state
    bool is_dirty = false;
    bool is_morph_animating = false;

    GameUI(int width = 1100, int height = 820);
    ~GameUI();

    void show();
    void mark_dirty_and_schedule();
    void publish_snapshot();

    static void timer_update_cb(void* userdata);
    static void timer_morph_cb(void* userdata);

    // Callbacks
    static void cb_slider_d(Fl_Widget* w, void* userdata);
    static void cb_slider_n(Fl_Widget* w, void* userdata);
    static void cb_slider_theta(Fl_Widget* w, void* userdata);
    static void cb_slider_transition(Fl_Widget* w, void* userdata);
    static void cb_choice_layer(Fl_Widget* w, void* userdata);
    static void cb_choice_backend(Fl_Widget* w, void* userdata);
    static void cb_btn_anim_morph(Fl_Widget* w, void* userdata);
    static void cb_btn_auto_router(Fl_Widget* w, void* userdata);
    static void cb_btn_info(Fl_Widget* w, void* userdata);
};

#endif // GAME_UI_H
