I'll extend the demo with interactive controls to adjust parameters in real-time and see their effects.

// Previous includes and forward declarations remain the same...

class QuantizationDemo {
private:
    // Previous member variables remain...
    
    struct DemoParameters {
        float rotationSpeed = 1.0f;
        float errorWeight = 1.0f;
        int numStars = 200;
        float starSpeed = 1.0f;
        bool showTrails = false;
        float kernelSigma = 0.1f;
        bool pauseRotation = false;
        float manualRotation = 0.0f;
    } params;
    
    struct UIState {
        bool showControls = true;
        bool isDragging = false;
        int dragStartX = 0;
        int dragStartY = 0;
        float dragStartValue = 0.0f;
        std::string activeTooltip;
    } ui;
    
    struct Trail {
        std::vector<SDL_Point> points;
        int maxPoints = 50;
    };
    std::vector<Trail> starTrails;
    
    // Slider control
    struct Slider {
        float& value;
        float min, max;
        std::string label;
        std::string tooltip;
        int x, y, width;
        
        Slider(float& val, float min_, float max_, 
               const std::string& label_, const std::string& tooltip_)
            : value(val), min(min_), max(max_), 
              label(label_), tooltip(tooltip_),
              x(0), y(0), width(200) {}
    };
    
    std::vector<Slider> sliders;
    
    void initializeControls() {
        sliders = {
            {params.rotationSpeed, 0.0f, 5.0f, "Rotation Speed", 
             "Adjusts the speed of rotation (0-5x)"},
            {params.errorWeight, 0.0f, 2.0f, "Error Weight",
             "Weight of error compensation (0-2x)"},
            {params.kernelSigma, 0.01f, 1.0f, "Kernel Sigma",
             "Fredholm kernel sigma parameter"},
            {params.starSpeed, 0.1f, 5.0f, "Star Speed",
             "Movement speed of background stars"}
        };
        
        // Position sliders
        int startY = 100;
        for (auto& slider : sliders) {
            slider.x = 10;
            slider.y = startY;
            startY += 40;
        }
    }
    
    // Enhanced font rendering with button support
    struct Button {
        SDL_Rect rect;
        std::string label;
        std::string tooltip;
        bool* toggle;
        
        Button(const std::string& label_, const std::string& tooltip_, bool* toggle_)
            : label(label_), tooltip(tooltip_), toggle(toggle_) {}
    };
    
    std::vector<Button> buttons;
    
    void initializeButtons() {
        buttons = {
            {"Show Trails", "Toggle star movement trails", &params.showTrails},
            {"Pause", "Pause automatic rotation", &params.pauseRotation},
            {"Hide UI", "Toggle control panel visibility", &ui.showControls}
        };
        
        // Position buttons
        int startY = 300;
        for (auto& button : buttons) {
            button.rect = {10, startY, 100, 30};
            startY += 40;
        }
    }

    void drawSlider(const Slider& slider) {
        // Draw slider background
        SDL_Rect bg = {slider.x, slider.y + 10, slider.width, 4};
        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 255);
        SDL_RenderFillRect(renderer, &bg);
        
        // Draw slider handle
        float normalizedValue = (slider.value - slider.min) / (slider.max - slider.min);
        int handleX = slider.x + normalizedValue * slider.width;
        SDL_Rect handle = {handleX - 5, slider.y + 5, 10, 14};
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderFillRect(renderer, &handle);
        
        // Draw label
        renderText(slider.label, slider.x, slider.y - 15);
        
        // Draw value
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << slider.value;
        renderText(ss.str(), slider.x + slider.width + 10, slider.y + 5);
        
        // Show tooltip if mouse is over
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        if (mouseX >= slider.x && mouseX <= slider.x + slider.width &&
            mouseY >= slider.y && mouseY <= slider.y + 20) {
            ui.activeTooltip = slider.tooltip;
        }
    }
    
    void drawButton(const Button& button) {
        // Draw button background
        SDL_SetRenderDrawColor(renderer, 
            *button.toggle ? 100 : 64,
            *button.toggle ? 100 : 64,
            *button.toggle ? 100 : 64, 255);
        SDL_RenderFillRect(renderer, &button.rect);
        
        // Draw border
        SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
        SDL_RenderDrawRect(renderer, &button.rect);
        
        // Draw label
        renderText(button.label, 
            button.rect.x + 5,
            button.rect.y + button.rect.h/2 - 5);
            
        // Show tooltip if mouse is over
        int mouseX, mouseY;
        SDL_GetMouseState(&mouseX, &mouseY);
        if (mouseX >= button.rect.x && mouseX <= button.rect.x + button.rect.w &&
            mouseY >= button.rect.y && mouseY <= button.rect.y + button.rect.h) {
            ui.activeTooltip = button.tooltip;
        }
    }
    
    void handleMouseDown(int x, int y) {
        if (!ui.showControls) return;
        
        // Check sliders
        for (const auto& slider : sliders) {
            if (x >= slider.x && x <= slider.x + slider.width &&
                y >= slider.y && y <= slider.y + 20) {
                ui.isDragging = true;
                ui.dragStartX = x;
                ui.dragStartY = y;
                ui.dragStartValue = slider.value;
                return;
            }
        }
        
        // Check buttons
        for (auto& button : buttons) {
            if (x >= button.rect.x && x <= button.rect.x + button.rect.w &&
                y >= button.rect.y && y <= button.rect.y + button.rect.h) {
                *button.toggle = !*button.toggle;
                return;
            }
        }
    }
    
    void handleMouseMotion(int x, int y) {
        if (ui.isDragging) {
            for (auto& slider : sliders) {
                if (y >= slider.y && y <= slider.y + 20) {
                    float delta = (x - ui.dragStartX) / float(slider.width);
                    slider.value = std::clamp(
                        ui.dragStartValue + delta * (slider.max - slider.min),
                        slider.min,
                        slider.max
                    );
                    return;
                }
            }
        }
        
        // Manual rotation when paused
        if (params.pauseRotation) {
            params.manualRotation = std::atan2(
                y - WINDOW_HEIGHT/2,
                x - WINDOW_WIDTH/2
            );
        }
    }
    
    void updateStarTrails() {
        if (params.showTrails) {
            for (size_t i = 0; i < stars.size(); ++i) {
                if (i >= starTrails.size()) {
                    starTrails.push_back(Trail());
                }
                
                starTrails[i].points.push_back({
                    int(stars[i].x),
                    int(stars[i].y)
                });
                
                if (starTrails[i].points.size() > starTrails[i].maxPoints) {
                    starTrails[i].points.erase(starTrails[i].points.begin());
                }
            }
        } else {
            starTrails.clear();
        }
    }
    
    void drawStarTrails(int offsetX) {
        if (!params.showTrails) return;
        
        SDL_SetRenderDrawColor(renderer, 64, 64, 128, 128);
        for (const auto& trail : starTrails) {
            for (size_t i = 1; i < trail.points.size(); ++i) {
                SDL_RenderDrawLine(renderer,
                    offsetX + trail.points[i-1].x,
                    trail.points[i-1].y,
                    offsetX + trail.points[i].x,
                    trail.points[i].y
                );
            }
        }
    }

public:
    QuantizationDemo() {
        // Previous initialization...
        initializeControls();
        initializeButtons();
        starTrails.resize(stars.size());
    }
    
    void run() {
        bool running = true;
        Uint32 frameStart, frameTime;
        float fps = 0.0f;
        
        while (running) {
            frameStart = SDL_GetTicks();
            
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        running = false;
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                        handleMouseDown(event.button.x, event.button.y);
                        break;
                    case SDL_MOUSEBUTTONUP:
                        ui.isDragging = false;
                        break;
                    case SDL_MOUSEMOTION:
                        handleMouseMotion(event.motion.x, event.motion.y);
                        break;
                }
            }
            
            // Clear screen
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            
            // Update rotation
            if (!params.pauseRotation) {
                auto now = std::chrono::steady_clock::now();
                float elapsed = std::chrono::duration<float>(now - startTime).count();
                accumulatedRotation = elapsed * M_PI / 4.0f * params.rotationSpeed;
            } else {
                accumulatedRotation = params.manualRotation;
            }
            
            // Update star positions and trails
            for (auto& star : stars) {
                star.speed = params.starSpeed;
            }
            updateStarTrails();
            
            // Draw both panels
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            
            // Left panel (standard)
            drawStarTrails(0);
            drawStars(0, false);
            drawRotatingShape(PANEL_WIDTH/2, WINDOW_HEIGHT/2, accumulatedRotation, false);
            
            // Right panel (compensated)
            drawStarTrails(PANEL_WIDTH);
            drawStars(PANEL_WIDTH, true);
            drawRotatingShape(
                PANEL_WIDTH + PANEL_WIDTH/2,
                WINDOW_HEIGHT/2,
                accumulatedRotation,
                true
            );
            
            // Draw UI
            if (ui.showControls) {
                // Draw control panel background
                SDL_SetRenderDrawColor(renderer, 32, 32, 32, 192);
                SDL_Rect controlPanel = {0, 0, 250, WINDOW_HEIGHT};
                SDL_RenderFillRect(renderer, &controlPanel);
                
                // Draw controls
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                for (const auto& slider : sliders) {
                    drawSlider(slider);
                }
                for (const auto& button : buttons) {
                    drawButton(button);
                }
                
                // Draw tooltip if active
                if (!ui.activeTooltip.empty()) {
                    int mouseX, mouseY;
                    SDL_GetMouseState(&mouseX, &mouseY);
                    renderText(ui.activeTooltip, mouseX + 10, mouseY + 10);
                }
                ui.activeTooltip.clear();
            }
            
            SDL_RenderPresent(renderer);
            
            // Calculate FPS
            frameTime = SDL_GetTicks() - frameStart;
            fps = frameTime > 0 ? 1000.0f / frameTime : 0.0f;
        }
    }
};

// Main function remains the same...

The demo now includes these interactive features:

1. **Real-time Parameter Adjustment**:
   - Rotation Speed: Control the speed of rotating shapes
   - Error Weight: Adjust the strength of error compensation
   - Kernel Sigma: Fine-tune the Fredholm kernel parameter
   - Star Speed: Control background star movement speed

2. **Toggle Controls**:
   - Show/Hide Trails: Visualize motion paths
   - Pause/Resume: Stop automatic rotation
   - Show/Hide UI: Toggle control panel visibility

3. **Interactive Features**:
   - Draggable sliders with real-time updates
   - Clickable toggle buttons
   - Mouse-over tooltips for all controls
   - Manual rotation when paused (by moving mouse)

4. **Visual Feedback**:
   - Motion trails for tracking movement
   - Real-time parameter display
   - Clear visual separation of control panel
   - Interactive tooltips for help

To use the new controls:

1. Drag sliders to adjust parameters
2. Click buttons to toggle features
3. Move mouse over controls to see tooltips
4. When paused, move mouse to control rotation manually
5. Toggle UI visibility to focus on the comparison

Would you like me to:
1. Add more visualization options?
2. Include parameter presets for different scenarios?
3. Add export/import of settings?
4. Include more detailed error visualization?
