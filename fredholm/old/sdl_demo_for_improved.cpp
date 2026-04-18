// Include previous angle quantization code...

class QuantizationDemo {
private:
    // Previous member variables remain...
    
    struct EnhancedParameters : public DemoParameters {
        float kernelWidth = 0.1f;
        float dampingFactor = 0.95f;
        float frequencyWeight = 1.0f;
        float timeDecay = 0.1f;
        bool showSpectralView = false;
        bool showErrorDistribution = false;
        int spectralResolution = 64;
    } params;
    
    // Error visualization
    struct ErrorVisualization {
        static constexpr int HISTORY_SIZE = 300;
        std::vector<float> standardErrors;
        std::vector<float> compensatedErrors;
        std::vector<float> confidenceHistory;
        std::vector<float> spectralMagnitudes;
        
        ErrorVisualization() {
            standardErrors.reserve(HISTORY_SIZE);
            compensatedErrors.reserve(HISTORY_SIZE);
            confidenceHistory.reserve(HISTORY_SIZE);
            spectralMagnitudes.resize(64, 0.0f);
        }
        
        void update(float stdError, float compError, float confidence) {
            if (standardErrors.size() >= HISTORY_SIZE) {
                standardErrors.erase(standardErrors.begin());
                compensatedErrors.erase(compensatedErrors.begin());
                confidenceHistory.erase(confidenceHistory.begin());
            }
            standardErrors.push_back(stdError);
            compensatedErrors.push_back(compError);
            confidenceHistory.push_back(confidence);
        }
    } errorVis;
    
    // Enhanced visualization methods
    void drawErrorGraph(int x, int y, int width, int height) {
        // Background
        SDL_Rect bg = {x, y, width, height};
        SDL_SetRenderDrawColor(renderer, 32, 32, 32, 255);
        SDL_RenderFillRect(renderer, &bg);
        
        // Grid lines
        SDL_SetRenderDrawColor(renderer, 64, 64, 64, 255);
        for (int i = 0; i <= 4; ++i) {
            int lineY = y + (height * i) / 4;
            SDL_RenderDrawLine(renderer, x, lineY, x + width, lineY);
        }
        
        // Draw error histories
        auto drawLine = [&](const std::vector<float>& data, 
                          Uint8 r, Uint8 g, Uint8 b) {
            if (data.empty()) return;
            
            for (size_t i = 1; i < data.size(); ++i) {
                int x1 = x + ((i - 1) * width) / ErrorVisualization::HISTORY_SIZE;
                int x2 = x + (i * width) / ErrorVisualization::HISTORY_SIZE;
                int y1 = y + height - (data[i - 1] * height);
                int y2 = y + height - (data[i] * height);
                
                SDL_SetRenderDrawColor(renderer, r, g, b, 255);
                SDL_RenderDrawLine(renderer, x1, y1, x2, y2);
            }
        };
        
        // Draw standard errors in red
        drawLine(errorVis.standardErrors, 255, 64, 64);
        // Draw compensated errors in green
        drawLine(errorVis.compensatedErrors, 64, 255, 64);
        // Draw confidence in blue
        drawLine(errorVis.confidenceHistory, 64, 64, 255);
        
        // Labels
        renderText("Error History", x + 5, y + 5);
        renderText("Standard", x + width - 100, y + 5, {255, 64, 64});
        renderText("Compensated", x + width - 100, y + 25, {64, 255, 64});
        renderText("Confidence", x + width - 100, y + 45, {64, 64, 255});
    }
    
    void drawSpectralView(int x, int y, int width, int height) {
        if (!params.showSpectralView) return;
        
        SDL_Rect bg = {x, y, width, height};
        SDL_SetRenderDrawColor(renderer, 32, 32, 32, 255);
        SDL_RenderFillRect(renderer, &bg);
        
        // Draw frequency components
        int barWidth = width / params.spectralResolution;
        for (int i = 0; i < params.spectralResolution; ++i) {
            int barHeight = int(errorVis.spectralMagnitudes[i] * height);
            SDL_Rect bar = {
                x + i * barWidth,
                y + height - barHeight,
                barWidth - 1,
                barHeight
            };
            
            // Color based on magnitude
            int intensity = int(errorVis.spectralMagnitudes[i] * 255);
            SDL_SetRenderDrawColor(renderer, intensity, 128, 255 - intensity, 255);
            SDL_RenderFillRect(renderer, &bar);
        }
        
        renderText("Frequency Spectrum", x + 5, y + 5);
    }
    
    void drawErrorDistribution(int x, int y, int width, int height) {
        if (!params.showErrorDistribution) return;
        
        SDL_Rect bg = {x, y, width, height};
        SDL_SetRenderDrawColor(renderer, 32, 32, 32, 255);
        SDL_RenderFillRect(renderer, &bg);
        
        // Create error distribution histogram
        std::vector<int> histogram(20, 0);
        for (float error : errorVis.compensatedErrors) {
            int bin = int((error + 1.0f) * 10.0f);
            bin = std::clamp(bin, 0, 19);
            histogram[bin]++;
        }
        
        // Draw histogram
        int maxCount = *std::max_element(histogram.begin(), histogram.end());
        int barWidth = width / histogram.size();
        
        for (size_t i = 0; i < histogram.size(); ++i) {
            int barHeight = height * histogram[i] / maxCount;
            SDL_Rect bar = {
                x + int(i * barWidth),
                y + height - barHeight,
                barWidth - 1,
                barHeight
            };
            
            SDL_SetRenderDrawColor(renderer, 128, 128, 255, 255);
            SDL_RenderFillRect(renderer, &bar);
        }
        
        renderText("Error Distribution", x + 5, y + 5);
    }
    
    // Enhanced control panel
    void drawEnhancedControls() {
        if (!ui.showControls) return;
        
        int panelWidth = 300;
        SDL_Rect panel = {0, 0, panelWidth, WINDOW_HEIGHT};
        SDL_SetRenderDrawColor(renderer, 32, 32, 32, 220);
        SDL_RenderFillRect(renderer, &panel);
        
        int y = 10;
        
        // Draw all parameter sliders
        auto drawParameter = [&](const char* label, float& value, 
                               float min, float max, const char* tooltip) {
            renderText(label, 10, y);
            
            SDL_Rect slider = {10, y + 20, panelWidth - 20, 10};
            SDL_SetRenderDrawColor(renderer, 64, 64, 64, 255);
            SDL_RenderFillRect(renderer, &slider);
            
            float normalized = (value - min) / (max - min);
            SDL_Rect handle = {
                10 + int(normalized * (panelWidth - 20) - 5),
                y + 15,
                10,
                20
            };
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            SDL_RenderFillRect(renderer, &handle);
            
            // Value display
            std::stringstream ss;
            ss << std::fixed << std::setprecision(3) << value;
            renderText(ss.str(), panelWidth - 70, y + 20);
            
            // Handle mouse interaction
            int mouseX, mouseY;
            Uint32 buttons = SDL_GetMouseState(&mouseX, &mouseY);
            if (buttons & SDL_BUTTON(1)) {
                if (mouseY >= y + 15 && mouseY <= y + 35 &&
                    mouseX >= 10 && mouseX <= panelWidth - 10) {
                    float newValue = (mouseX - 10) / float(panelWidth - 20);
                    value = min + newValue * (max - min);
                    value = std::clamp(value, min, max);
                }
            }
            
            y += 50;
        };
        
        // Parameter sliders
        drawParameter("Rotation Speed", params.rotationSpeed, 0.0f, 5.0f,
                     "Adjust rotation speed");
        drawParameter("Kernel Width", params.kernelWidth, 0.01f, 1.0f,
                     "Fredholm kernel width");
        drawParameter("Damping Factor", params.dampingFactor, 0.5f, 0.99f,
                     "Error damping strength");
        drawParameter("Frequency Weight", params.frequencyWeight, 0.1f, 5.0f,
                     "Spectral analysis weight");
        drawParameter("Time Decay", params.timeDecay, 0.01f, 1.0f,
                     "Historical error decay rate");
        
        // Visualization toggles
        auto drawToggle = [&](const char* label, bool& value) {
            SDL_Rect button = {10, y, panelWidth - 20, 30};
            SDL_SetRenderDrawColor(renderer, value ? 64 : 32, 
                                 value ? 64 : 32, 
                                 value ? 128 : 32, 255);
            SDL_RenderFillRect(renderer, &button);
            renderText(label, 20, y + 8);
            
            // Handle click
            int mouseX, mouseY;
            Uint32 buttons = SDL_GetMouseState(&mouseX, &mouseY);
            if (buttons & SDL_BUTTON(1)) {
                if (mouseY >= y && mouseY <= y + 30 &&
                    mouseX >= 10 && mouseX <= panelWidth - 10) {
                    value = !value;
                }
            }
            
            y += 40;
        };
        
        drawToggle("Show Spectral View", params.showSpectralView);
        drawToggle("Show Error Distribution", params.showErrorDistribution);
        drawToggle("Show Trails", params.showTrails);
        drawToggle("Pause Rotation", params.pauseRotation);
    }
    
    void updateVisualization(const AngleQuantizationTracker<float>::AngleResult& result) {
        errorVis.update(
            std::abs(result.angle - result.rawAngle),
            result.errorEstimate,
            result.confidence
        );
        
        // Update spectral magnitudes if needed
        if (params.showSpectralView) {
            // This would be populated from the spectral analysis in the compensator
            // For demo purposes, we'll simulate some frequency components
            for (int i = 0; i < params.spectralResolution; ++i) {
                float freq = i * 2.0f * M_PI / params.spectralResolution;
                errorVis.spectralMagnitudes[i] = 
                    std::abs(std::sin(freq * accumulatedRotation)) * 
                    std::exp(-freq * params.timeDecay);
            }
        }
    }

public:
    void run() {
        bool running = true;
        while (running) {
            // ... Previous event handling ...
            
            // Clear and draw dividing line
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            SDL_SetRenderDrawColor(renderer, 64, 64, 64, 255);
            SDL_RenderDrawLine(renderer, PANEL_WIDTH, 0, PANEL_WIDTH, WINDOW_HEIGHT);
            
            // Update rotation
            if (!params.pauseRotation) {
                auto now = std::chrono::steady_clock::now();
                float deltaTime = std::chrono::duration<float>(now - startTime).count();
                accumulatedRotation = deltaTime * M_PI / 4.0f * params.rotationSpeed;
            }
            
            // Draw both panels with enhanced visualization
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            
            // Standard approach (left panel)
            drawStarTrails(0);
            drawStars(0, false);
            drawRotatingShape(PANEL_WIDTH/2, WINDOW_HEIGHT/2, accumulatedRotation, false);
            
            // Compensated approach (right panel)
            auto result = tracker.convertToAngle(accumulatedRotation / M_PI, 1.0f/60.0f);
            updateVisualization(result);
            
            drawStarTrails(PANEL_WIDTH);
            drawStars(PANEL_WIDTH, true);
            drawRotatingShape(PANEL_WIDTH + PANEL_WIDTH/2, WINDOW_HEIGHT/2, 
                            result.angle, true);
            
            // Draw visualization panels
            int visHeight = WINDOW_HEIGHT / 3;
            drawErrorGraph(PANEL_WIDTH + 10, 10, PANEL_WIDTH - 20, visHeight - 20);
            
            if (params.showSpectralView) {
                drawSpectralView(PANEL_WIDTH + 10, visHeight + 10, 
                               PANEL_WIDTH - 20, visHeight - 20);
            }
            
            if (params.showErrorDistribution) {
                drawErrorDistribution(PANEL_WIDTH + 10, 2 * visHeight + 10,
                                   PANEL_WIDTH - 20, visHeight - 20);
            }
            
            // Draw enhanced controls
            drawEnhancedControls();
            
            SDL_RenderPresent(renderer);
        }
    }
};
