#include <SDL2/SDL.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <algorithm>

// Include previous angle quantization code here...
// (Previous FredholmErrorCompensator and AngleQuantizationTracker implementations)

class QuantizationDemo {
private:
    SDL_Window* window = nullptr;
    SDL_Renderer* renderer = nullptr;
    const int WINDOW_WIDTH = 1280;
    const int WINDOW_HEIGHT = 720;
    const int PANEL_WIDTH = WINDOW_WIDTH / 2;
    
    struct Star {
        float x, y;
        float angle;
        float speed;
        int size;
    };
    
    std::vector<Star> stars;
    AngleQuantizationTracker<float> tracker;
    std::chrono::steady_clock::time_point startTime;
    float accumulatedRotation = 0.0f;
    
    // Font rendering helpers
    struct Character {
        std::vector<SDL_Point> points;
        int width;
    };
    std::unordered_map<char, Character> font;
    
    void initFont() {
        // Simple vector font data for characters
        font['F'] = {{{{0,0},{0,12},{8,12}}, {{0,6},{6,6}}}, 9};
        font['P'] = {{{{0,0},{0,12},{8,12},{8,6},{0,6}}}, 9};
        font['S'] = {{{{8,0},{0,0},{0,6},{8,6},{8,12},{0,12}}}, 9};
        // Add more characters as needed...
    }
    
    void renderText(const std::string& text, int x, int y, float scale = 1.0f) {
        int curX = x;
        for (char c : text) {
            if (font.count(c)) {
                const auto& character = font[c];
                for (size_t i = 0; i < character.points.size() - 1; i++) {
                    SDL_RenderDrawLine(renderer,
                        curX + character.points[i].x * scale,
                        y + character.points[i].y * scale,
                        curX + character.points[i + 1].x * scale,
                        y + character.points[i + 1].y * scale
                    );
                }
                curX += (character.width + 2) * scale;
            }
        }
    }

public:
    QuantizationDemo() : startTime(std::chrono::steady_clock::now()) {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            throw std::runtime_error("SDL initialization failed");
        }
        
        window = SDL_CreateWindow(
            "Angle Quantization Comparison",
            SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH, WINDOW_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        
        if (!window) {
            throw std::runtime_error("Window creation failed");
        }
        
        renderer = SDL_CreateRenderer(
            window, -1,
            SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC
        );
        
        if (!renderer) {
            throw std::runtime_error("Renderer creation failed");
        }
        
        initFont();
        initializeStars();
    }
    
    ~QuantizationDemo() {
        if (renderer) SDL_DestroyRenderer(renderer);
        if (window) SDL_DestroyWindow(window);
        SDL_Quit();
    }
    
    void initializeStars() {
        const int NUM_STARS = 200;
        stars.reserve(NUM_STARS);
        
        for (int i = 0; i < NUM_STARS; ++i) {
            Star star;
            star.x = rand() % PANEL_WIDTH;
            star.y = rand() % WINDOW_HEIGHT;
            star.angle = (rand() % 360) * M_PI / 180.0f;
            star.speed = 1.0f + (rand() % 5);
            star.size = 1 + (rand() % 3);
            stars.push_back(star);
        }
    }
    
    void drawRotatingShape(int centerX, int centerY, float angle, bool useCompensation) {
        const int SHAPE_SIZE = 100;
        std::vector<SDL_Point> points = {
            {-SHAPE_SIZE, -SHAPE_SIZE},
            {SHAPE_SIZE, -SHAPE_SIZE},
            {SHAPE_SIZE, SHAPE_SIZE},
            {-SHAPE_SIZE, SHAPE_SIZE},
            {-SHAPE_SIZE, -SHAPE_SIZE}
        };
        
        float finalAngle;
        if (useCompensation) {
            auto result = tracker.convertToAngle(angle / M_PI);
            finalAngle = result.angle;
        } else {
            finalAngle = angle;
        }
        
        // Transform points
        std::vector<SDL_Point> transformed;
        float cos_a = cos(finalAngle);
        float sin_a = sin(finalAngle);
        
        for (const auto& p : points) {
            SDL_Point tp;
            tp.x = centerX + (p.x * cos_a - p.y * sin_a);
            tp.y = centerY + (p.x * sin_a + p.y * cos_a);
            transformed.push_back(tp);
        }
        
        // Draw lines
        for (size_t i = 0; i < transformed.size() - 1; ++i) {
            SDL_RenderDrawLine(renderer,
                transformed[i].x, transformed[i].y,
                transformed[i + 1].x, transformed[i + 1].y
            );
        }
    }
    
    void drawStars(int offsetX, bool useCompensation) {
        for (auto& star : stars) {
            int x = offsetX + star.x;
            int y = star.y;
            
            float angle = useCompensation ? 
                tracker.convertToAngle(star.angle / M_PI).angle :
                star.angle;
            
            // Draw star
            SDL_Rect rect = {
                x - star.size/2,
                y - star.size/2,
                star.size,
                star.size
            };
            SDL_RenderFillRect(renderer, &rect);
            
            // Update position based on angle
            star.x += cos(angle) * star.speed;
            star.y += sin(angle) * star.speed;
            
            // Wrap around
            if (star.x < 0) star.x = PANEL_WIDTH;
            if (star.x > PANEL_WIDTH) star.x = 0;
            if (star.y < 0) star.y = WINDOW_HEIGHT;
            if (star.y > WINDOW_HEIGHT) star.y = 0;
        }
    }
    
    void renderStats(float fps, float standardError, float compensatedError) {
        std::stringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << fps;
        renderText(ss.str(), 10, 10);
        
        ss.str("");
        ss << "Standard Error: " << std::fixed << std::setprecision(6) << standardError;
        renderText(ss.str(), 10, 30);
        
        ss.str("");
        ss << "Compensated Error: " << std::fixed << std::setprecision(6) << compensatedError;
        renderText(ss.str(), 10, 50);
    }
    
    void run() {
        bool running = true;
        Uint32 frameStart, frameTime;
        float fps = 0.0f;
        float standardError = 0.0f;
        float compensatedError = 0.0f;
        
        while (running) {
            frameStart = SDL_GetTicks();
            
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    running = false;
                }
            }
            
            // Clear screen
            SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
            SDL_RenderClear(renderer);
            
            // Draw dividing line
            SDL_SetRenderDrawColor(renderer, 128, 128, 128, 255);
            SDL_RenderDrawLine(renderer,
                PANEL_WIDTH, 0,
                PANEL_WIDTH, WINDOW_HEIGHT
            );
            
            // Calculate current rotation
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            accumulatedRotation = elapsed * M_PI / 4.0f; // One rotation every 8 seconds
            
            // Draw standard approach (left panel)
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            drawRotatingShape(PANEL_WIDTH/2, WINDOW_HEIGHT/2, accumulatedRotation, false);
            drawStars(0, false);
            
            // Draw compensated approach (right panel)
            drawRotatingShape(PANEL_WIDTH + PANEL_WIDTH/2, WINDOW_HEIGHT/2, accumulatedRotation, true);
            drawStars(PANEL_WIDTH, true);
            
            // Calculate errors
            standardError = fmod(accumulatedRotation, 2.0f * M_PI);
            compensatedError = tracker.getAccumulatedError();
            
            // Render stats
            SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
            renderStats(fps, standardError, compensatedError);
            
            SDL_RenderPresent(renderer);
            
            // Calculate FPS
            frameTime = SDL_GetTicks() - frameStart;
            fps = frameTime > 0 ? 1000.0f / frameTime : 0.0f;
        }
    }
};

int main(int argc, char* argv[]) {
    try {
        QuantizationDemo demo;
        demo.run();
        return 0;
    } catch (const std::exception& e) {
        SDL_ShowSimpleMessageBox(
            SDL_MESSAGEBOX_ERROR,
            "Error",
            e.what(),
            nullptr
        );
        return 1;
    }
}
