#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "FredholmEngine.h"
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace Fredholm;

const int SCREEN_WIDTH = 1200;
const int SCREEN_HEIGHT = 800;

enum class AppMode {
    THEORY,
    APPLICATION
};

class UI {
public:
    struct Slider {
        std::string label;
        float* value;
        float min, max;
        SDL_Rect rect;
        bool dragging = false;
    };

    struct Button {
        std::string label;
        AppMode mode;
        SDL_Rect rect;
    };

    std::vector<Slider> sliders;
    std::vector<Button> buttons;
    TTF_Font* font;

    UI(TTF_Font* f) : font(f) {}

    void addSlider(std::string label, float* val, float min, float max, int x, int y) {
        sliders.push_back({label, val, min, max, {x, y, 200, 20}, false});
    }

    void addButton(std::string label, AppMode mode, int x, int y) {
        buttons.push_back({label, mode, {x, y, 150, 40}});
    }

    void handleEvent(SDL_Event& e) {
        if (e.type == SDL_MOUSEBUTTONDOWN) {
            int mx = e.button.x;
            int my = e.button.y;
            for (auto& s : sliders) {
                if (mx >= s.rect.x && mx <= s.rect.x + s.rect.w &&
                    my >= s.rect.y && my <= s.rect.y + s.rect.h) {
                    s.dragging = true;
                }
            }
            for (auto& b : buttons) {
                if (mx >= b.rect.x && mx <= b.rect.x + b.rect.w &&
                    my >= b.rect.y && my <= b.rect.y + b.rect.h) {
                    currentMode = b.mode;
                }
            }
        } else if (e.type == SDL_MOUSEBUTTONUP) {
            for (auto& s : sliders) s.dragging = false;
        } else if (e.type == SDL_MOUSEMOTION) {
            int mx = e.motion.x;
            for (auto& s : sliders) {
                if (s.dragging) {
                    float pct = (float)(mx - s.rect.x) / s.rect.w;
                    if (pct < 0) pct = 0;
                    if (pct > 1) pct = 1;
                    *s.value = s.min + pct * (s.max - s.min);
                }
            }
        }
    }

    void draw(SDL_Renderer* ren) {
        for (auto& s : sliders) {
            // Label
            SDL_Color white = {255, 255, 255, 255};
            std::stringstream ss;
            ss << s.label << ": " << std::fixed << std::setprecision(2) << *s.value;
            SDL_Surface* surf = TTF_RenderText_Blended(font, ss.str().c_str(), white);
            SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
            SDL_Rect tr = {s.rect.x, s.rect.y - 25, surf->w, surf->h};
            SDL_RenderCopy(ren, tex, NULL, &tr);
            SDL_FreeSurface(surf);
            SDL_DestroyTexture(tex);

            // Track
            SDL_SetRenderDrawColor(ren, 100, 100, 100, 255);
            SDL_RenderFillRect(ren, &s.rect);

            // Handle
            float pct = (*s.value - s.min) / (s.max - s.min);
            SDL_Rect hr = {s.rect.x + (int)(pct * s.rect.w) - 5, s.rect.y - 5, 10, 30};
            SDL_SetRenderDrawColor(ren, 200, 200, 200, 255);
            SDL_RenderFillRect(ren, &hr);
        }

        for (auto& b : buttons) {
            if (currentMode == b.mode)
                SDL_SetRenderDrawColor(ren, 100, 150, 100, 255);
            else
                SDL_SetRenderDrawColor(ren, 70, 70, 70, 255);
            SDL_RenderFillRect(ren, &b.rect);
            SDL_SetRenderDrawColor(ren, 200, 200, 200, 255);
            SDL_RenderDrawRect(ren, &b.rect);

            SDL_Color white = {255, 255, 255, 255};
            SDL_Surface* surf = TTF_RenderText_Blended(font, b.label.c_str(), white);
            SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
            SDL_Rect tr = {b.rect.x + (b.rect.w - surf->w)/2, b.rect.y + (b.rect.h - surf->h)/2, surf->w, surf->h};
            SDL_RenderCopy(ren, tex, NULL, &tr);
            SDL_FreeSurface(surf);
            SDL_DestroyTexture(tex);
        }
    }

    AppMode currentMode = AppMode::THEORY;
};

void drawGraph(SDL_Renderer* ren, int x, int y, int w, int h,
               const std::vector<double>& data, SDL_Color color, double minV, double maxV) {
    if (data.size() < 2) return;
    SDL_SetRenderDrawColor(ren, color.r, color.g, color.b, color.a);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        int x1 = x + (i * w) / (data.size() - 1);
        int x2 = x + ((i + 1) * w) / (data.size() - 1);
        int y1 = y + h - (int)((data[i] - minV) / (maxV - minV) * h);
        int y2 = y + h - (int)((data[i+1] - minV) / (maxV - minV) * h);
        SDL_RenderDrawLine(ren, x1, y1, x2, y2);
    }
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    if (TTF_Init() < 0) return 1;

    SDL_Window* window = SDL_CreateWindow("Fredholm Theory Education Suite",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    TTF_Font* font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18);
    if (!font) return 1;

    UI ui(font);
    float sigma = 0.2f;
    float lambda = 0.5f;
    float sourceFreq = 2.0f;
    ui.addSlider("Kernel Sigma", &sigma, 0.01f, 1.0f, 50, 100);
    ui.addSlider("Lambda", &lambda, -2.0f, 2.0f, 50, 170);
    ui.addSlider("Source Freq", &sourceFreq, 0.1f, 10.0f, 50, 240);
    ui.addButton("Theory Mode", AppMode::THEORY, 50, 20);
    ui.addButton("App Mode", AppMode::APPLICATION, 210, 20);

    // For Application mode
    float jitter = 0.1f;
    float compLambda = 0.8f;
    ui.addSlider("Jitter/Noise", &jitter, 0.0f, 0.5f, 50, 400); // Only shown in App mode
    ui.addSlider("Comp Lambda", &compLambda, 0.0f, 1.0f, 50, 470);

    bool quit = false;
    SDL_Event e;

    Fredholm::AdaptiveCompensator<double> compensator;

    auto startTime = std::chrono::steady_clock::now();

    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = true;
            ui.handleEvent(e);
        }

        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255);
        SDL_RenderClear(renderer);

        if (ui.currentMode == AppMode::THEORY) {
            // Solve Fredholm Equation
            // phi(x) = sin(freq * x) + lambda * integral_0^1 exp(-(x-y)^2/2sigma^2) phi(y) dy
            auto K = [&](double x, double y) {
                double d = x - y;
                return std::exp(-d*d / (2*sigma*sigma));
            };
            auto f = [&](double x) { return std::sin(sourceFreq * M_PI * x); };

            int N = 32;
            auto phi_nodes = Solver::solve(0, 1, lambda, K, f, N);

            std::vector<double> x_vals, f_vals, phi_vals;
            int plotN = 100;
            for (int i = 0; i <= plotN; ++i) {
                double x = (double)i / plotN;
                x_vals.push_back(x);
                f_vals.push_back(f(x));
                if (!phi_nodes.empty())
                    phi_vals.push_back(Solver::interpolate(x, 0, 1, lambda, K, f, phi_nodes, N));
                else
                    phi_vals.push_back(0);
            }

            // Draw Graphs
            int gx = 400, gy = 100, gw = 700, gh = 300;
            SDL_SetRenderDrawColor(renderer, 40, 40, 45, 255);
            SDL_Rect graphRect = {gx, gy, gw, gh};
            SDL_RenderFillRect(renderer, &graphRect);

            drawGraph(renderer, gx, gy, gw, gh, f_vals, {255, 100, 100, 255}, -2, 2);
            drawGraph(renderer, gx, gy, gw, gh, phi_vals, {100, 255, 100, 255}, -2, 2);

            // Draw axis labels for the graph
            SDL_Color textColor = {180, 180, 180, 255};
            SDL_Surface* s_min = TTF_RenderText_Blended(font, "-2.0", textColor);
            SDL_Texture* t_min = SDL_CreateTextureFromSurface(renderer, s_min);
            SDL_Rect r_min = {gx - s_min->w - 5, gy + gh - s_min->h, s_min->w, s_min->h};
            SDL_RenderCopy(renderer, t_min, NULL, &r_min);
            SDL_FreeSurface(s_min); SDL_DestroyTexture(t_min);

            SDL_Surface* s_max = TTF_RenderText_Blended(font, "2.0", textColor);
            SDL_Texture* t_max = SDL_CreateTextureFromSurface(renderer, s_max);
            SDL_Rect r_max = {gx - s_max->w - 5, gy, s_max->w, s_max->h};
            SDL_RenderCopy(renderer, t_max, NULL, &r_max);
            SDL_FreeSurface(s_max); SDL_DestroyTexture(t_max);

            // Draw Kernel Heatmap
            int kx = 400, ky = 450, kw = 300, kh = 300;
            for (int i = 0; i < 50; ++i) {
                for (int j = 0; j < 50; ++j) {
                    double val = K((double)i/50.0, (double)j/50.0);
                    Uint8 c = (Uint8)(val * 255);
                    SDL_SetRenderDrawColor(renderer, c, c/2, 255-c, 255);
                    SDL_Rect r = {kx + i*6, ky + j*6, 6, 6};
                    SDL_RenderFillRect(renderer, &r);
                }
            }

            // Explanation Text
            SDL_Color white = {200, 200, 200, 255};
            const char* theoryText[] = {
                "Fredholm Equation of the Second Kind:",
                "phi(x) = f(x) + lambda * Integral[ K(x,y) phi(y) dy ]",
                "",
                "Red Graph: f(x) - The input 'source' signal.",
                "Green Graph: phi(x) - The output 'balanced' signal.",
                "Heatmap: K(x,y) - The interaction kernel.",
                "",
                "Theory:",
                "Fredholm theory describes how local interactions",
                "(defined by the kernel) reach a global equilibrium.",
                "In Application, we use this to 'smooth' noise while",
                "preserving the structure of the signal."
            };
            int ty = 450;
            for (const char* t : theoryText) {
                SDL_Surface* s = TTF_RenderText_Blended(font, t, white);
                if (s) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    if (tex) {
                        SDL_Rect tr = {720, ty, s->w, s->h};
                        SDL_RenderCopy(renderer, tex, NULL, &tr);
                        SDL_DestroyTexture(tex);
                    }
                    SDL_FreeSurface(s);
                }
                ty += 25;
            }

        } else {
            // Application Mode: Quantization Jitter Reduction
            auto now = std::chrono::steady_clock::now();
            float t = std::chrono::duration<float>(now - startTime).count();

            float rawAngle = t * 1.0f;
            float noisyAngle = rawAngle + ((rand() % 1000) / 1000.0f - 0.5f) * jitter;

            // Quantize noisy angle to simulate low precision
            float qLevels = 16.0f;
            float quantizedAngle = std::floor(noisyAngle * qLevels) / qLevels;

            compensator.setParams(0.1, compLambda);
            float correctedAngle = compensator.compensate(quantizedAngle);

            // Draw two circles with lines
            int cx1 = 500, cy1 = 300, r = 100;
            int cx2 = 800, cy2 = 300;

            auto drawPointer = [&](int cx, int cy, float angle, const char* label) {
                SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
                // Draw circle
                for(int i=0; i<360; i++) {
                    float a = i * M_PI / 180.0;
                    SDL_RenderDrawPoint(renderer, cx + r*cos(a), cy + r*sin(a));
                }
                // Draw Line
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                SDL_RenderDrawLine(renderer, cx, cy, cx + r*cos(angle), cy + r*sin(angle));

                SDL_Color white = {255, 255, 255, 255};
                SDL_Surface* s = TTF_RenderText_Blended(font, label, white);
                SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                SDL_Rect tr = {cx - s->w/2, cy + r + 20, s->w, s->h};
                SDL_RenderCopy(renderer, tex, NULL, &tr);
                SDL_FreeSurface(s);
                SDL_DestroyTexture(tex);
            };

            drawPointer(cx1, cy1, quantizedAngle, "Quantized + Noisy");
            drawPointer(cx2, cy2, correctedAngle, "Fredholm Compensated");

            // Text
            const char* appText[] = {
                "Application: Noise & Quantization Smoothing",
                "Fredholm operators act as advanced filters.",
                "By treating the signal as an integral manifold,",
                "we can recover smoothness lost to quantization.",
                "",
                "Adjust 'Jitter' to see the input destabilize.",
                "Adjust 'Comp Lambda' to see Fredholm recovery."
            };
            int ty = 500;
            for (const char* txt : appText) {
                SDL_Surface* s = TTF_RenderText_Blended(font, txt, {200,200,200,255});
                if (s) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    if (tex) {
                        SDL_Rect tr = {400, ty, s->w, s->h};
                        SDL_RenderCopy(renderer, tex, NULL, &tr);
                        SDL_DestroyTexture(tex);
                    }
                    SDL_FreeSurface(s);
                }
                ty += 25;
            }
        }

        ui.draw(renderer);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    TTF_CloseFont(font);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    TTF_Quit();
    SDL_Quit();

    return 0;
}
