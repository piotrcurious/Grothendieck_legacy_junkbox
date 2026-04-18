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
    COMPENSATOR,
    BVP,
    DEBLUR
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
    AppMode currentMode = AppMode::THEORY;

    UI(TTF_Font* f) : font(f) {}

    void addSlider(std::string label, float* val, float min, float max, int x, int y) {
        sliders.push_back({label, val, min, max, {x, y, 200, 20}, false});
    }

    void addButton(std::string label, AppMode mode, int x, int y) {
        buttons.push_back({label, mode, {x, y, 140, 40}});
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
            SDL_Color white = {255, 255, 255, 255};
            std::stringstream ss;
            ss << s.label << ": " << std::fixed << std::setprecision(2) << *s.value;
            SDL_Surface* surf = TTF_RenderText_Blended(font, ss.str().c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                SDL_Rect tr = {s.rect.x, s.rect.y - 25, surf->w, surf->h};
                SDL_RenderCopy(ren, tex, NULL, &tr);
                SDL_FreeSurface(surf);
                SDL_DestroyTexture(tex);
            }

            SDL_SetRenderDrawColor(ren, 100, 100, 100, 255);
            SDL_RenderFillRect(ren, &s.rect);
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
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                SDL_Rect tr = {b.rect.x + (b.rect.w - surf->w)/2, b.rect.y + (b.rect.h - surf->h)/2, surf->w, surf->h};
                SDL_RenderCopy(ren, tex, NULL, &tr);
                SDL_FreeSurface(surf);
                SDL_DestroyTexture(tex);
            }
        }
    }
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
        y1 = std::clamp(y1, y, y + h);
        y2 = std::clamp(y2, y, y + h);
        SDL_RenderDrawLine(ren, x1, y1, x2, y2);
    }
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) return 1;
    if (TTF_Init() < 0) return 1;

    SDL_Window* window = SDL_CreateWindow("Fredholm Theory Education Suite",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    TTF_Font* font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16);
    if (!font) return 1;

    UI ui(font);
    float sigma = 0.2f;
    float lambda = 0.5f;
    float sourceFreq = 2.0f;
    float jitter = 0.1f;
    float compLambda = 0.8f;
    float potential = 5.0f;
    float alpha = 0.05f;

    ui.addButton("Theory", AppMode::THEORY, 20, 20);
    ui.addButton("Compensator", AppMode::COMPENSATOR, 170, 20);
    ui.addButton("BVP", AppMode::BVP, 320, 20);
    ui.addButton("Deblur", AppMode::DEBLUR, 470, 20);

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

        ui.sliders.clear();
        if (ui.currentMode == AppMode::THEORY) {
            ui.addSlider("Kernel Sigma", &sigma, 0.01f, 1.0f, 50, 120);
            ui.addSlider("Lambda", &lambda, -2.0f, 2.0f, 50, 190);
            ui.addSlider("Source Freq", &sourceFreq, 0.1f, 10.0f, 50, 260);
        } else if (ui.currentMode == AppMode::COMPENSATOR) {
            ui.addSlider("Jitter/Noise", &jitter, 0.0f, 0.5f, 50, 120);
            ui.addSlider("Comp Lambda", &compLambda, 0.0f, 1.0f, 50, 190);
        } else if (ui.currentMode == AppMode::BVP) {
            ui.addSlider("Potential V", &potential, 0.0f, 20.0f, 50, 120);
            ui.addSlider("Source Freq", &sourceFreq, 0.1f, 5.0f, 50, 190);
        } else if (ui.currentMode == AppMode::DEBLUR) {
            ui.addSlider("Blur Sigma", &sigma, 0.01f, 0.5f, 50, 120);
            ui.addSlider("Alpha (Reg)", &alpha, 0.001f, 0.2f, 50, 190);
        }

        if (ui.currentMode == AppMode::THEORY) {
            auto K = [&](double x, double y) {
                double d = x - y; return std::exp(-d*d / (2*sigma*sigma));
            };
            auto f = [&](double x) { return std::sin(sourceFreq * M_PI * x); };
            auto phi_nodes = Solver::solve(0, 1, lambda, K, f, 32);

            std::vector<double> f_vals, phi_vals;
            for (int i = 0; i <= 100; ++i) {
                double x = (double)i / 100.0;
                f_vals.push_back(f(x));
                phi_vals.push_back(phi_nodes.empty() ? 0 : Solver::interpolate(x, 0, 1, lambda, K, f, phi_nodes, 32));
            }
            int gx = 450, gy = 100, gw = 700, gh = 300;
            SDL_SetRenderDrawColor(renderer, 40, 40, 45, 255);
            SDL_Rect gr = {gx, gy, gw, gh}; SDL_RenderFillRect(renderer, &gr);
            drawGraph(renderer, gx, gy, gw, gh, f_vals, {255, 100, 100, 255}, -2, 2);
            drawGraph(renderer, gx, gy, gw, gh, phi_vals, {100, 255, 100, 255}, -2, 2);

            // Draw Kernel Heatmap
            int kx = 450, ky = 450, kw = 200, kh = 200;
            for (int i = 0; i < 40; ++i) {
                for (int j = 0; j < 40; ++j) {
                    double val = K((double)i/40.0, (double)j/40.0);
                    Uint8 c = (Uint8)(val * 255);
                    SDL_SetRenderDrawColor(renderer, c, c/2, 255-c, 255);
                    SDL_Rect r = {kx + i*5, ky + j*5, 5, 5};
                    SDL_RenderFillRect(renderer, &r);
                }
            }

            const char* desc[] = {"General Fredholm Equation:", "phi(x) = f(x) + lambda * Integral[ K(x,y) phi(y) dy ]", "",
                                 "Red: Source f(x), Green: Solution phi(x)",
                                 "Heatmap: Interaction Kernel K(x,y)"};
            int ty = 450;
            for (auto t : desc) {
                SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255});
                if(s) {
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr);
                    ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex);
                }
            }
        } else if (ui.currentMode == AppMode::COMPENSATOR) {
            auto now = std::chrono::steady_clock::now();
            float t = std::chrono::duration<float>(now - startTime).count();
            float raw = t * 1.0f;
            float noisy = raw + ((rand() % 1000) / 1000.0f - 0.5f) * jitter;
            float quantized = std::floor(noisy * 16.0f) / 16.0f;
            compensator.setParams(0.1, compLambda);
            float corrected = compensator.compensate(quantized);

            auto drawPtr = [&](int cx, int cy, float angle, const char* lbl) {
                int r = 80; SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255);
                for(int i=0; i<360; i++) SDL_RenderDrawPoint(renderer, cx + r*cos(i*M_PI/180), cy + r*sin(i*M_PI/180));
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
                SDL_RenderDrawLine(renderer, cx, cy, cx + r*cos(angle), cy + r*sin(angle));
                SDL_Surface* s = TTF_RenderText_Blended(font, lbl, {255,255,255,255});
                if(s){
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    SDL_Rect tr = {cx - s->w/2, cy + r + 10, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr);
                    SDL_FreeSurface(s); SDL_DestroyTexture(tex);
                }
            };
            drawPtr(550, 300, quantized, "Quantized");
            drawPtr(850, 300, corrected, "Corrected");
        } else if (ui.currentMode == AppMode::BVP) {
            // -u'' + V u = f  => u(x) + integral G(x,y) V(y) u(y) dy = integral G(x,y) f(y) dy
            auto G = [](double x, double y) { return (x < y) ? x * (1.0 - y) : y * (1.0 - x); };
            auto V = [&](double x) { return (double)potential; };
            auto f = [&](double x) { return std::sin(sourceFreq * M_PI * x); };

            // F(x) = integral G(x,y) f(y) dy
            auto F = [&](double x) {
                double sum = 0; int n = 32;
                Quadrature q = Quadrature::GaussLegendreN(n, 0, 1);
                for(int i=0; i<n; i++) sum += q.weights[i] * G(x, q.points[i]) * f(q.points[i]);
                return sum;
            };
            auto K = [&](double x, double y) { return G(x, y) * V(y); };
            auto phi_nodes = Solver::solve(0, 1, -1.0, K, F, 32);

            std::vector<double> u_vals, f_scaled;
            for(int i=0; i<=100; i++){
                double x = i/100.0;
                f_scaled.push_back(f(x)*0.1); // Scale for vis
                u_vals.push_back(phi_nodes.empty() ? 0 : Solver::interpolate(x, 0, 1, -1.0, K, F, phi_nodes, 32));
            }
            int gx = 450, gy = 100, gw = 700, gh = 300;
            SDL_SetRenderDrawColor(renderer, 40, 40, 45, 255);
            SDL_Rect gr = {gx, gy, gw, gh}; SDL_RenderFillRect(renderer, &gr);
            drawGraph(renderer, gx, gy, gw, gh, f_scaled, {255, 100, 100, 255}, -0.5, 0.5);
            drawGraph(renderer, gx, gy, gw, gh, u_vals, {100, 255, 100, 255}, -0.5, 0.5);

            const char* desc[] = {"Boundary Value Problem (BVP):", "-u''(x) + V(x)u(x) = f(x), u(0)=u(1)=0", "",
                                 "Green's Function G(x,y) converts the ODE to a Fredholm Eq.",
                                 "Green: Deflection u(x), Red: Force f(x)"};
            int ty = 450;
            for(auto t : desc){
                SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255});
                if(s){
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr);
                    ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex);
                }
            }
        } else if (ui.currentMode == AppMode::DEBLUR) {
            // Blurred b = K s. Solve via Tikhonov: (alpha I + K* K) s = K* b
            auto K_base = [&](double x, double y) {
                double d = x - y; return std::exp(-d*d / (2*sigma*sigma));
            };
            auto s_orig = [&](double x) { return (x > 0.4 && x < 0.6) ? 1.0 : 0.0; };

            // Generate blurred signal b(x)
            auto b_func = [&](double x) {
                double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1);
                for(int i=0; i<n; i++) sum += q.weights[i] * K_base(x, q.points[i]) * s_orig(q.points[i]);
                return sum;
            };

            // K* b = integral K(y,x) b(y) dy
            auto Kb_func = [&](double x) {
                double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1);
                for(int i=0; i<n; i++) sum += q.weights[i] * K_base(q.points[i], x) * b_func(q.points[i]);
                return sum;
            };
            // K*K (x,y) = integral K(z,x) K(z,y) dz
            auto KK_func = [&](double x, double y) {
                double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1);
                for(int i=0; i<n; i++) sum += q.weights[i] * K_base(q.points[i], x) * K_base(q.points[i], y);
                return sum;
            };

            // Equation: alpha s(x) + integral KK(x,y) s(y) dy = Kb(x)
            // => s(x) = (1/alpha) Kb(x) - (1/alpha) integral KK(x,y) s(y) dy
            // This is s = f + lambda integral L s  with f = Kb/alpha, lambda = -1/alpha, L = KK
            auto f_reg = [&](double x) { return Kb_func(x) / alpha; };
            auto phi_nodes = Solver::solve(0, 1, -1.0/alpha, KK_func, f_reg, 32);

            std::vector<double> orig_v, blur_v, deblur_v;
            for(int i=0; i<=100; i++){
                double x = i/100.0;
                orig_v.push_back(s_orig(x));
                blur_v.push_back(b_func(x));
                deblur_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolate(x, 0, 1, -1.0/alpha, KK_func, f_reg, phi_nodes, 32));
            }
            int gx = 450, gy = 100, gw = 700, gh = 300;
            SDL_SetRenderDrawColor(renderer, 40, 40, 45, 255);
            SDL_Rect gr = {gx, gy, gw, gh}; SDL_RenderFillRect(renderer, &gr);
            drawGraph(renderer, gx, gy, gw, gh, orig_v, {80, 80, 80, 255}, -0.2, 1.2);
            drawGraph(renderer, gx, gy, gw, gh, blur_v, {255, 100, 100, 255}, -0.2, 1.2);
            drawGraph(renderer, gx, gy, gw, gh, deblur_v, {100, 255, 100, 255}, -0.2, 1.2);

            const char* desc[] = {"Signal Deblurring (Deconvolution):", "Ill-posed problem solved via Tikhonov Regularization.", "",
                                 "Red: Blurred signal, Green: Recovered signal", "Grey: Original signal (square pulse)"};
            int ty = 450;
            for(auto t : desc){
                SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255});
                if(s){
                    SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s);
                    SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr);
                    ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex);
                }
            }
        }

        ui.draw(renderer);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    TTF_CloseFont(font); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window);
    TTF_Quit(); SDL_Quit();
    return 0;
}
