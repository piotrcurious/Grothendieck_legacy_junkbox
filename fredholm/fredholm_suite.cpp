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
const int SCREEN_HEIGHT = 850;

enum class AppMode { THEORY, COMPENSATOR, BVP, DEBLUR, SPECTRAL, VOLTERRA, ALTERNATIVE };

class UI {
public:
    struct Slider { std::string label; float* value; float min, max; SDL_Rect rect; bool dragging = false; };
    struct Button { std::string label; AppMode mode; SDL_Rect rect; };
    std::vector<Slider> sliders;
    std::vector<Button> buttons;
    TTF_Font* font;
    AppMode currentMode = AppMode::THEORY;

    UI(TTF_Font* f) : font(f) {}

    void addSlider(std::string label, float* val, float min, float max, int x, int y) {
        sliders.push_back({label, val, min, max, {x, y, 200, 20}, false});
    }

    void addButton(std::string label, AppMode mode, int x, int y) {
        buttons.push_back({label, mode, {x, y, 100, 40}});
    }

    void handleEvent(SDL_Event& e) {
        if (e.type == SDL_MOUSEBUTTONDOWN) {
            int mx = e.button.x; int my = e.button.y;
            for (auto& s : sliders) if (mx >= s.rect.x && mx <= s.rect.x + s.rect.w && my >= s.rect.y && my <= s.rect.y + s.rect.h) s.dragging = true;
            for (auto& b : buttons) if (mx >= b.rect.x && mx <= b.rect.x + b.rect.w && my >= b.rect.y && my <= b.rect.y + b.rect.h) currentMode = b.mode;
        } else if (e.type == SDL_MOUSEBUTTONUP) {
            for (auto& s : sliders) s.dragging = false;
        } else if (e.type == SDL_MOUSEMOTION) {
            int mx = e.motion.x;
            for (auto& s : sliders) if (s.dragging) {
                float pct = (float)(mx - s.rect.x) / s.rect.w;
                *s.value = std::clamp(s.min + pct * (s.max - s.min), s.min, s.max);
            }
        }
    }

    void draw(SDL_Renderer* ren) {
        for (auto& s : sliders) {
            SDL_Color white = {255, 255, 255, 255};
            std::stringstream ss; ss << s.label << ": " << std::fixed << std::setprecision(2) << *s.value;
            SDL_Surface* surf = TTF_RenderText_Blended(font, ss.str().c_str(), white);
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                SDL_Rect tr = {s.rect.x, s.rect.y - 25, surf->w, surf->h};
                SDL_RenderCopy(ren, tex, NULL, &tr);
                SDL_FreeSurface(surf); SDL_DestroyTexture(tex);
            }
            SDL_SetRenderDrawColor(ren, 100, 100, 100, 255); SDL_RenderFillRect(ren, &s.rect);
            float pct = (*s.value - s.min) / (s.max - s.min);
            SDL_Rect hr = {s.rect.x + (int)(pct * s.rect.w) - 5, s.rect.y - 5, 10, 30};
            SDL_SetRenderDrawColor(ren, 200, 200, 200, 255); SDL_RenderFillRect(ren, &hr);
        }
        for (auto& b : buttons) {
            SDL_SetRenderDrawColor(ren, currentMode == b.mode ? 100 : 70, currentMode == b.mode ? 150 : 70, currentMode == b.mode ? 100 : 70, 255);
            SDL_RenderFillRect(ren, &b.rect); SDL_SetRenderDrawColor(ren, 200, 200, 200, 255); SDL_RenderDrawRect(ren, &b.rect);
            SDL_Surface* surf = TTF_RenderText_Blended(font, b.label.c_str(), {255,255,255,255});
            if (surf) {
                SDL_Texture* tex = SDL_CreateTextureFromSurface(ren, surf);
                SDL_Rect tr = {b.rect.x + (b.rect.w - surf->w)/2, b.rect.y + (b.rect.h - surf->h)/2, surf->w, surf->h};
                SDL_RenderCopy(ren, tex, NULL, &tr);
                SDL_FreeSurface(surf); SDL_DestroyTexture(tex);
            }
        }
    }
};

void drawGraph(SDL_Renderer* ren, int x, int y, int w, int h, const std::vector<double>& data, SDL_Color color, double minV, double maxV, TTF_Font* font, bool drawAxes = true) {
    if (data.size() < 2) return;
    if (drawAxes) {
        SDL_SetRenderDrawColor(ren, 60, 60, 65, 255);
        SDL_RenderDrawLine(ren, x, y + h/2, x + w, y + h/2);
        SDL_RenderDrawLine(ren, x, y, x, y + h);
        SDL_SetRenderDrawColor(ren, 40, 40, 45, 255);
        for(int i=1; i<4; i++) { int gy = y + (i * h) / 4; SDL_RenderDrawLine(ren, x, gy, x + w, gy); }
    }
    SDL_SetRenderDrawColor(ren, color.r, color.g, color.b, color.a);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        int x1 = x + (i * w) / (data.size() - 1);
        int x2 = x + ((i + 1) * w) / (data.size() - 1);
        int y1 = y + h - (int)((data[i] - minV) / (maxV - minV) * h);
        int y2 = y + h - (int)((data[i+1] - minV) / (maxV - minV) * h);
        y1 = std::clamp(y1, y, y + h); y2 = std::clamp(y2, y, y + h);
        SDL_RenderDrawLine(ren, x1, y1, x2, y2);
    }
}

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0 || TTF_Init() < 0) return 1;
    SDL_Window* window = SDL_CreateWindow("Fredholm Education Suite Ultimate", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    TTF_Font* font = TTF_OpenFont("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14);
    if (!font) return 1;

    UI ui(font);
    float sigma = 0.2f, lambda = 0.5f, sourceFreq = 2.0f, jitter = 0.1f, compLambda = 0.8f, potential = 5.0f, alpha = 0.05f, eigenIndex = 0.0f, kernelType = 0.0f;
    ui.addButton("Theory", AppMode::THEORY, 20, 20);
    ui.addButton("Comp", AppMode::COMPENSATOR, 125, 20);
    ui.addButton("BVP", AppMode::BVP, 230, 20);
    ui.addButton("Deblur", AppMode::DEBLUR, 335, 20);
    ui.addButton("Spectral", AppMode::SPECTRAL, 440, 20);
    ui.addButton("Volterra", AppMode::VOLTERRA, 545, 20);
    ui.addButton("Alt", AppMode::ALTERNATIVE, 650, 20);

    bool quit = false; SDL_Event e; Fredholm::AdaptiveCompensator<double> compensator; auto startTime = std::chrono::steady_clock::now();

    while (!quit) {
        while (SDL_PollEvent(&e)) { if (e.type == SDL_QUIT) quit = true; ui.handleEvent(e); }
        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255); SDL_RenderClear(renderer);
        ui.sliders.clear();
        if (ui.currentMode == AppMode::THEORY) { ui.addSlider("Sigma", &sigma, 0.01f, 1.0f, 50, 120); ui.addSlider("Lambda", &lambda, -2.0f, 2.0f, 50, 190); ui.addSlider("Freq", &sourceFreq, 0.1f, 10.0f, 50, 260); ui.addSlider("Kernel (0-G, 1-L)", &kernelType, 0.0f, 1.0f, 50, 330); }
        else if (ui.currentMode == AppMode::COMPENSATOR) { ui.addSlider("Noise", &jitter, 0.0f, 0.5f, 50, 120); ui.addSlider("Lambda", &compLambda, 0.0f, 1.0f, 50, 190); }
        else if (ui.currentMode == AppMode::BVP) { ui.addSlider("Potential", &potential, 0.0f, 20.0f, 50, 120); ui.addSlider("Freq", &sourceFreq, 0.1f, 5.0f, 50, 190); }
        else if (ui.currentMode == AppMode::DEBLUR) { ui.addSlider("Sigma", &sigma, 0.01f, 0.5f, 50, 120); ui.addSlider("Alpha", &alpha, 0.001f, 0.2f, 50, 190); }
        else if (ui.currentMode == AppMode::SPECTRAL) { ui.addSlider("Sigma", &sigma, 0.01f, 1.0f, 50, 120); ui.addSlider("Index", &eigenIndex, 0.0f, 7.0f, 50, 190); }
        else if (ui.currentMode == AppMode::VOLTERRA) { ui.addSlider("Lambda", &lambda, -5.0f, 5.0f, 50, 120); ui.addSlider("Freq", &sourceFreq, 0.1f, 5.0f, 50, 190); }
        else if (ui.currentMode == AppMode::ALTERNATIVE) { ui.addSlider("Lambda", &lambda, 0.0f, 2.0f, 50, 120); ui.addSlider("Sigma", &sigma, 0.1f, 1.0f, 50, 190); }

        int gx = 450, gy = 100, gw = 700, gh = 300; SDL_SetRenderDrawColor(renderer, 35, 35, 40, 255); SDL_Rect gr = {gx, gy, gw, gh}; SDL_RenderFillRect(renderer, &gr);

        if (ui.currentMode == AppMode::THEORY) {
            auto K = [&](double x, double y) {
                double d = x - y;
                if (kernelType < 0.5) return std::exp(-d*d / (2*sigma*sigma)); // Gaussian
                else return 1.0 / (1.0 + (d*d)/(sigma*sigma)); // Lorentzian
            };
            auto f = [&](double x) { return std::sin(sourceFreq * M_PI * x); };
            auto phi_nodes = Solver::solveFredholm(0, 1, lambda, K, f, 32);
            std::vector<double> f_v, phi_v; for (int i = 0; i <= 100; ++i) { double x = i/100.0; f_v.push_back(f(x)); phi_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, lambda, K, f, phi_nodes, 32)); }
            drawGraph(renderer, gx, gy, gw, gh, f_v, {255, 100, 100, 255}, -2, 2, font); drawGraph(renderer, gx, gy, gw, gh, phi_v, {100, 255, 100, 255}, -2, 2, font);
            const char* desc[] = {"Theory: phi(x) = f(x) + lambda * Integral[ K(x,y) phi(y) dy ]", "Red: Source f(x), Green: Solution phi(x)", "This mode shows how an integral operator transforms an input signal."};
            int ty = 420; for(auto t : desc){ SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255}); if(s){ SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s); SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr); ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex); } }
        } else if (ui.currentMode == AppMode::VOLTERRA) {
            auto K = [&](double x, double y) { return std::sin(x - y); };
            auto f = [&](double x) { return std::cos(sourceFreq * M_PI * x); };
            auto phi = Solver::solveVolterra(0, 2, lambda, K, f, 100);
            std::vector<double> f_v; for(int i=0; i<=100; i++) f_v.push_back(f(i*2.0/100.0));
            drawGraph(renderer, gx, gy, gw, gh, f_v, {255, 100, 100, 255}, -10, 10, font); drawGraph(renderer, gx, gy, gw, gh, phi, {100, 255, 100, 255}, -10, 10, font);
            const char* desc[] = {"Volterra Mode: phi(x) = f(x) + lambda * Integral[a to x] K(x,y) phi(y) dy", "Causal system behavior. The integral depends only on history."};
            int ty = 420; for(auto t : desc){ SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255}); if(s){ SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s); SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr); ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex); } }
        } else if (ui.currentMode == AppMode::SPECTRAL) {
            int N = 16; Fredholm::Matrix M(N, N); Quadrature q = Quadrature::GaussLegendre16();
            auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
            for(int i=0; i<N; i++) for(int j=0; j<N; j++) M(i, j) = q.weights[j] * K(q.points[i], q.points[j]);
            std::vector<double> evals; std::vector<std::vector<double>> evecs; Fredholm::computeEigen(M, evals, evecs);
            int idx = std::clamp((int)eigenIndex, 0, N-1);
            std::vector<double> ev_v; for(int i=0; i<N; i++) ev_v.push_back(evecs[idx][i]);
            drawGraph(renderer, gx, gy, gw, gh, ev_v, {100, 255, 255, 255}, -1, 1, font);
            std::stringstream ss; ss << "Eigenvalue: " << evals[idx];
            const char* desc[] = {"Spectral Mode: Analyzing Kernel Eigenfunctions", ss.str().c_str(), "Eigenfunctions represent the 'natural modes' of the kernel.", "Large eigenvalues correspond to smoother, more dominant features."};
            int ty = 420; for(auto t : desc){ SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255}); if(s){ SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s); SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr); ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex); } }
        } else if (ui.currentMode == AppMode::COMPENSATOR) {
            auto now = std::chrono::steady_clock::now(); float t = std::chrono::duration<float>(now - startTime).count();
            float raw = t * 1.0f, noisy = raw + ((rand() % 1000) / 1000.0f - 0.5f) * jitter, quantized = std::floor(noisy * 16.0f) / 16.0f;
            compensator.setParams(0.1, compLambda); float corrected = compensator.compensate(quantized);
            auto drawPtr = [&](int cx, int cy, float angle, const char* lbl) {
                int r = 80; SDL_SetRenderDrawColor(renderer, 150, 150, 150, 255); for(int i=0; i<360; i++) SDL_RenderDrawPoint(renderer, cx + r*cos(i*M_PI/180), cy + r*sin(i*M_PI/180));
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); SDL_RenderDrawLine(renderer, cx, cy, cx + r*cos(angle), cy + r*sin(angle));
                SDL_Surface* s = TTF_RenderText_Blended(font, lbl, {255,255,255,255}); if(s){ SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s); SDL_Rect tr = {cx - s->w/2, cy + r + 10, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr); SDL_FreeSurface(s); SDL_DestroyTexture(tex); }
            };
            drawPtr(550, 300, quantized, "Quantized"); drawPtr(850, 300, corrected, "Corrected");
        } else if (ui.currentMode == AppMode::BVP) {
            auto G = [](double x, double y) { return (x < y) ? x * (1.0 - y) : y * (1.0 - x); };
            auto V = [&](double x) { return (double)potential; }; auto f = [&](double x) { return std::sin(sourceFreq * M_PI * x); };
            auto F = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * G(x, q.points[i]) * f(q.points[i]); return sum; };
            auto phi_nodes = Solver::solveFredholm(0, 1, -1.0, [&](double x, double y){return G(x,y)*V(y);}, F, 32);
            std::vector<double> u_v, f_s; for(int i=0; i<=100; i++){ double x = i/100.0; f_s.push_back(f(x)*0.1); u_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, -1.0, [&](double x, double y){return G(x,y)*V(y);}, F, phi_nodes, 32)); }
            drawGraph(renderer, gx, gy, gw, gh, f_s, {255, 100, 100, 255}, -0.5, 0.5, font); drawGraph(renderer, gx, gy, gw, gh, u_v, {100, 255, 100, 255}, -0.5, 0.5, font);
        } else if (ui.currentMode == AppMode::DEBLUR) {
            auto K_b = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
            auto s_o = [&](double x) { return (x > 0.4 && x < 0.6) ? 1.0 : 0.0; };
            auto b_f = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(x, q.points[i]) * s_o(q.points[i]); return sum; };
            auto Kb_f = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(q.points[i], x) * b_f(q.points[i]); return sum; };
            auto KK_f = [&](double x, double y) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(q.points[i], x) * K_b(q.points[i], y); return sum; };
            auto phi_nodes = Solver::solveFredholm(0, 1, -1.0/alpha, KK_f, [&](double x){return Kb_f(x)/alpha;}, 32);
            std::vector<double> o_v, b_v, d_v; for(int i=0; i<=100; i++){ double x = i/100.0; o_v.push_back(s_o(x)); b_v.push_back(b_f(x)); d_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, -1.0/alpha, KK_f, [&](double x){return Kb_f(x)/alpha;}, phi_nodes, 32)); }
            drawGraph(renderer, gx, gy, gw, gh, o_v, {80, 80, 80, 255}, -0.2, 1.2, font); drawGraph(renderer, gx, gy, gw, gh, b_v, {255, 100, 100, 255}, -0.2, 1.2, font); drawGraph(renderer, gx, gy, gw, gh, d_v, {100, 255, 100, 255}, -0.2, 1.2, font);
        } else if (ui.currentMode == AppMode::ALTERNATIVE) {
            auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
            auto f = [&](double x) { return 1.0; };
            auto phi_nodes = Solver::solveFredholm(0, 1, lambda, K, f, 16);
            std::vector<double> phi_v; double maxVal = 0;
            for(int i=0; i<=100; i++){
                double x = i/100.0;
                double val = phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, lambda, K, f, phi_nodes, 16);
                phi_v.push_back(val); if(std::abs(val) > maxVal) maxVal = std::abs(val);
            }
            double graphRange = std::max(2.0, maxVal * 1.1);
            drawGraph(renderer, gx, gy, gw, gh, phi_v, {100, 255, 100, 255}, -graphRange, graphRange, font);
            const char* desc[] = {"Fredholm Alternative: Resonance Near Eigenvalues", "As lambda approaches 1/eigenvalue, the solution magnitude grows.", "This demonstrates the system's sensitivity to characteristic values."};
            int ty = 420; for(auto t : desc){ SDL_Surface* s = TTF_RenderText_Blended(font, t, {200,200,200,255}); if(s){ SDL_Texture* tex = SDL_CreateTextureFromSurface(renderer, s); SDL_Rect tr = {450, ty, s->w, s->h}; SDL_RenderCopy(renderer, tex, NULL, &tr); ty += 25; SDL_FreeSurface(s); SDL_DestroyTexture(tex); } }
        }

        ui.draw(renderer); SDL_RenderPresent(renderer); SDL_Delay(16);
    }
    TTF_CloseFont(font); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); TTF_Quit(); SDL_Quit(); return 0;
}
