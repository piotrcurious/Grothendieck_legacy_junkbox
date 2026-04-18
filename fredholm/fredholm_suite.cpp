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
#include <map>

using namespace Fredholm;

const int SCREEN_WIDTH = 1250;
const int SCREEN_HEIGHT = 850;

enum class AppMode { THEORY, COMPENSATOR, BVP, DEBLUR, SPECTRAL, VOLTERRA, ALTERNATIVE, NEUMANN, GALERKIN, KINDS };

class TextureCache {
public:
    struct CacheKey {
        std::string text; SDL_Color color;
        bool operator<(const CacheKey& other) const {
            if (text != other.text) return text < other.text;
            return *(uint32_t*)&color < *(uint32_t*)&other.color;
        }
    };
    std::map<CacheKey, SDL_Texture*> cache;
    SDL_Renderer* renderer;
    TTF_Font* font;

    TextureCache(SDL_Renderer* r, TTF_Font* f) : renderer(r), font(f) {}
    ~TextureCache() { clear(); }
    void clear() { for (auto& pair : cache) SDL_DestroyTexture(pair.second); cache.clear(); }

    void renderText(const std::string& text, SDL_Color color, int x, int y, bool centered = false) {
        if (text.empty()) return;
        CacheKey key = {text, color};
        if (cache.find(key) == cache.end()) {
            SDL_Surface* surf = TTF_RenderText_Blended(font, text.c_str(), color);
            if (!surf) return;
            cache[key] = SDL_CreateTextureFromSurface(renderer, surf);
            SDL_FreeSurface(surf);
        }
        SDL_Texture* tex = cache[key];
        int w, h; SDL_QueryTexture(tex, NULL, NULL, &w, &h);
        SDL_Rect dst = {centered ? x - w/2 : x, y, w, h};
        SDL_RenderCopy(renderer, tex, NULL, &dst);
    }
};

class UI {
public:
    struct Slider { std::string label; float* value; float min, max; SDL_Rect rect; bool dragging = false; };
    struct Button { std::string label; AppMode mode; SDL_Rect rect; };
    std::vector<Slider> sliders;
    std::vector<Button> buttons;
    TTF_Font* font;
    TextureCache& cache;
    AppMode currentMode = AppMode::THEORY;

    UI(TTF_Font* f, TextureCache& c) : font(f), cache(c) {}
    void addSlider(std::string label, float* val, float min, float max, int x, int y) { sliders.push_back({label, val, min, max, {x, y, 200, 20}, false}); }
    void addButton(std::string label, AppMode mode, int x, int y) { buttons.push_back({label, mode, {x, y, 90, 40}}); }
    void handleEvent(SDL_Event& e) {
        if (e.type == SDL_MOUSEBUTTONDOWN) {
            int mx = e.button.x, my = e.button.y;
            for (auto& s : sliders) if (mx >= s.rect.x && mx <= s.rect.x + s.rect.w && my >= s.rect.y && my <= s.rect.y + s.rect.h) s.dragging = true;
            for (auto& b : buttons) if (mx >= b.rect.x && mx <= b.rect.x + b.rect.w && my >= b.rect.y && my <= b.rect.y + b.rect.h) currentMode = b.mode;
        } else if (e.type == SDL_MOUSEBUTTONUP) { for (auto& s : sliders) s.dragging = false; }
        else if (e.type == SDL_MOUSEMOTION) {
            int mx = e.motion.x; for (auto& s : sliders) if (s.dragging) *s.value = std::clamp(s.min + (float)(mx - s.rect.x) / s.rect.w * (s.max - s.min), s.min, s.max);
        }
    }
    void draw(SDL_Renderer* ren) {
        for (auto& s : sliders) {
            std::stringstream ss; ss << s.label << ": " << std::fixed << std::setprecision(2) << *s.value;
            cache.renderText(ss.str(), {255,255,255,255}, s.rect.x, s.rect.y - 25);
            SDL_SetRenderDrawColor(ren, 100, 100, 100, 255); SDL_RenderFillRect(ren, &s.rect);
            float pct = (*s.value - s.min) / (s.max - s.min); SDL_Rect hr = {s.rect.x + (int)(pct * s.rect.w) - 5, s.rect.y - 5, 10, 30}; SDL_SetRenderDrawColor(ren, 200, 200, 200, 255); SDL_RenderFillRect(ren, &hr);
        }
        for (auto& b : buttons) {
            SDL_SetRenderDrawColor(ren, currentMode == b.mode ? 100 : 70, currentMode == b.mode ? 150 : 70, currentMode == b.mode ? 100 : 70, 255);
            SDL_RenderFillRect(ren, &b.rect); SDL_SetRenderDrawColor(ren, 200, 200, 200, 255); SDL_RenderDrawRect(ren, &b.rect);
            cache.renderText(b.label, {255,255,255,255}, b.rect.x + b.rect.w/2, b.rect.y + b.rect.h/2 - 10, true);
        }
    }
};

void drawGraph(SDL_Renderer* ren, int x, int y, int w, int h, const std::vector<double>& data, SDL_Color color, double minV, double maxV, TextureCache& cache, bool drawAxes = true) {
    if (data.size() < 2) return;
    if (drawAxes) {
        SDL_SetRenderDrawColor(ren, 60, 60, 65, 255); SDL_RenderDrawLine(ren, x, y + h/2, x + w, y + h/2); SDL_RenderDrawLine(ren, x, y, x, y + h);
        SDL_SetRenderDrawColor(ren, 40, 40, 45, 255); for(int i=1; i<4; i++) { int gy = y + (i * h) / 4; SDL_RenderDrawLine(ren, x, gy, x + w, gy); }
    }
    SDL_SetRenderDrawColor(ren, color.r, color.g, color.b, color.a);
    for (size_t i = 0; i < data.size() - 1; ++i) {
        int x1 = x + (i * w) / (data.size() - 1), x2 = x + ((i + 1) * w) / (data.size() - 1);
        int y1 = y + h - (int)((data[i] - minV) / (maxV - minV) * h), y2 = y + h - (int)((data[i+1] - minV) / (maxV - minV) * h);
        y1 = std::clamp(y1, y, y + h); y2 = std::clamp(y2, y, y + h); SDL_RenderDrawLine(ren, x1, y1, x2, y2);
    }
}

class FredholmDemo {
public:
    virtual ~FredholmDemo() {}
    virtual void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) = 0;
    virtual void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) = 0;
    virtual std::string getEquation() const = 0;
    void drawHeatmap(SDL_Renderer* ren, int kx, int ky, int kw, int kh, std::function<double(double, double)> K) {
        for (int i = 0; i < 40; ++i) for (int j = 0; j < 40; ++j) {
            double val = K((double)i/40.0, (double)j/40.0);
            Uint8 c = (Uint8)(std::clamp(val, 0.0, 1.0) * 255);
            SDL_SetRenderDrawColor(ren, c, c/2, 255-c, 255);
            SDL_Rect r = {kx + i*5, ky + j*5, 5, 5}; SDL_RenderFillRect(ren, &r);
        }
    }
};

class TheoryDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Sigma", &sigma, 0.01f, 1.0f, 50, 120); ui.addSlider("Lambda", &lambda, -2.0f, 2.0f, 50, 190); ui.addSlider("Freq", &freq, 0.1f, 10.0f, 50, 260); ui.addSlider("Kernel", &kType, 0.0f, 1.0f, 50, 330);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { double d = x - y; return (kType < 0.5) ? std::exp(-d*d / (2*sigma*sigma)) : 1.0 / (1.0 + (d*d)/(sigma*sigma)); };
        auto f = [&](double x) { return std::sin(freq * M_PI * x); };
        auto phi_nodes = Solver::solveFredholm(0, 1, lambda, K, f, 16);
        std::vector<double> f_v, phi_v; for (int i = 0; i <= 100; ++i) { double x = i/100.0; f_v.push_back(f(x)); phi_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, lambda, K, f, phi_nodes, 16)); }
        drawGraph(ren, gx, gy, gw, gh, f_v, {255, 100, 100, 255}, -2, 2, cache); drawGraph(ren, gx, gy, gw, gh, phi_v, {100, 255, 100, 255}, -2, 2, cache);
        cache.renderText("Theory: Fredholm Equation of the Second Kind", {200,200,200,255}, 450, 420);
        cache.renderText("Red: Source f(x), Green: Solution phi(x)", {200,200,200,255}, 450, 445);
        drawHeatmap(ren, 20, 450, 200, 200, K);
    }
    std::string getEquation() const override { return "phi(x) = f(x) + lambda * integral[ K(x,y) phi(y) dy ]"; }
};

class SpectralDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Sigma", &sigma, 0.01f, 1.0f, 50, 120); ui.addSlider("Index", &index, 0.0f, 15.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        int N = 16; Fredholm::Matrix M(N, N); Quadrature q = Quadrature::GaussLegendre16();
        auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
        for(int i=0; i<N; i++) for(int j=0; j<N; j++) M(i, j) = q.weights[j] * K(0.5*q.points[i]+0.5, 0.5*q.points[j]+0.5);
        std::vector<double> evals; std::vector<std::vector<double>> evecs; Fredholm::computeEigen(M, evals, evecs);
        int idx = std::clamp((int)index, 0, N-1); std::vector<double> ev_v; for(int i=0; i<N; i++) ev_v.push_back(evecs[idx][i]);
        drawGraph(ren, gx, gy, gw, gh, ev_v, {100, 255, 255, 255}, -1, 1, cache);
        std::stringstream ss; ss << "Eigenvalue: " << evals[idx];
        cache.renderText("Spectral Mode: Natural oscillations of the Kernel", {200,200,200,255}, 450, 420);
        cache.renderText(ss.str(), {200,200,200,255}, 450, 445);
    }
    std::string getEquation() const override { return "integral[ K(x,y) phi(y) dy ] = mu * phi(x)"; }
};

class CompDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Noise", &jitter, 0.0f, 0.5f, 50, 120); ui.addSlider("Lambda", &compLambda, 0.0f, 1.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto now = std::chrono::steady_clock::now(); float t = std::chrono::duration<float>(now - start).count();
        float raw = t * 1.0f, noisy = raw + ((rand() % 1000) / 1000.0f - 0.5f) * jitter, quantized = std::floor(noisy * 16.0f) / 16.0f;
        comp.setParams(0.1, compLambda); float corrected = comp.compensate(quantized);
        auto drawPtr = [&](int cx, int cy, float angle, const char* lbl) {
            int r = 80; SDL_SetRenderDrawColor(ren, 150, 150, 150, 255); for(int i=0; i<360; i++) SDL_RenderDrawPoint(ren, cx + r*cos(i*M_PI/180), cy + r*sin(i*M_PI/180));
            SDL_SetRenderDrawColor(ren, 255, 255, 255, 255); SDL_RenderDrawLine(ren, cx, cy, cx + r*cos(angle), cy + r*sin(angle));
            cache.renderText(lbl, {255,255,255,255}, cx, cy + r + 10, true);
        };
        drawPtr(550, 300, quantized, "Quantized"); drawPtr(850, 300, corrected, "Corrected");
    }
    std::string getEquation() const override { return "y_corr(t) = y_raw(t) - lambda * integral[ K(t-s) (y_corr(s) - y_raw(s)) ds ]"; }
};

class BVPDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Potential", &potential, 0.0f, 20.0f, 50, 120); ui.addSlider("Freq", &freq, 0.1f, 5.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto G = [](double x, double y) { return (x < y) ? x * (1.0 - y) : y * (1.0 - x); };
        auto V = [&](double x) { return (double)potential; }; auto f = [&](double x) { return std::sin(freq * M_PI * x); };
        auto F = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * G(x, q.points[i]) * f(q.points[i]); return sum; };
        auto phi_nodes = Solver::solveFredholm(0, 1, -1.0, [&](double x, double y){return G(x,y)*V(y);}, F, 16);
        std::vector<double> u_v, f_s; for(int i=0; i<=100; i++){ double x = i/100.0; f_s.push_back(f(x)*0.1); u_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, -1.0, [&](double x, double y){return G(x,y)*V(y);}, F, phi_nodes, 16)); }
        drawGraph(ren, gx, gy, gw, gh, f_s, {255, 100, 100, 255}, -0.5, 0.5, cache); drawGraph(ren, gx, gy, gw, gh, u_v, {100, 255, 100, 255}, -0.5, 0.5, cache);
    }
    std::string getEquation() const override { return "u''(x) - V(x)u(x) = f(x)  =>  u(x) = integral[ G(x,y) (f(y) - V(y)u(y)) dy ]"; }
};

class DeblurDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Sigma", &sigma, 0.01f, 0.5f, 50, 120); ui.addSlider("Alpha", &alpha, 0.001f, 0.2f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K_b = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
        auto s_o = [&](double x) { return (x > 0.4 && x < 0.6) ? 1.0 : 0.0; };
        auto b_f = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(x, q.points[i]) * s_o(q.points[i]); return sum; };
        auto Kb_f = [&](double x) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(q.points[i], x) * b_f(q.points[i]); return sum; };
        auto KK_f = [&](double x, double y) { double sum = 0; int n = 32; Quadrature q = Quadrature::GaussLegendreN(n, 0, 1); for(int i=0; i<n; i++) sum += q.weights[i] * K_b(q.points[i], x) * K_b(q.points[i], y); return sum; };
        auto phi_nodes = Solver::solveFredholm(0, 1, -1.0/alpha, KK_f, [&](double x){return Kb_f(x)/alpha;}, 16);
        std::vector<double> o_v, b_v, d_v; for(int i=0; i<=100; i++){ double x = i/100.0; o_v.push_back(s_o(x)); b_v.push_back(b_f(x)); d_v.push_back(phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, -1.0/alpha, KK_f, [&](double x){return Kb_f(x)/alpha;}, phi_nodes, 16)); }
        drawGraph(ren, gx, gy, gw, gh, o_v, {80, 80, 80, 255}, -0.2, 1.2, cache); drawGraph(ren, gx, gy, gw, gh, b_v, {255, 100, 100, 255}, -0.2, 1.2, cache); drawGraph(ren, gx, gy, gw, gh, d_v, {100, 255, 100, 255}, -0.2, 1.2, cache);
    }
    std::string getEquation() const override { return "min ||K*phi - f||^2 + alpha*||phi||^2  =>  (K*K + alpha*I)phi = K*f"; }
};

class VolterraDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Lambda", &lambda, -5.0f, 5.0f, 50, 120); ui.addSlider("Freq", &freq, 0.1f, 5.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { return std::sin(x - y); };
        auto f = [&](double x) { return std::cos(freq * M_PI * x); };
        auto phi = Solver::solveVolterra(0, 2, lambda, K, f, 100);
        std::vector<double> f_v; for(int i=0; i<=100; i++) f_v.push_back(f(i*2.0/100.0));
        drawGraph(ren, gx, gy, gw, gh, f_v, {255, 100, 100, 255}, -10, 10, cache); drawGraph(ren, gx, gy, gw, gh, phi, {100, 255, 100, 255}, -10, 10, cache);
    }
    std::string getEquation() const override { return "phi(x) = f(x) + lambda * integral[0 to x] K(x,y) phi(y) dy"; }
};

class AltDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Lambda", &lambda, 0.0f, 2.0f, 50, 120); ui.addSlider("Sigma", &sigma, 0.1f, 1.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
        auto f = [&](double x) { return 1.0; };
        auto phi_nodes = Solver::solveFredholm(0, 1, lambda, K, f, 16);
        std::vector<double> phi_v; double maxVal = 0;
        for(int i=0; i<=100; i++){ double x = i/100.0; double val = phi_nodes.empty() ? 0 : Solver::interpolateFredholm(x, 0, 1, lambda, K, f, phi_nodes, 16); phi_v.push_back(val); if(std::abs(val) > maxVal) maxVal = std::abs(val); }
        double graphRange = std::max(2.0, maxVal * 1.1);
        drawGraph(ren, gx, gy, gw, gh, phi_v, {100, 255, 100, 255}, -graphRange, graphRange, cache);
    }
    std::string getEquation() const override { return "Fredholm Alternative: If lambda is an eigenvalue, (I - lambda K)phi = f may have no solution."; }
};

class NeumannDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Iters", &nIters, 0.0f, 20.0f, 50, 120); ui.addSlider("Lambda", &lambda, -1.5f, 1.5f, 50, 190); ui.addSlider("Sigma", &sigma, 0.1f, 1.0f, 50, 260);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
        auto f = [&](double x) { return std::sin(freq * M_PI * x); };
        int N = 16; std::vector<double> phi_prev(N, 0.0);
        for(int i=0; i<N; i++) phi_prev[i] = f(Quadrature::GaussLegendre16().points[i] * 0.5 + 0.5);
        for(int i=0; i<(int)nIters; i++) phi_prev = Solver::neumannStep(phi_prev, 0, 1, lambda, K, f, N);
        std::vector<double> phi_v; for(int i=0; i<=100; i++) phi_v.push_back(Solver::interpolateFredholm(i/100.0, 0, 1, lambda, K, f, phi_prev, N));
        drawGraph(ren, gx, gy, gw, gh, phi_v, {255, 255, 100, 255}, -5, 5, cache);
    }
    std::string getEquation() const override { return "phi_{n+1} = f + lambda K phi_n  =>  phi = (I + lambda K + lambda^2 K^2 + ...)f"; }
};

class GalerkinDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Degree", &index, 0.0f, 8.0f, 50, 120); ui.addSlider("Lambda", &lambda, -1.0f, 1.0f, 50, 190); ui.addSlider("Sigma", &sigma, 0.1f, 1.0f, 50, 260); ui.addSlider("Kernel", &kType, 0.0f, 1.0f, 50, 330);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { double d = x - y; return (kType < 0.5) ? std::exp(-d*d / (2*sigma*sigma)) : 1.0 / (1.0 + (d*d)/(sigma*sigma)); };
        auto f = [&](double x) { return std::sin(freq * M_PI * x); };
        auto coeffs = Solver::solveGalerkinOptimized(0, 1, lambda, K, f, (int)index);
        std::vector<double> phi_v;
        for(int i=0; i<=100; i++) {
            double x = (double)i/100.0, val = 0;
            for(int j=0; j<(int)coeffs.size(); j++) val += coeffs[j] * Solver::legendreP(j, 2.0*x - 1.0);
            phi_v.push_back(val);
        }
        drawGraph(ren, gx, gy, gw, gh, phi_v, {100, 255, 100, 255}, -2, 2, cache);
    }
    std::string getEquation() const override { return "phi(x) approx sum c_i P_i(x); <P_i, phi - lambda K phi - f> = 0"; }
};

class KindsDemo : public FredholmDemo {
public:
    void setupSliders(UI& ui, float& sigma, float& lambda, float& freq, float& jitter, float& compLambda, float& potential, float& alpha, float& index, float& kType, float& nIters) override {
        ui.addSlider("Sigma", &sigma, 0.01f, 1.0f, 50, 120); ui.addSlider("Freq", &freq, 0.1f, 5.0f, 50, 190);
    }
    void render(SDL_Renderer* ren, TextureCache& cache, int gx, int gy, int gw, int gh, float sigma, float lambda, float freq, float jitter, float compLambda, float potential, float alpha, float index, float kType, float nIters, std::chrono::steady_clock::time_point start, AdaptiveCompensator<double>& comp) override {
        auto K = [&](double x, double y) { double d = x - y; return std::exp(-d*d / (2*sigma*sigma)); };
        auto f = [&](double x) { return std::sin(freq * M_PI * x); };
        int N = 16; std::vector<std::vector<double>> A(N, std::vector<double>(N)); std::vector<double> B(N);
        Quadrature q = Quadrature::GaussLegendre16();
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) A[i][j] = q.weights[j] * K(0.5*q.points[i]+0.5, 0.5*q.points[j]+0.5);
            B[i] = f(0.5*q.points[i]+0.5);
        }
        std::vector<double> phi_nodes; Fredholm::solveLinearSystem(A, B, phi_nodes);
        std::vector<double> phi_v;
        for(int i=0; i<=100; i++) {
            double x = i/100.0, val = 0; if(!phi_nodes.empty()) for(int j=0; j<N; j++) val += q.weights[j]*K(x, 0.5*q.points[j]+0.5)*phi_nodes[j];
            phi_v.push_back(val);
        }
        drawGraph(ren, gx, gy, gw, gh, phi_v, {255, 100, 255, 255}, -2, 2, cache);
    }
    std::string getEquation() const override { return "Fredholm Equation of the First Kind: integral[ K(x,y) phi(y) dy ] = f(x)"; }
};

int main() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0 || TTF_Init() < 0) return 1;
    SDL_Window* window = SDL_CreateWindow("Fredholm Architect Suite", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    const char* fontPaths[] = {
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "DejaVuSans.ttf"
    };
    TTF_Font* font = nullptr;
    for (auto path : fontPaths) { font = TTF_OpenFont(path, 14); if (font) break; }
    if (!font) { std::cerr << "Failed to load any system font." << std::endl; return 1; }

    TextureCache cache(renderer, font);
    UI ui(font, cache);
    float sigma=0.2f, lambda=0.5f, freq=2.0f, jitter=0.1f, compL=0.8f, pot=5.0f, alpha=0.05f, eigenIdx=0.0f, kType=0.0f, nIters=0.0f;
    ui.addButton("Theory", AppMode::THEORY, 20, 20); ui.addButton("Comp", AppMode::COMPENSATOR, 115, 20); ui.addButton("BVP", AppMode::BVP, 210, 20); ui.addButton("Deblur", AppMode::DEBLUR, 305, 20);
    ui.addButton("Spectral", AppMode::SPECTRAL, 400, 20); ui.addButton("Volterra", AppMode::VOLTERRA, 495, 20); ui.addButton("Alt", AppMode::ALTERNATIVE, 590, 20); ui.addButton("Neumann", AppMode::NEUMANN, 685, 20);
    ui.addButton("Galerkin", AppMode::GALERKIN, 780, 20); ui.addButton("Kinds", AppMode::KINDS, 875, 20);

    std::map<AppMode, std::unique_ptr<FredholmDemo>> demos;
    demos[AppMode::THEORY] = std::make_unique<TheoryDemo>();
    demos[AppMode::COMPENSATOR] = std::make_unique<CompDemo>();
    demos[AppMode::BVP] = std::make_unique<BVPDemo>();
    demos[AppMode::DEBLUR] = std::make_unique<DeblurDemo>();
    demos[AppMode::SPECTRAL] = std::make_unique<SpectralDemo>();
    demos[AppMode::VOLTERRA] = std::make_unique<VolterraDemo>();
    demos[AppMode::ALTERNATIVE] = std::make_unique<AltDemo>();
    demos[AppMode::NEUMANN] = std::make_unique<NeumannDemo>();
    demos[AppMode::GALERKIN] = std::make_unique<GalerkinDemo>();
    demos[AppMode::KINDS] = std::make_unique<KindsDemo>();

    bool quit = false; SDL_Event e; Fredholm::AdaptiveCompensator<double> compensator; auto startTime = std::chrono::steady_clock::now();
    while (!quit) {
        while (SDL_PollEvent(&e)) { if (e.type == SDL_QUIT) quit = true; ui.handleEvent(e); }
        SDL_SetRenderDrawColor(renderer, 20, 20, 25, 255); SDL_RenderClear(renderer);
        ui.sliders.clear();
        if(demos.count(ui.currentMode)) {
            demos[ui.currentMode]->setupSliders(ui, sigma, lambda, freq, jitter, compL, pot, alpha, eigenIdx, kType, nIters);
            int gx = 450, gy = 100, gw = 700, gh = 300; SDL_SetRenderDrawColor(renderer, 35, 35, 40, 255); SDL_Rect gr = {gx, gy, gw, gh}; SDL_RenderFillRect(renderer, &gr);
            demos[ui.currentMode]->render(renderer, cache, gx, gy, gw, gh, sigma, lambda, freq, jitter, compL, pot, alpha, eigenIdx, kType, nIters, startTime, compensator);

            // Shared metadata rendering
            auto K_theory = [&](double x, double y) { double d = x - y; return (kType < 0.5) ? std::exp(-d*d / (2*sigma*sigma)) : 1.0 / (1.0 + (d*d)/(sigma*sigma)); };
            double condNum = Solver::estimateConditionNumber(0, 1, lambda, K_theory, 16);
            std::stringstream statSS; statSS << "Stability Index: " << std::fixed << std::setprecision(3) << condNum;
            cache.renderText(statSS.str(), {150, 200, 255, 255}, 20, 800);
            if (condNum > 5.0) cache.renderText("DIVERGENCE RISK: HIGH", {255, 100, 100, 255}, 20, 820);
            int mx, my; SDL_GetMouseState(&mx, &my);
            if (mx >= gx && mx <= gx + gw && my >= gy && my <= gy + gh) {
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, 100); SDL_RenderDrawLine(renderer, mx, gy, mx, gy + gh); SDL_RenderDrawLine(renderer, gx, my, gx + gw, my);
                double vx = (double)(mx - gx) / gw;
                double v_min = -2.0, v_max = 2.0;
                if (ui.currentMode == AppMode::VOLTERRA || ui.currentMode == AppMode::ALTERNATIVE) { v_min = -10.0; v_max = 10.0; }
                else if (ui.currentMode == AppMode::BVP) { v_min = -0.5; v_max = 0.5; }
                else if (ui.currentMode == AppMode::SPECTRAL) { v_min = -1.0; v_max = 1.0; }
                double vy = v_min + (double)(gy + gh - my) / gh * (v_max - v_min);
                std::stringstream ss; ss << std::fixed << std::setprecision(3) << "(" << vx << ", " << vy << ")";
                cache.renderText(ss.str(), {255, 255, 255, 255}, mx + 10, my - 20);
            }
            // Tooltip for Equation
            if (my >= 420 && my <= 480 && mx >= 450 && mx <= 950) {
                std::string eq = demos[ui.currentMode]->getEquation();
                int w, h; TTF_SizeText(font, eq.c_str(), &w, &h);
                int tx = mx + 20; if (tx + w > SCREEN_WIDTH) tx = SCREEN_WIDTH - w - 10;
                SDL_SetRenderDrawColor(renderer, 30, 30, 30, 240);
                SDL_Rect bg = {tx-5, my+15, w+10, h+10}; SDL_RenderFillRect(renderer, &bg);
                SDL_SetRenderDrawColor(renderer, 200, 200, 100, 255); SDL_RenderDrawRect(renderer, &bg);
                cache.renderText(eq, {255, 255, 100, 255}, tx, my + 20);
            }
        }
        ui.draw(renderer); SDL_RenderPresent(renderer); SDL_Delay(10);
    }
    TTF_CloseFont(font); SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); TTF_Quit(); SDL_Quit(); return 0;
}
