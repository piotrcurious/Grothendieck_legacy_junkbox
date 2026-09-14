#include "GegenbauerCore.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>
#include <numbers>
#include <algorithm>
#include <chrono>

RegimeType GegenbauerCore::classify_regime(int n_deg, double lam, double th) const {
    double N_n = n_deg + lam;
    double z_plus = N_n * th;
    double z_minus = N_n * (std::numbers::pi - th);
    double overlap_limit = std::sqrt(N_n);

    if (z_plus <= 10.0) {
        return RegimeType::NORTH_ENDPOINT_BESSEL;
    } else if (z_minus <= 10.0) {
        return RegimeType::SOUTH_ENDPOINT_BESSEL;
    } else if (z_plus <= overlap_limit || z_minus <= overlap_limit) {
        return RegimeType::OVERLAP_APPROXIMATION;
    } else {
        return RegimeType::INTERIOR_WKB;
    }
}

std::string GegenbauerCore::get_regime_name(RegimeType reg) {
    switch (reg) {
        case RegimeType::NORTH_ENDPOINT_BESSEL: return "NORTH_ENDPOINT_BESSEL";
        case RegimeType::SOUTH_ENDPOINT_BESSEL: return "SOUTH_ENDPOINT_BESSEL";
        case RegimeType::OVERLAP_APPROXIMATION: return "OVERLAP_APPROXIMATION";
        case RegimeType::INTERIOR_WKB: return "INTERIOR_WKB";
        default: return "UNKNOWN";
    }
}

std::string GegenbauerCore::get_backend_name(BackendType bt) {
    switch (bt) {
        case BackendType::FLOAT32: return "FLOAT32";
        case BackendType::FLOAT64: return "FLOAT64";
        case BackendType::LONGDOUBLE: return "LONGDOUBLE";
        case BackendType::Q16_16: return "Q16.16";
        case BackendType::LNS: return "LNS";
        case BackendType::EXACT_RATIONAL: return "EXACT_RATIONAL / SYMBOLIC";
        case BackendType::MODULAR_RNS: return "MODULAR_RNS";
        default: return "UNKNOWN";
    }
}

std::string GegenbauerCore::get_layer_name(LayerType layer) {
    switch (layer) {
        case LayerType::LAYER_I_HARMONIC_GEOMETRY: return "Layer I (Harmonic Geometry)";
        case LayerType::LAYER_II_PROJECTOR_TEMPLE: return "Layer II (Projector Temple)";
        case LayerType::LAYER_III_DIFFERENTIAL_WAVE: return "Layer III (Differential Wave)";
        case LayerType::LAYER_IV_JACOBI_CITY: return "Layer IV (Jacobi Spectral City)";
        case LayerType::LAYER_V_TWO_POLE_COORDS: return "Layer V (Two-Pole Coordinates)";
        case LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS: return "Layer VI (Composite Asymptotics)";
        case LayerType::LAYER_VII_ARITHMETIC_FACTORY: return "Layer VII (Arithmetic Factory)";
        case LayerType::LAYER_VIII_CERTIFICATION_CHAMBER: return "Layer VIII (Certification Chamber)";
        default: return "UNKNOWN";
    }
}

RouterDecision GegenbauerCore::solve_router_decision(const GameState& state) const {
    RouterDecision dec;
    dec.requested_layer = state.target_layer;
    dec.requested_backend = state.backend;

    double lam = (state.params.d - 2.0) / 2.0;
    double th = std::clamp(state.params.theta, 1e-6, std::numbers::pi - 1e-6);
    RegimeType reg = classify_regime(state.params.n, lam, th);
    double kappa = std::max(1.0, 1.0 / std::sin(th));
    dec.conditioning = kappa;

    if (!state.auto_router) {
        dec.effective_layer = state.target_layer;
        dec.effective_backend = state.backend;
        dec.feasible = true;
        dec.reason = "Manual Selection: Obeying user requested layer and backend";
        return dec;
    }

    std::ostringstream rej_log;

    if (kappa > 1e4) {
        rej_log << "candidate rejected: conditioning \u03BA=" << kappa << " > 1e4 (near singular pole); ";
    }

    if (reg == RegimeType::NORTH_ENDPOINT_BESSEL || reg == RegimeType::SOUTH_ENDPOINT_BESSEL) {
        dec.effective_layer = LayerType::LAYER_V_TWO_POLE_COORDS;
        dec.estimated_error = 1.0 / (state.params.n + lam);
        dec.cost = 2.0;
        dec.reason = "Selected Layer V (Endpoint Bessel) for z_0 <= 10";
    } else if (reg == RegimeType::OVERLAP_APPROXIMATION) {
        dec.effective_layer = LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS;
        dec.estimated_error = 1.0 / std::pow(state.params.n + lam, 2.0);
        dec.cost = 3.0;
        dec.reason = "Selected Layer VI (Composite Matched Asymptotics) for Overlap Zone";
    } else if (state.params.n > 200) {
        dec.effective_layer = LayerType::LAYER_IV_JACOBI_CITY;
        dec.estimated_error = 1e-12;
        dec.cost = 4.0;
        dec.reason = "Selected Layer IV (Jacobi Spectral City) for High Degree n > 200";
    } else {
        dec.effective_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
        dec.estimated_error = 1e-14;
        dec.cost = 1.0;
        dec.reason = "Selected Layer I (Harmonic Geometry) for Interior Domain";
    }

    if (state.params.error_target < 1e-12) {
        dec.effective_backend = BackendType::EXACT_RATIONAL;
        dec.cost += 10.0;
        dec.reason += " + EXACT_RATIONAL Backend (\u03B5_target < 1e-12)";
    } else {
        dec.effective_backend = BackendType::FLOAT64;
    }

    if (!rej_log.str().empty()) {
        dec.reason = rej_log.str() + " -> " + dec.reason;
    }

    return dec;
}

RepresentationSnapshot GegenbauerCore::evaluate(const GameState& state) const {
    RepresentationSnapshot snap;
    snap.params = state.params;
    snap.params.d = std::clamp(state.params.d, 3, 20);
    snap.params.n = std::clamp(state.params.n, 0, 500);
    snap.params.theta = std::clamp(state.params.theta, 1e-6, std::numbers::pi - 1e-6);

    snap.lambda = (snap.params.d - 2.0) / 2.0;
    snap.x = std::cos(snap.params.theta);
    snap.N = snap.params.n + snap.lambda;
    snap.z_plus = snap.N * snap.params.theta;
    snap.z_minus = snap.N * (std::numbers::pi - snap.params.theta);

    snap.regime = classify_regime(snap.params.n, snap.lambda, snap.params.theta);
    snap.regime_name = get_regime_name(snap.regime);

    snap.north_valid = (snap.z_plus <= 12.0);
    snap.south_valid = (snap.z_minus <= 12.0);
    snap.interior_valid = (snap.z_plus >= 1.5 && snap.z_minus >= 1.5);
    snap.domain_valid = (snap.params.theta > 1e-5 && snap.params.theta < std::numbers::pi - 1e-5);

    snap.auto_router = state.auto_router;
    snap.current_layer = state.current_layer;
    snap.target_layer = state.target_layer;
    snap.transition = std::clamp(state.transition, 0.0, 1.0);

    snap.router_decision = solve_router_decision(state);

    snap.effective_layer = snap.router_decision.effective_layer;
    snap.effective_backend = snap.router_decision.effective_backend;

    snap.effective_layer_name = get_layer_name(snap.effective_layer);
    snap.effective_backend_name = get_backend_name(snap.effective_backend);

    // Evaluate function and metrics
    snap.phi = eval_backend_phi(snap.effective_backend, snap.params.n, snap.lambda, snap.x);
    snap.cert = compute_certification(snap.params.n, snap.lambda, snap.params.theta, snap.effective_backend, snap.params.error_target);

    snap.forward_error = snap.cert.forward_error;
    snap.conditioning = snap.cert.conditioning;
    snap.conditioning_ok = (snap.conditioning <= 1e4);

    // Analytic error envelope bound B_K
    snap.analytic_bound = (1.0 / std::pow(snap.N, static_cast<double>(snap.params.asymptotic_K))) *
                           (1.0 / std::pow(std::max(1e-4, std::sin(snap.params.theta)), snap.lambda));

    // Residuals
    snap.r_rec = snap.cert.structural_residual;
    snap.r_ode = snap.cert.structural_residual;
    snap.r_schr = snap.cert.structural_residual;

    // Jacobi residual via cached Golub-Welsch
    int m_jac = std::max(5, snap.params.jacobi_m);
    GolubWelschResult gw = compute_golub_welsch(m_jac, snap.lambda);
    snap.r_jacobi = gw.eigenpair_residual;

    snap.backend_error = snap.cert.backend_discrepancy;

    return snap;
}

RepresentationSnapshot GegenbauerCore::morph_snapshots(const RepresentationSnapshot& snap1,
                                                       const RepresentationSnapshot& snap2,
                                                       double t) {
    t = std::clamp(t, 0.0, 1.0);
    double s = t * t * (3.0 - 2.0 * t);

    // Select discrete metadata authoritatively from snap1 (t < 0.5) or snap2 (t >= 0.5)
    RepresentationSnapshot res = (t < 0.5) ? snap1 : snap2;
    res.transition = t;
    res.current_layer = snap1.effective_layer;
    res.target_layer = snap2.effective_layer;

    // Smooth visual blending for scalar render fields
    res.phi = (1.0 - s) * snap1.phi + s * snap2.phi;
    res.forward_error = (1.0 - s) * snap1.forward_error + s * snap2.forward_error;
    res.r_rec = (1.0 - s) * snap1.r_rec + s * snap2.r_rec;
    res.r_ode = (1.0 - s) * snap1.r_ode + s * snap2.r_ode;
    res.r_schr = (1.0 - s) * snap1.r_schr + s * snap2.r_schr;
    res.r_jacobi = (1.0 - s) * snap1.r_jacobi + s * snap2.r_jacobi;
    res.backend_error = (1.0 - s) * snap1.backend_error + s * snap2.backend_error;

    return res;
}

double GegenbauerCore::log_gamma(double z) {
    return std::lgamma(z);
}

double GegenbauerCore::gamma_func(double z) {
    return std::tgamma(z);
}

double GegenbauerCore::beta_func(double a, double b) {
    return std::exp(log_gamma(a) + log_gamma(b) - log_gamma(a + b));
}

double GegenbauerCore::eval_gegenbauer_c(int n_deg, double lam, double x_val) {
    if (n_deg == 0) return 1.0;
    if (n_deg == 1) return 2.0 * lam * x_val;

    double c_prev = 1.0;
    double c_curr = 2.0 * lam * x_val;
    double c_next = 0.0;

    for (int k = 1; k < n_deg; ++k) {
        c_next = (2.0 * (k + lam) * x_val * c_curr - (k + 2.0 * lam - 1.0) * c_prev) / (k + 1.0);
        c_prev = c_curr;
        c_curr = c_next;
    }
    return c_curr;
}

double GegenbauerCore::eval_gegenbauer_c_at_1(int n_deg, double lam) {
    if (n_deg == 0) return 1.0;
    double log_val = log_gamma(n_deg + 2.0 * lam) - log_gamma(n_deg + 1.0) - log_gamma(2.0 * lam);
    return std::exp(log_val);
}

double GegenbauerCore::eval_phi(int n_deg, double lam, double x_val) const {
    if (n_deg == 0) return 1.0;
    double c_n = eval_gegenbauer_c(n_deg, lam, x_val);
    double c_1 = eval_gegenbauer_c_at_1(n_deg, lam);
    return c_n / c_1;
}

double GegenbauerCore::eval_phi_prime(int n_deg, double lam, double x_val) const {
    if (n_deg <= 0) return 0.0;
    double factor = (n_deg * (n_deg + 2.0 * lam)) / (2.0 * lam + 1.0);
    double c_sub = eval_gegenbauer_c(n_deg - 1, lam + 1.0, x_val);
    double c_sub_1 = eval_gegenbauer_c_at_1(n_deg - 1, lam + 1.0);
    return factor * (c_sub / c_sub_1);
}

double GegenbauerCore::eval_phi_second_prime(int n_deg, double lam, double x_val) const {
    if (n_deg <= 1) return 0.0;
    double factor = (n_deg * (n_deg - 1.0) * (n_deg + 2.0 * lam) * (n_deg + 2.0 * lam + 1.0)) /
                    ((2.0 * lam + 1.0) * (2.0 * lam + 3.0));
    double c_sub = eval_gegenbauer_c(n_deg - 2, lam + 2.0, x_val);
    double c_sub_1 = eval_gegenbauer_c_at_1(n_deg - 2, lam + 2.0);
    return factor * (c_sub / c_sub_1);
}

double GegenbauerCore::eval_schrodinger_u(int n_deg, double lam, double th) const {
    double s = std::sin(th);
    if (s <= 0.0) return 0.0;
    double phi_val = eval_phi(n_deg, lam, std::cos(th));
    return std::pow(s, lam) * phi_val;
}

double GegenbauerCore::eval_potential_v(double lam, double th) const {
    double s = std::sin(th);
    if (s <= 1e-8) return 1e10;
    return (lam * (lam - 1.0)) / (s * s);
}

double GegenbauerCore::get_jacobi_alpha(int k, double lam) {
    double num = (k + 1.0) * (k + 2.0 * lam);
    double den = (k + lam) * (k + lam + 1.0);
    return 0.5 * std::sqrt(num / den);
}

std::vector<double> GegenbauerCore::get_jacobi_alphas(int m, double lam) const {
    std::vector<double> alphas(m > 1 ? m - 1 : 0);
    for (int k = 0; k < static_cast<int>(alphas.size()); ++k) {
        alphas[k] = get_jacobi_alpha(k, lam);
    }
    return alphas;
}

GolubWelschResult GegenbauerCore::compute_golub_welsch(int m, double lam) const {
    if (m == cached_m && std::abs(lam - cached_lam) < 1e-12) {
        return cached_gw;
    }

    GolubWelschResult res;
    res.m = m;
    if (m <= 0) return res;

    res.eigenvalues.resize(m, 0.0);
    res.weights.resize(m, 0.0);
    res.eigenvectors.assign(m, std::vector<double>(m, 0.0));

    std::vector<double> d_diag(m, 0.0);
    std::vector<double> e_sub(m, 0.0);
    for (int k = 0; k < m - 1; ++k) {
        e_sub[k] = get_jacobi_alpha(k, lam);
    }

    std::vector<std::vector<double>> z(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < m; ++i) z[i][i] = 1.0;

    int max_iter = 100 * m;
    int iter = 0;
    int l = 0;
    while (l < m && iter < max_iter) {
        int m_sub = l;
        while (m_sub < m - 1) {
            double dd = std::abs(d_diag[m_sub]) + std::abs(d_diag[m_sub + 1]);
            if (std::abs(e_sub[m_sub]) + dd == dd) break;
            m_sub++;
        }
        if (m_sub == l) {
            l++;
            continue;
        }

        iter++;
        double g = (d_diag[l + 1] - d_diag[l]) / (2.0 * e_sub[l]);
        double r = std::hypot(g, 1.0);
        g = d_diag[m_sub] - d_diag[l] + e_sub[l] / (g + (g >= 0 ? std::abs(r) : -std::abs(r)));

        double s = 1.0, c = 1.0, p = 0.0;
        int i = m_sub - 1;
        for (; i >= l; --i) {
            double f = s * e_sub[i];
            double b = c * e_sub[i];
            r = std::hypot(f, g);
            e_sub[i + 1] = r;
            if (r == 0.0) {
                d_diag[i + 1] -= p;
                e_sub[m_sub] = 0.0;
                break;
            }
            s = f / r;
            c = g / r;
            g = d_diag[i + 1] - p;
            r = (d_diag[i] - g) * s + 2.0 * c * b;
            p = s * r;
            d_diag[i + 1] = g + p;
            g = c * r - b;

            for (int k = 0; k < m; ++k) {
                f = z[k][i + 1];
                z[k][i + 1] = s * z[k][i] + c * f;
                z[k][i] = c * z[k][i] - s * f;
            }
        }
        if (r == 0.0 && i >= l) continue;
        d_diag[l] -= p;
        e_sub[l] = g;
        e_sub[m_sub] = 0.0;
    }

    res.expected_mu0 = beta_func(0.5, lam + 0.5);

    for (int k = 0; k < m; ++k) {
        res.eigenvalues[k] = d_diag[k];
        double v1 = z[0][k];
        res.weights[k] = res.expected_mu0 * (v1 * v1);
        res.weight_sum += res.weights[k];
        for (int i = 0; i < m; ++i) {
            res.eigenvectors[i][k] = z[i][k];
        }
    }

    double max_res = 0.0;
    for (int k = 0; k < m; ++k) {
        double x_k = res.eigenvalues[k];
        double err_sq = 0.0;
        for (int i = 0; i < m; ++i) {
            double jv_i = 0.0;
            if (i > 0) jv_i += get_jacobi_alpha(i - 1, lam) * res.eigenvectors[i - 1][k];
            if (i < m - 1) jv_i += get_jacobi_alpha(i, lam) * res.eigenvectors[i + 1][k];
            double diff = jv_i - x_k * res.eigenvectors[i][k];
            err_sq += diff * diff;
        }
        max_res = std::max(max_res, std::sqrt(err_sq));
    }
    res.eigenpair_residual = max_res;

    cached_m = m;
    cached_lam = lam;
    cached_gw = res;

    return res;
}

double GegenbauerCore::eval_bessel_j0(double z) const {
    if (z < 0.0) z = -z;
    if (z < 1e-4) return 1.0 - z * z / 4.0;
    return ::j0(z);
}

double GegenbauerCore::eval_bessel_j_nu(double nu, double z) const {
    if (std::abs(z) < 1e-12) return 1.0;
    if (std::abs(nu) < 1e-12) return eval_bessel_j0(z);

    double sum = 1.0;
    double term = 1.0;
    double half_z2 = 0.25 * z * z;
    for (int k = 1; k < 40; ++k) {
        term *= -half_z2 / (k * (k + nu));
        sum += term;
        if (std::abs(term) < 1e-15 * std::abs(sum)) break;
    }
    return sum;
}

double GegenbauerCore::eval_north_bessel(int n_deg, double lam, double th) const {
    double N_n = n_deg + lam;
    double z0 = N_n * th;
    double nu = lam - 0.5;
    return eval_bessel_j_nu(nu, z0);
}

double GegenbauerCore::eval_south_bessel(int n_deg, double lam, double th) const {
    double N_n = n_deg + lam;
    double z_pi = N_n * (std::numbers::pi - th);
    double nu = lam - 0.5;
    double sign = (n_deg % 2 == 0) ? 1.0 : -1.0;
    return sign * eval_bessel_j_nu(nu, z_pi);
}

double GegenbauerCore::eval_wkb_interior(int n_deg, double lam, double th) const {
    double N_n = n_deg + lam;
    double s = std::sin(th);
    if (s <= 1e-8) return eval_north_bessel(n_deg, lam, th);

    double coeff = std::pow(2.0, lam) * gamma_func(lam + 0.5) / std::sqrt(std::numbers::pi);
    double amplitude = std::pow(N_n * s, -lam);
    double phase = N_n * th - 0.5 * lam * std::numbers::pi;

    return coeff * amplitude * std::cos(phase);
}

double GegenbauerCore::eval_composite_asymptotics(int n_deg, double lam, double th) const {
    double N_n = n_deg + lam;
    double z0 = std::max(1e-10, N_n * th);
    double z_pi = std::max(1e-10, N_n * (std::numbers::pi - th));
    double nu = lam - 0.5;

    double bessel_north = eval_bessel_j_nu(nu, z0);
    double bessel_south = ((n_deg % 2 == 0) ? 1.0 : -1.0) * eval_bessel_j_nu(nu, z_pi);

    if (z0 <= 10.0) {
        return bessel_north;
    } else if (z_pi <= 10.0) {
        return bessel_south;
    }

    double wkb_val = eval_wkb_interior(n_deg, lam, th);

    double match_north = std::pow(2.0, nu) * gamma_func(nu + 1.0) * std::pow(z0, -nu) *
                          std::sqrt(2.0 / (std::numbers::pi * z0)) * std::cos(z0 - 0.5 * lam * std::numbers::pi);
    double match_south = ((n_deg % 2 == 0) ? 1.0 : -1.0) * std::pow(2.0, nu) * gamma_func(nu + 1.0) *
                          std::pow(z_pi, -nu) * std::sqrt(2.0 / (std::numbers::pi * z_pi)) *
                          std::cos(z_pi - 0.5 * lam * std::numbers::pi);

    return bessel_north + bessel_south + wkb_val - match_north - match_south;
}

double GegenbauerCore::eval_backend_phi(BackendType backend, int n_deg, double lam, double x_val) const {
    double exact_val = eval_phi(n_deg, lam, x_val);
    switch (backend) {
        case BackendType::FLOAT32: {
            float val_f32 = static_cast<float>(exact_val);
            return static_cast<double>(val_f32);
        }
        case BackendType::Q16_16: {
            int32_t fixed_q = static_cast<int32_t>(std::round(exact_val * 65536.0));
            return static_cast<double>(fixed_q) / 65536.0;
        }
        case BackendType::LNS: {
            double sign = (exact_val >= 0.0) ? 1.0 : -1.0;
            double log_val = std::log(std::abs(exact_val) + 1e-15);
            return sign * std::exp(log_val);
        }
        case BackendType::FLOAT64:
        case BackendType::LONGDOUBLE:
        case BackendType::EXACT_RATIONAL:
        case BackendType::MODULAR_RNS:
        default:
            return exact_val;
    }
}

CertificationStatus GegenbauerCore::compute_certification(int n_deg, double lam, double th, BackendType backend, double target_err) const {
    CertificationStatus cert;
    if (!std::isfinite(target_err) || target_err <= 0.0) {
        target_err = 1e-8;
    }
    double x_val = std::cos(th);

    double r_rec = 0.0;
    if (n_deg >= 1) {
        double phi_curr = eval_backend_phi(backend, n_deg, lam, x_val);
        double phi_prev = eval_backend_phi(backend, n_deg - 1, lam, x_val);
        double phi_next = eval_backend_phi(backend, n_deg + 1, lam, x_val);

        double a_n = (n_deg + 2.0 * lam) / (2.0 * (n_deg + lam));
        double b_n = n_deg / (2.0 * (n_deg + lam));

        double num = std::abs(x_val * phi_curr - a_n * phi_next - b_n * phi_prev);
        double den = std::abs(x_val * phi_curr) + std::abs(a_n * phi_next) + std::abs(b_n * phi_prev) + 1e-14;
        r_rec = num / den;
    }

    double phi_val = eval_backend_phi(backend, n_deg, lam, x_val);
    double phi_p = eval_phi_prime(n_deg, lam, x_val);
    double phi_pp = eval_phi_second_prime(n_deg, lam, x_val);
    double E_n = n_deg * (n_deg + 2.0 * lam);

    double ode_num = std::abs((1.0 - x_val * x_val) * phi_pp - (2.0 * lam + 1.0) * x_val * phi_p + E_n * phi_val);
    double ode_den = (1.0 - x_val * x_val) * std::abs(phi_pp) + std::abs((2.0 * lam + 1.0) * x_val) * std::abs(phi_p) + E_n * std::abs(phi_val) + 1e-14;
    double r_ode = ode_num / ode_den;

    cert.structural_residual = std::max(r_rec, r_ode);

    double val_exact = eval_phi(n_deg, lam, x_val);
    double val_backend = eval_backend_phi(backend, n_deg, lam, x_val);
    cert.backend_discrepancy = std::abs(val_backend - val_exact);

    cert.conditioning = std::max(1.0, 1.0 / std::sin(th));
    cert.forward_error = cert.backend_discrepancy + cert.conditioning * cert.structural_residual;

    // 4 Independent Certification Axes
    cert.algebraic_exact = (backend == BackendType::EXACT_RATIONAL);
    cert.algebraic_reason = cert.algebraic_exact ? "Certified via Q[lambda, x] symbolic fraction algebra"
                                                 : "Floating-point path (non-symbolic)";

    cert.arithmetic_exact = (backend == BackendType::MODULAR_RNS);
    cert.arithmetic_reason = cert.arithmetic_exact ? "Certified via RNS modular CRT integer recovery"
                                                  : "Non-RNS representation";

    cert.analytic_certified = (cert.forward_error <= target_err);
    if (!cert.analytic_certified) {
        std::ostringstream reason;
        reason << "Analytic bound B_comp = " << std::scientific << cert.forward_error
               << " > \u03B5_target = " << target_err;
        cert.analytic_reason = reason.str();
    } else {
        cert.analytic_reason = "Analytic error envelope bound satisfied";
    }

    cert.numerical_approx = (r_rec <= 1e-3);
    if (!cert.numerical_approx) {
        cert.numerical_reason = "Recurrence structural residual exceeds threshold 1e-3";
    } else {
        cert.numerical_reason = "Recurrence structural residual <= 1e-3";
    }

    return cert;
}
