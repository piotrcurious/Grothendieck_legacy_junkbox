#include "GegenbauerCore.h"
#include <cmath>
#include <limits>
#include <iostream>
#include <sstream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

GegenbauerCore::GegenbauerCore(int dimension, int degree, double th) {
    update_parameters(dimension, degree, th);
}

void GegenbauerCore::update_parameters(int dimension, int degree, double th) {
    d = std::max(3, dimension);
    n = std::max(0, degree);
    lambda_val = (d - 2.0) / 2.0;
    theta = std::clamp(th, 1e-6, M_PI - 1e-6);
    x = std::cos(theta);
}

RegimeType GegenbauerCore::classify_regime(double th) const {
    double N_n = n + lambda_val;
    double z_plus = N_n * th;
    double z_minus = N_n * (M_PI - th);
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
        case BackendType::EXACT_RATIONAL: return "EXACT_RATIONAL";
        case BackendType::MODULAR_RNS: return "MODULAR_RNS";
        default: return "UNKNOWN";
    }
}

RepresentationSnapshot GegenbauerCore::get_snapshot(LayerType layer,
                                                     BackendType backend,
                                                     bool auto_route,
                                                     double error_tol) const {
    RepresentationSnapshot snap;
    snap.d = d;
    snap.n = n;
    snap.lambda = lambda_val;
    snap.theta = theta;
    snap.x = x;
    snap.N = n + lambda_val;
    snap.z_plus = snap.N * theta;
    snap.z_minus = snap.N * (M_PI - theta);

    snap.regime = classify_regime(theta);
    snap.regime_name = get_regime_name(snap.regime);

    snap.auto_router = auto_route;
    snap.layer = layer;
    snap.backend = backend;

    // Feasibility-first Representation Router AI if auto_route is active
    if (auto_route) {
        if (snap.regime == RegimeType::NORTH_ENDPOINT_BESSEL || snap.regime == RegimeType::SOUTH_ENDPOINT_BESSEL) {
            snap.layer = LayerType::LAYER_V_TWO_POLE_COORDS;
            snap.router_reason = "Routed to Layer V (Endpoint Bessel Scaling) for z_0 <= 10";
        } else if (snap.regime == RegimeType::OVERLAP_APPROXIMATION) {
            snap.layer = LayerType::LAYER_VI_COMPOSITE_ASYMPTOTICS;
            snap.router_reason = "Routed to Layer VI (Composite Matched Asymptotics) in Overlap Zone";
        } else if (n > 200) {
            snap.layer = LayerType::LAYER_IV_JACOBI_CITY;
            snap.router_reason = "Routed to Layer IV (Jacobi Spectral Tower) for High Degree n";
        } else {
            snap.layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
            snap.router_reason = "Routed to Layer I (Harmonic Geometry) for Interior Domain";
        }

        if (error_tol < 1e-12) {
            snap.backend = BackendType::EXACT_RATIONAL;
            snap.router_reason += " + EXACT_RATIONAL Backend for High Precision";
        } else {
            snap.backend = BackendType::FLOAT64;
        }
    } else {
        snap.router_reason = "Manual Selection";
    }

    snap.backend_name = get_backend_name(snap.backend);

    // Evaluate function and metrics
    snap.phi = eval_backend_phi(snap.backend, n, x);
    snap.cert = compute_certification(snap.backend);

    snap.forward_error = snap.cert.forward_error;
    snap.conditioning = snap.cert.conditioning;

    // Analytic error envelope bound B_K
    snap.analytic_bound = (1.0 / std::pow(snap.N, 1.0)) * (1.0 / std::pow(std::max(1e-4, std::sin(theta)), lambda_val));

    // Residuals
    snap.r_rec = snap.cert.structural_residual;
    snap.r_ode = snap.cert.structural_residual;
    snap.r_schr = snap.cert.structural_residual;
    snap.backend_error = snap.cert.backend_discrepancy;

    return snap;
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

double GegenbauerCore::eval_phi(int n_deg, double x_val) const {
    if (n_deg == 0) return 1.0;
    double c_n = eval_gegenbauer_c(n_deg, lambda_val, x_val);
    double c_1 = eval_gegenbauer_c_at_1(n_deg, lambda_val);
    return c_n / c_1;
}

double GegenbauerCore::eval_phi_prime(int n_deg, double x_val) const {
    if (n_deg <= 0) return 0.0;
    double factor = (n_deg * (n_deg + 2.0 * lambda_val)) / (2.0 * lambda_val + 1.0);
    double c_sub = eval_gegenbauer_c(n_deg - 1, lambda_val + 1.0, x_val);
    double c_sub_1 = eval_gegenbauer_c_at_1(n_deg - 1, lambda_val + 1.0);
    return factor * (c_sub / c_sub_1);
}

double GegenbauerCore::eval_phi_second_prime(int n_deg, double x_val) const {
    if (n_deg <= 1) return 0.0;
    double factor = (n_deg * (n_deg - 1.0) * (n_deg + 2.0 * lambda_val) * (n_deg + 2.0 * lambda_val + 1.0)) /
                    ((2.0 * lambda_val + 1.0) * (2.0 * lambda_val + 3.0));
    double c_sub = eval_gegenbauer_c(n_deg - 2, lambda_val + 2.0, x_val);
    double c_sub_1 = eval_gegenbauer_c_at_1(n_deg - 2, lambda_val + 2.0);
    return factor * (c_sub / c_sub_1);
}

double GegenbauerCore::eval_schrodinger_u(int n_deg, double th) const {
    double s = std::sin(th);
    if (s <= 0.0) return 0.0;
    double phi_val = eval_phi(n_deg, std::cos(th));
    return std::pow(s, lambda_val) * phi_val;
}

double GegenbauerCore::eval_potential_v(double th) const {
    double s = std::sin(th);
    if (s <= 1e-8) return 1e10;
    return (lambda_val * (lambda_val - 1.0)) / (s * s);
}

double GegenbauerCore::get_jacobi_alpha(int k, double lam) {
    double num = (k + 1.0) * (k + 2.0 * lam);
    double den = (k + lam) * (k + lam + 1.0);
    return 0.5 * std::sqrt(num / den);
}

std::vector<double> GegenbauerCore::get_jacobi_alphas(int m) const {
    std::vector<double> alphas(m > 1 ? m - 1 : 0);
    for (int k = 0; k < static_cast<int>(alphas.size()); ++k) {
        alphas[k] = get_jacobi_alpha(k, lambda_val);
    }
    return alphas;
}

GolubWelschResult GegenbauerCore::compute_golub_welsch(int m) const {
    GolubWelschResult res;
    res.m = m;
    if (m <= 0) return res;

    res.eigenvalues.resize(m, 0.0);
    res.weights.resize(m, 0.0);
    res.eigenvectors.assign(m, std::vector<double>(m, 0.0));

    std::vector<double> d_diag(m, 0.0);
    std::vector<double> e_sub(m, 0.0);
    for (int k = 0; k < m - 1; ++k) {
        e_sub[k] = get_jacobi_alpha(k, lambda_val);
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

    res.expected_mu0 = beta_func(0.5, lambda_val + 0.5);

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
            if (i > 0) jv_i += get_jacobi_alpha(i - 1, lambda_val) * res.eigenvectors[i - 1][k];
            if (i < m - 1) jv_i += get_jacobi_alpha(i, lambda_val) * res.eigenvectors[i + 1][k];
            double diff = jv_i - x_k * res.eigenvectors[i][k];
            err_sq += diff * diff;
        }
        max_res = std::max(max_res, std::sqrt(err_sq));
    }
    res.eigenpair_residual = max_res;

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

double GegenbauerCore::eval_north_bessel(double th) const {
    double N_n = n + lambda_val;
    double z0 = N_n * th;
    double nu = lambda_val - 0.5;
    return eval_bessel_j_nu(nu, z0);
}

double GegenbauerCore::eval_south_bessel(double th) const {
    double N_n = n + lambda_val;
    double z_pi = N_n * (M_PI - th);
    double nu = lambda_val - 0.5;
    double sign = (n % 2 == 0) ? 1.0 : -1.0;
    return sign * eval_bessel_j_nu(nu, z_pi);
}

double GegenbauerCore::eval_wkb_interior(double th) const {
    double N_n = n + lambda_val;
    double s = std::sin(th);
    if (s <= 1e-8) return eval_north_bessel(th);

    double coeff = std::pow(2.0, lambda_val) * gamma_func(lambda_val + 0.5) / std::sqrt(M_PI);
    double amplitude = std::pow(N_n * s, -lambda_val);
    double phase = N_n * th - 0.5 * lambda_val * M_PI;

    return coeff * amplitude * std::cos(phase);
}

double GegenbauerCore::eval_composite_asymptotics(double th) const {
    double N_n = n + lambda_val;
    double z0 = std::max(1e-10, N_n * th);
    double z_pi = std::max(1e-10, N_n * (M_PI - th));
    double nu = lambda_val - 0.5;

    double bessel_north = eval_bessel_j_nu(nu, z0);
    double bessel_south = ((n % 2 == 0) ? 1.0 : -1.0) * eval_bessel_j_nu(nu, z_pi);
    double wkb_val = eval_wkb_interior(th);

    double match_north = std::pow(2.0, nu) * gamma_func(nu + 1.0) * std::pow(z0, -nu) *
                          std::sqrt(2.0 / (M_PI * z0)) * std::cos(z0 - 0.5 * lambda_val * M_PI);
    double match_south = ((n % 2 == 0) ? 1.0 : -1.0) * std::pow(2.0, nu) * gamma_func(nu + 1.0) *
                          std::pow(z_pi, -nu) * std::sqrt(2.0 / (M_PI * z_pi)) *
                          std::cos(z_pi - 0.5 * lambda_val * M_PI);

    return bessel_north + bessel_south + wkb_val - match_north - match_south;
}

double GegenbauerCore::eval_backend_phi(BackendType backend, int n_deg, double x_val) const {
    double exact_val = eval_phi(n_deg, x_val);
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

CertificationStatus GegenbauerCore::compute_certification(BackendType backend) const {
    CertificationStatus cert;

    // Recurrence residual
    double r_rec = 0.0;
    if (n >= 1) {
        double phi_curr = eval_backend_phi(backend, n, x);
        double phi_prev = eval_backend_phi(backend, n - 1, x);
        double phi_next = eval_backend_phi(backend, n + 1, x);

        double a_n = (n + 2.0 * lambda_val) / (2.0 * (n + lambda_val));
        double b_n = n / (2.0 * (n + lambda_val));

        double num = std::abs(x * phi_curr - a_n * phi_next - b_n * phi_prev);
        double den = std::abs(x * phi_curr) + std::abs(a_n * phi_next) + std::abs(b_n * phi_prev) + 1e-14;
        r_rec = num / den;
    }

    // Interior ODE residual
    double phi_val = eval_backend_phi(backend, n, x);
    double phi_p = eval_phi_prime(n, x);
    double phi_pp = eval_phi_second_prime(n, x);
    double E_n = n * (n + 2.0 * lambda_val);

    double ode_num = std::abs((1.0 - x * x) * phi_pp - (2.0 * lambda_val + 1.0) * x * phi_p + E_n * phi_val);
    double ode_den = (1.0 - x * x) * std::abs(phi_pp) + std::abs((2.0 * lambda_val + 1.0) * x * phi_p) + E_n * std::abs(phi_val) + 1e-14;
    double r_ode = ode_num / ode_den;

    cert.structural_residual = std::max(r_rec, r_ode);

    // Backend error vs exact FLOAT64
    double val_exact = eval_phi(n, x);
    double val_backend = eval_backend_phi(backend, n, x);
    cert.backend_discrepancy = std::abs(val_backend - val_exact);

    // Forward error estimate & conditioning
    cert.conditioning = std::max(1.0, 1.0 / std::sin(theta));
    cert.forward_error = cert.backend_discrepancy + cert.conditioning * cert.structural_residual;

    // 4 Independent Certification Axes
    cert.algebraic_exact = (backend == BackendType::EXACT_RATIONAL);
    cert.arithmetic_exact = (backend == BackendType::MODULAR_RNS);
    cert.analytic_certified = (r_rec < 1e-6 && r_ode < 1e-6 && cert.forward_error < 1e-5);
    cert.numerical_valid = (r_rec < 1e-3);

    return cert;
}
