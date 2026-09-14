#ifndef GEGENBAUER_CORE_H
#define GEGENBAUER_CORE_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

// Truth classes matching Layer VIII taxonomy
enum class TruthClass {
    ALGEBRAIC_EXACT,
    ARITHMETIC_EXACT,
    ANALYTIC_CERTIFIED,
    NUMERICAL_APPROX
};

// Arithmetic backends matching Layer VII factory
enum class BackendType {
    FLOAT32,
    FLOAT64,
    LONGDOUBLE,
    Q16_16,
    LNS,
    EXACT_RATIONAL,
    MODULAR_RNS
};

// Representation layers matching VIII-Layer framework
enum class LayerType {
    LAYER_I_HARMONIC_GEOMETRY = 0,
    LAYER_II_PROJECTOR_TEMPLE = 1,
    LAYER_III_DIFFERENTIAL_WAVE = 2,
    LAYER_IV_JACOBI_CITY = 3,
    LAYER_V_TWO_POLE_COORDS = 4,
    LAYER_VI_COMPOSITE_ASYMPTOTICS = 5,
    LAYER_VII_ARITHMETIC_FACTORY = 6,
    LAYER_VIII_CERTIFICATION_CHAMBER = 7
};

struct ResidualState {
    double r_rec = 0.0;
    double r_ode = 0.0;
    double r_schr = 0.0;
    double r_jacobi = 0.0;
    double e_backend = 0.0;
    double forward_error_est = 0.0;
    double condition_number = 1.0;
    bool is_certified = false;
    TruthClass truth_class = TruthClass::NUMERICAL_APPROX;
};

struct GolubWelschResult {
    int m = 0;
    std::vector<double> eigenvalues;      // x_k in (-1, 1)
    std::vector<double> weights;          // w_k
    std::vector<std::vector<double>> eigenvectors; // Columns v_k
    double weight_sum = 0.0;             // sum w_k (should equal mu_0)
    double expected_mu0 = 0.0;
    double eigenpair_residual = 0.0;     // max ||J_m v_k - x_k v_k||
};

class GegenbauerCore {
public:
    int d = 3;              // Dimension (>= 3)
    int n = 5;              // Degree (>= 0)
    double lambda_val = 0.5;// (d-2)/2
    double theta = 0.5;     // Angle in (0, pi)
    double x = 0.87758256;  // cos(theta)

    GegenbauerCore(int dimension = 3, int degree = 5, double th = 0.5);

    void update_parameters(int dimension, int degree, double th);

    // Exact Gegenbauer C_n^{(lambda)}(x) via recurrence
    static double eval_gegenbauer_c(int n_deg, double lam, double x_val);

    // C_n^{(lambda)}(1) = (2*lam)_n / n!
    static double eval_gegenbauer_c_at_1(int n_deg, double lam);

    // Normalized phi_n(x) = C_n^{(lambda)}(x) / C_n^{(lambda)}(1)
    double eval_phi(int n_deg, double x_val) const;
    double eval_phi() const { return eval_phi(n, x); }

    // Derivatives phi_n'(x) and phi_n''(x)
    double eval_phi_prime(int n_deg, double x_val) const;
    double eval_phi_second_prime(int n_deg, double x_val) const;

    // Schrödinger wave u_n(theta) = (sin theta)^lambda * phi_n(cos theta)
    double eval_schrodinger_u(int n_deg, double th) const;
    double eval_potential_v(double th) const;

    // Jacobi matrix coefficients alpha_k for k = 0 ... m-2
    static double get_jacobi_alpha(int k, double lam);
    std::vector<double> get_jacobi_alphas(int m) const;

    // Golub-Welsch spectral tridiagonal solver for J_m
    GolubWelschResult compute_golub_welsch(int m) const;

    // Boundary layer asymptotics
    double eval_bessel_j0(double z) const;
    double eval_bessel_j_nu(double nu, double z) const;
    double eval_north_bessel(double th) const;
    double eval_south_bessel(double th) const;
    double eval_wkb_interior(double th) const;
    double eval_composite_asymptotics(double th) const;

    // Multi-backend simulations
    double eval_backend_phi(BackendType backend, int n_deg, double x_val) const;

    // Residual taxonomy and certification
    ResidualState compute_residuals(BackendType backend = BackendType::FLOAT64) const;

    // Helper: log gamma function
    static double log_gamma(double z);
    static double gamma_func(double z);
    static double beta_func(double a, double b);
};

#endif // GEGENBAUER_CORE_H
