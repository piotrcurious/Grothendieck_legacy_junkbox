#ifndef GEGENBAUER_CORE_H
#define GEGENBAUER_CORE_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>

// Certification status matching Layer VIII taxonomy
struct CertificationStatus {
    bool algebraic_exact = false;
    bool arithmetic_exact = false;
    bool analytic_certified = false;
    bool numerical_valid = false;

    double structural_residual = 0.0;
    double forward_error = 0.0;
    double conditioning = 1.0;
    double backend_discrepancy = 0.0;
};

// Arithmetic backends matching Layer VII factory
enum class BackendType {
    FLOAT32 = 0,
    FLOAT64 = 1,
    LONGDOUBLE = 2,
    Q16_16 = 3,
    LNS = 4,
    EXACT_RATIONAL = 5,
    MODULAR_RNS = 6
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

// Regime classification matching Layer V / VI
enum class RegimeType {
    NORTH_ENDPOINT_BESSEL = 0,
    SOUTH_ENDPOINT_BESSEL = 1,
    OVERLAP_APPROXIMATION = 2,
    INTERIOR_WKB = 3
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

struct RepresentationSnapshot {
    int d = 3;
    int n = 5;
    double lambda = 0.5;
    double theta = 0.5;
    double x = 0.87758256;
    double N = 5.5;
    double z_plus = 2.75;
    double z_minus = 14.528;

    RegimeType regime = RegimeType::INTERIOR_WKB;
    std::string regime_name = "INTERIOR_WKB";

    LayerType layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType backend = BackendType::FLOAT64;
    std::string backend_name = "FLOAT64";

    bool auto_router = false;
    std::string router_reason = "Manual Selection";

    double phi = 0.0;
    double forward_error = 0.0;
    double conditioning = 1.0;
    double analytic_bound = 0.0;

    double r_rec = 0.0;
    double r_ode = 0.0;
    double r_schr = 0.0;
    double r_jacobi = 0.0;
    double backend_error = 0.0;

    CertificationStatus cert;
};

class GegenbauerCore {
public:
    int d = 3;              // Dimension (>= 3)
    int n = 5;              // Degree (>= 0)
    double lambda_val = 0.5;// (d-2)/2
    double theta = 0.5;     // Angle in (0, pi)
    double x = 0.87758256;  // cos(theta)
    BackendType current_backend = BackendType::FLOAT64;

    GegenbauerCore(int dimension = 3, int degree = 5, double th = 0.5);

    void update_parameters(int dimension, int degree, double th);

    // Snapshot generator producing an immutable mathematical state snapshot
    RepresentationSnapshot get_snapshot(LayerType layer = LayerType::LAYER_I_HARMONIC_GEOMETRY,
                                         BackendType backend = BackendType::FLOAT64,
                                         bool auto_route = false,
                                         double error_tol = 1e-8) const;

    // Domain regime classifier
    RegimeType classify_regime(double th) const;
    static std::string get_regime_name(RegimeType reg);
    static std::string get_backend_name(BackendType bt);

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

    // Residual taxonomy & 4-axis certification
    CertificationStatus compute_certification(BackendType backend = BackendType::FLOAT64) const;

    // Helper math functions
    static double log_gamma(double z);
    static double gamma_func(double z);
    static double beta_func(double a, double b);
};

#endif // GEGENBAUER_CORE_H
