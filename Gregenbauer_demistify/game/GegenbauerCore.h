#ifndef GEGENBAUER_CORE_H
#define GEGENBAUER_CORE_H

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <numbers>
#include <chrono>
#include <tuple>

constexpr int kNumLayers = 8;
constexpr int kNumBackends = 7;

// Certification status matching Layer VIII taxonomy
struct CertificationStatus {
    bool algebraic_exact = false;
    bool arithmetic_exact = false;
    bool analytic_certified = false;
    bool numerical_approx = false;

    std::string algebraic_reason;
    std::string arithmetic_reason;
    std::string analytic_reason;
    std::string numerical_reason;

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

struct RouterCandidate {
    LayerType layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType backend = BackendType::FLOAT64;
    double estimated_error = 0.0;
    double conditioning = 1.0;
    double cost = 1.0;
    bool feasible = true;
    std::string rejection_reason;
};

struct RouterDecision {
    LayerType requested_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType requested_backend = BackendType::FLOAT64;
    LayerType effective_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType effective_backend = BackendType::FLOAT64;

    double estimated_error = 0.0;
    double conditioning = 1.0;
    double cost = 1.0;
    bool feasible = true;
    std::string reason;
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

// Interactive Probe Particle moving on S^{d-1} or in quantum potential V(theta)
struct ProbeParticle {
    double theta = 0.5;
    double phi_angle = 0.0;
    double v_theta = 0.0;
    double v_phi = 0.0;
    double energy = 1.0;
    bool active = false;
    bool pinned = false;
    std::vector<std::tuple<double, double, double>> trail;
};

// Anchor Point: zero node, extrema, or derivative boundary anchor
enum class AnchorType {
    ZERO_NODE = 0,
    LOCAL_EXTREMUM = 1,
    NORTH_POLE_ANCHOR = 2,
    SOUTH_POLE_ANCHOR = 3,
    TURNING_POINT = 4
};

struct AnchorPoint {
    AnchorType type = AnchorType::ZERO_NODE;
    double theta = 0.0;
    double x = 1.0;
    double val = 0.0;
    double deriv = 0.0;
    std::string label;
};

// Boundary Layer Specification
struct BoundaryZone {
    double north_bessel_limit = 0.1; // z_plus <= threshold
    double south_bessel_limit = 0.1; // z_minus <= threshold
    double turning_point_north = 0.05;
    double turning_point_south = std::numbers::pi - 0.05;
    bool in_north_zone = false;
    bool in_south_zone = false;
    bool in_interior_wkb = true;
};

// Projection Specification for S^{d-1} -> R^3
struct ProjectionSpec {
    double slice_theta = 0.5;
    double slice_offset = 0.0;
    double focal_distance = 3.0;
    double proj_matrix[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    bool show_projection_plane = true;
    bool show_rays = true;
};

struct CoreParameters {
    int d = 3;
    int n = 5;
    double theta = 0.5;
    double error_target = 1e-8;
    int error_target_idx = 1;
    int asymptotic_K = 1;
    int jacobi_m = 10;
};

struct GameState {
    CoreParameters params;
    LayerType current_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    LayerType target_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType backend = BackendType::FLOAT64;
    bool auto_router = false;
    double transition = 0.0; // T in [0, 1]

    // Interactive mechanics & objects
    ProbeParticle probe;
    ProjectionSpec projection;
    bool sim_running = false;
    bool show_anchors = true;
    bool show_boundaries = true;
};

struct SnapshotKey {
    int d = 3;
    int n = 5;
    double theta = 0.5;
    int K = 1;
    int m = 10;
    int error_target_idx = 1;
    BackendType backend = BackendType::FLOAT64;
    LayerType evaluation_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    bool auto_router = false;

    bool operator==(const SnapshotKey& o) const {
        return d == o.d && n == o.n && std::abs(theta - o.theta) < 1e-12 &&
               K == o.K && m == o.m && error_target_idx == o.error_target_idx &&
               backend == o.backend && evaluation_layer == o.evaluation_layer &&
               auto_router == o.auto_router;
    }
    bool operator!=(const SnapshotKey& o) const { return !(*this == o); }
};

struct RepresentationSnapshot {
    CoreParameters params;
    double lambda = 0.5;
    double x = 0.87758256;
    double N = 5.5;
    double z_plus = 2.75;
    double z_minus = 14.528;

    RegimeType regime = RegimeType::INTERIOR_WKB;
    std::string regime_name = "INTERIOR_WKB";
    bool north_valid = false;
    bool interior_valid = true;
    bool south_valid = false;
    bool domain_valid = true;
    bool conditioning_ok = true;

    // Interactive Objects & Mechanical Anchors
    ProbeParticle probe;
    ProjectionSpec projection;
    BoundaryZone boundaries;
    std::vector<AnchorPoint> anchors;
    double probe_potential = 0.0;
    double probe_force = 0.0;

    LayerType current_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    LayerType target_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    double transition = 0.0;

    LayerType effective_layer = LayerType::LAYER_I_HARMONIC_GEOMETRY;
    BackendType effective_backend = BackendType::FLOAT64;
    std::string effective_layer_name = "LAYER_I_HARMONIC_GEOMETRY";
    std::string effective_backend_name = "FLOAT64";

    bool auto_router = false;
    RouterDecision router_decision;

    double phi = 0.0;
    double forward_error = 0.0;
    double conditioning = 1.0;
    double analytic_bound = 0.0;
    double structural_scale = 1.0;
    double certification_floor = 1e-14;
    double forward_error_bound = 1e-5;

    double r_rec = 0.0;
    double r_ode = 0.0;
    double r_schr = 0.0;
    double r_jacobi = 0.0;
    double backend_error = 0.0;

    CertificationStatus cert;

    SnapshotKey make_key(LayerType eval_layer) const {
        return SnapshotKey{params.d, params.n, params.theta, params.asymptotic_K,
                           params.jacobi_m, params.error_target_idx, params.error_target_idx >= 0 ? effective_backend : effective_backend, eval_layer, auto_router};
    }
};

class GegenbauerCore {
private:
    mutable int cached_m = -1;
    mutable double cached_lam = -1.0;
    mutable GolubWelschResult cached_gw;

public:
    GegenbauerCore() = default;

    RepresentationSnapshot evaluate(const GameState& state, LayerType eval_layer) const;
    RepresentationSnapshot evaluate(const GameState& state) const { return evaluate(state, state.target_layer); }

    static RepresentationSnapshot morph_snapshots(const RepresentationSnapshot& snap1,
                                                   const RepresentationSnapshot& snap2,
                                                   double t);

    RouterDecision solve_router_decision(const GameState& state, LayerType req_layer) const;

    RegimeType classify_regime(int n_deg, double lam, double th) const;
    static std::string get_regime_name(RegimeType reg);
    static std::string get_backend_name(BackendType bt);
    static std::string get_layer_name(LayerType layer);

    static double eval_gegenbauer_c(int n_deg, double lam, double x_val);
    static double eval_gegenbauer_c_at_1(int n_deg, double lam);

    double eval_phi(int n_deg, double lam, double x_val) const;
    double eval_phi_prime(int n_deg, double lam, double x_val) const;
    double eval_phi_second_prime(int n_deg, double lam, double x_val) const;

    double eval_schrodinger_u(int n_deg, double lam, double th) const;
    double eval_potential_v(double lam, double th) const;

    static double get_jacobi_alpha(int k, double lam);
    std::vector<double> get_jacobi_alphas(int m, double lam) const;

    GolubWelschResult compute_golub_welsch(int m, double lam) const;

    double eval_bessel_j0(double z) const;
    double eval_bessel_j_nu(double nu, double z) const;
    double eval_north_bessel(int n_deg, double lam, double th) const;
    double eval_south_bessel(int n_deg, double lam, double th) const;
    double eval_wkb_interior(int n_deg, double lam, double th) const;
    double eval_composite_asymptotics(int n_deg, double lam, double th) const;

    double eval_backend_phi(BackendType backend, int n_deg, double lam, double x_val) const;

    CertificationStatus compute_certification(int n_deg, double lam, double th, BackendType backend, double target_err = 1e-8) const;

    static double log_gamma(double z);
    static double gamma_func(double z);
    static double beta_func(double a, double b);
};

#endif // GEGENBAUER_CORE_H
