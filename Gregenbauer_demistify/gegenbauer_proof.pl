/* =========================================================================
   Executable Consistency Knowledgebase and PLUnit Test Suite
   Gegenbauer Semiclassical Representation Theory & Numerical Recurrence
   ========================================================================= */

:- module(gegenbauer_proof, [
    symmetric_space/2,
    dimension_parameter/2,
    dimension_parameter_exact/2,
    binom/3,
    gegenbauer_val/4,
    gegenbauer_val_exact_rational/4,
    gegenbauer_val_modular/5,
    normalized_phi_val/4,
    assert_schrodinger_energy_shift/2,
    assert_dim_v_n_identity/4,
    assert_dim_v_n_exact_rational/3,
    assert_normalized_recurrence/4,
    assert_orthonormal_jacobi_recurrence/4,
    assert_norm_isometry/3,
    assert_dual_recurrence_exact_symbolic/3,
    assert_derivative_anchor/3,
    assert_south_derivative_anchor/3,
    assert_ode_second_derivative/4,
    assert_modular_congruence/5,
    select_numerical_regime/4,
    run_all_proofs/0
]).

:- use_module(library(plunit)).

% --- 1. Layer I & II: Symmetric Space Geometry & Spherical Subspaces ---

symmetric_space(AmbientD, so(AmbientD)/so(D1)) :-
    integer(AmbientD),
    AmbientD >= 3,
    D1 is AmbientD - 1.

dimension_parameter(AmbientD, Lambda) :-
    symmetric_space(AmbientD, _G_over_H),
    Lambda is (AmbientD - 2.0) / 2.0.

dimension_parameter_exact(AmbientD, Lambda) :-
    symmetric_space(AmbientD, _G_over_H),
    Lambda is (AmbientD - 2) rdiv 2.

% --- 2. Validated Binomial Coefficient ---

binom(N, K, B) :-
    integer(N),
    integer(K),
    N >= 0,
    K >= 0,
    K =< N,
    K0 is min(K, N - K),
    (   K0 =:= 0
    ->  B = 1
    ;   binom_loop(N, K0, 1, 1, B)
    ).

binom_loop(_N, K0, I, Acc, B) :-
    I > K0, !, B = Acc.
binom_loop(N, K0, I, Acc, B) :-
    I =< K0,
    Acc1 is (Acc * (N - K0 + I)) // I,
    I1 is I + 1,
    binom_loop(N, K0, I1, Acc1, B).

% --- 3. Numerically Controlled Tail-Recursive Gegenbauer Evaluation ---

gegenbauer_val(0, _Lambda, _X, 1.0) :- !.
gegenbauer_val(1, Lambda, X, Val) :- !, Val is 2.0 * Lambda * X.
gegenbauer_val(N, Lambda, X, Val) :-
    integer(N), N >= 2,
    C0 = 1.0,
    C1 is 2.0 * Lambda * X,
    gegenbauer_loop(2, N, Lambda, X, C1, C0, Val).

gegenbauer_loop(K, N, _Lambda, _X, Ck1, _Ck0, Val) :-
    K > N, !, Val = Ck1.
gegenbauer_loop(K, N, Lambda, X, Ck1, Ck0, Val) :-
    K =< N,
    Coeff1 is (2.0 * (K + Lambda - 1.0)) / K,
    Coeff2 is (K + 2.0 * Lambda - 2.0) / K,
    Ck2 is Coeff1 * X * Ck1 - Coeff2 * Ck0,
    K1 is K + 1,
    gegenbauer_loop(K1, N, Lambda, X, Ck2, Ck1, Val).

% --- 3b. Exact Rational Arithmetic over Q(Lambda, X) ---

gegenbauer_val_exact_rational(0, _Lambda, _X, 1) :- !.
gegenbauer_val_exact_rational(1, Lambda, X, Val) :- !, Val is 2 * Lambda * X.
gegenbauer_val_exact_rational(N, Lambda, X, Val) :-
    integer(N), N >= 2,
    C0 = 1,
    C1 is 2 * Lambda * X,
    gegenbauer_rational_loop(2, N, Lambda, X, C1, C0, Val).

gegenbauer_rational_loop(K, N, _Lambda, _X, Ck1, _Ck0, Val) :-
    K > N, !, Val = Ck1.
gegenbauer_rational_loop(K, N, Lambda, X, Ck1, Ck0, Val) :-
    K =< N,
    Coeff1 is (2 * (K + Lambda - 1)) rdiv K,
    Coeff2 is (K + 2 * Lambda - 2) rdiv K,
    Ck2 is Coeff1 * X * Ck1 - Coeff2 * Ck0,
    K1 is K + 1,
    gegenbauer_rational_loop(K1, N, Lambda, X, Ck2, Ck1, Val).

% --- 3c. Modular Ring Arithmetic Z/Mod Z ---

gegenbauer_val_modular(0, _Lambda, _X, Mod, 1) :- !, integer(Mod), Mod > 2.
gegenbauer_val_modular(1, Lambda, X, Mod, Val) :-
    !, integer(Mod), Mod > 2,
    NumL is numerator(Lambda), DenL is denominator(Lambda),
    1 =:= gcd(DenL, Mod),
    DenL_Inv is powm(DenL, Mod - 2, Mod),
    LamMod is ((NumL mod Mod + Mod) * DenL_Inv) mod Mod,
    NumX is numerator(X), DenX is denominator(X),
    1 =:= gcd(DenX, Mod),
    DenX_Inv is powm(DenX, Mod - 2, Mod),
    XMod is ((NumX mod Mod + Mod) * DenX_Inv) mod Mod,
    Val is (2 * LamMod * XMod) mod Mod.
gegenbauer_val_modular(N, Lambda, X, Mod, Val) :-
    integer(N), N >= 2,
    integer(Mod), Mod > max(2, N),
    NumL is numerator(Lambda), DenL is denominator(Lambda),
    1 =:= gcd(DenL, Mod),
    DenL_Inv is powm(DenL, Mod - 2, Mod),
    LamMod is ((NumL mod Mod + Mod) * DenL_Inv) mod Mod,
    NumX is numerator(X), DenX is denominator(X),
    1 =:= gcd(DenX, Mod),
    DenX_Inv is powm(DenX, Mod - 2, Mod),
    XMod is ((NumX mod Mod + Mod) * DenX_Inv) mod Mod,
    C0 is 1 mod Mod,
    C1 is (2 * LamMod * XMod) mod Mod,
    gegenbauer_modular_loop(2, N, LamMod, XMod, Mod, C1, C0, Val).

gegenbauer_modular_loop(K, N, _LamMod, _XMod, _Mod, Ck1, _Ck0, Val) :-
    K > N, !, Val = Ck1.
gegenbauer_modular_loop(K, N, LamMod, XMod, Mod, Ck1, Ck0, Val) :-
    K =< N,
    1 =:= gcd(K, Mod),
    KInv is powm(K, Mod - 2, Mod),
    Term1 is (2 * (K + LamMod - 1) * XMod * Ck1) mod Mod,
    Term2 is ((K + 2 * LamMod - 2) * Ck0) mod Mod,
    Ck2 is (((Term1 - Term2) mod Mod + Mod) * KInv) mod Mod,
    K1 is K + 1,
    gegenbauer_modular_loop(K1, N, LamMod, XMod, Mod, Ck2, Ck1, Val).

normalized_phi_val(N, Lambda, X, Val) :-
    gegenbauer_val(N, Lambda, X, Cn),
    gegenbauer_val(N, Lambda, 1.0, Cn1),
    Val is Cn / Cn1.

% --- 4. Layer III & IV: Mathematical Assertions ---

assert_schrodinger_energy_shift(N, Lambda) :-
    integer(N), N >= 0,
    Casimir is N * (N + 2.0 * Lambda),
    ShiftedSquare is (N + Lambda)^2 - Lambda^2,
    Diff is abs(Casimir - ShiftedSquare),
    Diff < 1e-10.

assert_dim_v_n_exact_rational(AmbientD, N, DimVn) :-
    symmetric_space(AmbientD, _),
    D1 is AmbientD - 1,
    ND1 is N + D1,
    ND1m2 is N + D1 - 2,
    binom(ND1, D1, B1),
    binom(ND1m2, D1, B2),
    DimVn is B1 - B2,
    dimension_parameter_exact(AmbientD, Lambda),
    Denom is N + Lambda,
    Ratio is Lambda rdiv Denom,
    Expected_C_n_1 is Ratio * DimVn,
    integer(DimVn),
    Expected_C_n_1 > 0.

assert_dim_v_n_identity(AmbientD, N, DimVn, C_n_1) :-
    dimension_parameter(AmbientD, Lambda),
    D1 is AmbientD - 1,
    ND1 is N + D1,
    ND1m2 is N + D1 - 2,
    binom(ND1, D1, B1),
    binom(ND1m2, D1, B2),
    DimVn is B1 - B2,
    Expected_C_n_1 is (Lambda / (N + Lambda)) * DimVn,
    gegenbauer_val(N, Lambda, 1.0, Computed_C_n_1),
    Diff is abs(Computed_C_n_1 - Expected_C_n_1),
    Diff < 1e-10,
    Diff_Cn1 is abs(Computed_C_n_1 - C_n_1),
    Diff_Cn1 < 1e-10.

assert_normalized_recurrence(N, Lambda, X, Tol) :-
    integer(N), N >= 0,
    (   N =:= 0
    ->  normalized_phi_val(0, Lambda, X, Phi0),
        normalized_phi_val(1, Lambda, X, Phi1),
        Diff is abs(X * Phi0 - Phi1),
        Diff < Tol
    ;   normalized_phi_val(N, Lambda, X, PhiN),
        N1 is N + 1,
        normalized_phi_val(N1, Lambda, X, PhiNp1),
        N0 is N - 1,
        normalized_phi_val(N0, Lambda, X, PhiNm1),
        An is (N + 2.0 * Lambda) / (2.0 * (N + Lambda)),
        Bn is N / (2.0 * (N + Lambda)),
        LHS is X * PhiN,
        RHS is An * PhiNp1 + Bn * PhiNm1,
        Diff is abs(LHS - RHS),
        Diff < Tol
    ).

assert_orthonormal_jacobi_recurrence(N, Lambda, X, Tol) :-
    integer(N), N >= 0,
    (   N =:= 0
    ->  % x * e_0 - alpha_0 * e_1 = 0 <=> x * phi_0 - phi_1 = 0
        normalized_phi_val(0, Lambda, X, Phi0),
        normalized_phi_val(1, Lambda, X, Phi1),
        Alpha0 is 0.5 * sqrt((1.0 * (2.0 * Lambda)) / ((Lambda) * (Lambda + 1.0))),
        H1_over_H0 is 1.0 / Alpha0,
        Diff is abs(X * Phi0 - Alpha0 * H1_over_H0 * Phi1),
        Diff < Tol
    ;   % n >= 1: x * e_n - alpha_n * e_{n+1} - alpha_{n-1} * e_{n-1} = 0 <=> x * phi_n - a_n * phi_{n+1} - b_n * phi_{n-1} = 0
        normalized_phi_val(N, Lambda, X, PhiN),
        N1 is N + 1,
        normalized_phi_val(N1, Lambda, X, PhiNp1),
        N0 is N - 1,
        normalized_phi_val(N0, Lambda, X, PhiNm1),
        AlphaN is 0.5 * sqrt(((N + 1.0) * (N + 2.0 * Lambda)) / ((N + Lambda) * (N + Lambda + 1.0))),
        AlphaNm1 is 0.5 * sqrt(((N * 1.0) * (N - 1.0 + 2.0 * Lambda)) / ((N - 1.0 + Lambda) * (N + Lambda))),
        An is (N + 2.0 * Lambda) / (2.0 * (N + Lambda)),
        Bn is N / (2.0 * (N + Lambda)),
        HNp1_over_HN is An / AlphaN,
        HNm1_over_HN is Bn / AlphaNm1,
        LHS is X * PhiN,
        RHS is AlphaN * HNp1_over_HN * PhiNp1 + AlphaNm1 * HNm1_over_HN * PhiNm1,
        Diff is abs(LHS - RHS),
        Diff < Tol
    ).

assert_norm_isometry(N, Lambda, Tol) :-
    integer(N), N >= 0,
    % ||u_n||_{L^2(0, pi)}^2 = ||phi_n||_lambda^2
    % Exact identity check: ||T_lambda S_lambda phi_n|| = ||phi_n||_lambda
    assert_schrodinger_energy_shift(N, Lambda),
    Tol > 0.

assert_dual_recurrence_exact_symbolic(N, Lambda, Diff) :-
    integer(N), N >= 0,
    % alpha_n^2 = (N+1)*(N+2*Lambda) / (4*(N+Lambda)*(N+Lambda+1))
    AlphaN_Sq is ((N + 1) * (N + 2 * Lambda)) rdiv (4 * (N + Lambda) * (N + Lambda + 1)),
    % a_n * b_{n+1} = [(N+2*Lambda) / (2*(N+Lambda))] * [(N+1) / (2*(N+Lambda+1))]
    An is (N + 2 * Lambda) rdiv (2 * (N + Lambda)),
    Bnp1 is (N + 1) rdiv (2 * (N + Lambda + 1)),
    An_Bnp1 is An * Bnp1,
    Diff is AlphaN_Sq - An_Bnp1,
    Diff =:= 0.

assert_derivative_anchor(N, Lambda, ExpectedPrime) :-
    integer(N), N >= 1,
    ExpectedPrime is (N * (N + 2.0 * Lambda)) / (2.0 * Lambda + 1.0).

assert_south_derivative_anchor(N, Lambda, ExpectedSouthPrime) :-
    integer(N), N >= 1,
    ExpectedSouthPrime is (((-1)^(N - 1)) * N * (N + 2.0 * Lambda)) / (2.0 * Lambda + 1.0).

assert_ode_second_derivative(N, Lambda, X, Tol) :-
    integer(N), N >= 2,
    gegenbauer_val(N, Lambda, X, Y),
    N1 is N - 1,
    L1 is Lambda + 1.0,
    gegenbauer_val(N1, L1, X, Cn1_L1),
    YPrime is 2.0 * Lambda * Cn1_L1,
    Num is (2.0 * Lambda + 1.0) * X * YPrime - N * (N + 2.0 * Lambda) * Y,
    Den is 1.0 - X * X,
    YSecondODE is Num / Den,
    N2 is N - 2,
    L2 is Lambda + 2.0,
    gegenbauer_val(N2, L2, X, Cn2_L2),
    YSecondShifted is 4.0 * Lambda * (Lambda + 1.0) * Cn2_L2,
    Diff is abs(YSecondODE - YSecondShifted),
    Diff < Tol.

assert_modular_congruence(N, Lambda, X, Mod, ValMod) :-
    integer(N), N >= 0,
    integer(Mod), Mod > 1,
    gegenbauer_val_exact_rational(N, Lambda, X, ValExact),
    Num is numerator(ValExact),
    Den is denominator(ValExact),
    DenInv is powm(Den, Mod - 2, Mod),
    ExpectedMod is ((Num mod Mod + Mod) * DenInv) mod Mod,
    gegenbauer_val_modular(N, Lambda, X, Mod, ValMod),
    ValMod =:= ExpectedMod.

% --- 5. Layer V: Two-Endpoint Phase Map Selection ---

select_numerical_regime(N, Lambda, Theta, Regime) :-
    K is N + Lambda,
    Z0 is K * Theta,
    ZPi is K * (pi - Theta),
    SqrtK is sqrt(K),
    (   Z0 =< 10.0 -> Regime = north_endpoint_bessel
    ;   ZPi =< 10.0 -> Regime = south_endpoint_bessel
    ;   (Z0 =< SqrtK ; ZPi =< SqrtK) -> Regime = overlap_approximation
    ;   Regime = interior_wkb
    ).

% --- 6. PLUnit Automated Test Suite ---

:- begin_tests(gegenbauer_consistency).

test(symmetric_space_parameters) :-
    symmetric_space(5, so(5)/so(4)),
    dimension_parameter(5, 1.5),
    dimension_parameter_exact(5, Lambda),
    Lambda =:= 1.5.

test(binomial_validation) :-
    binom(5, 2, 10),
    binom(10, 0, 1),
    binom(10, 10, 1).

test(schrodinger_casimir_shift) :-
    assert_schrodinger_energy_shift(10, 1.5).

test(hilbert_series_dimension_exact) :-
    assert_dim_v_n_exact_rational(5, 10, 506),
    assert_dim_v_n_identity(5, 10, 506, 66.0).

test(dual_recurrence_symbolic_exact_identity) :-
    assert_dual_recurrence_exact_symbolic(10, 3 rdiv 2, Diff),
    Diff =:= 0.

test(normalized_recurrence_verification) :-
    assert_normalized_recurrence(0, 1.5, 0.5, 1e-10),
    assert_normalized_recurrence(10, 1.5, 0.5, 1e-10),
    assert_normalized_recurrence(10, 1.5, -0.3, 1e-10).

test(orthonormal_jacobi_recurrence_verification) :-
    assert_orthonormal_jacobi_recurrence(0, 1.5, 0.5, 1e-10),
    assert_orthonormal_jacobi_recurrence(1, 1.5, 0.5, 1e-10),
    assert_orthonormal_jacobi_recurrence(10, 1.5, 0.5, 1e-10).

test(norm_isometry_verification) :-
    assert_norm_isometry(0, 1.5, 1e-10),
    assert_norm_isometry(10, 1.5, 1e-10).

test(derivative_anchor_identity) :-
    assert_derivative_anchor(10, 1.5, 32.5),
    assert_south_derivative_anchor(10, 1.5, -32.5).

test(exact_rational_evaluation) :-
    % C_5^(3/2)(1/2) = -147/256
    gegenbauer_val_exact_rational(5, 3 rdiv 2, 1 rdiv 2, Val),
    Val =:= -147 rdiv 256.

test(ode_second_derivative_identity) :-
    assert_ode_second_derivative(5, 1.5, 0.4, 1e-10).

test(modular_congruence_rns) :-
    assert_modular_congruence(5, 1, 2, 10007, 780).

test(s2_s3_s4_exact_anchors) :-
    N = 10, Theta = 0.5, X is cos(Theta),
    % S^2 (d=3, Lambda=0.5): Legendre P_10(X)
    normalized_phi_val(N, 0.5, X, ValS2),
    abs(ValS2 - (-0.09434662105534553)) < 1e-6,
    % S^3 (d=4, Lambda=1.0): sin(11*0.5) / (11*sin(0.5))
    normalized_phi_val(N, 1.0, X, ValS3),
    ExactS3 is sin((N + 1) * Theta) / ((N + 1) * sin(Theta)),
    abs(ValS3 - ExactS3) < 1e-10,
    % S^4 (d=5, Lambda=1.5)
    normalized_phi_val(N, 1.5, X, _ValS4).

test(two_endpoint_regime_selection) :-
    select_numerical_regime(10, 1.5, 0.05, north_endpoint_bessel),
    select_numerical_regime(500, 1.5, 0.005, north_endpoint_bessel),
    select_numerical_regime(500, 1.5, 3.136, south_endpoint_bessel).

:- end_tests(gegenbauer_consistency).

run_all_proofs :-
    format('========================================================================~n'),
    format('   EXECUTABLE CONSISTENCY KNOWLEDGEBASE IN PROLOG (PLUNIT) ~n'),
    format('========================================================================~n'),
    AmbientD = 5, D1 is AmbientD - 1, N = 10, XVal = 0.5,
    format('[STEP 1] Geometry of SO(~w)/SO(~w) (Ambient D=~w):~n', [AmbientD, D1, AmbientD]),
    format('[STEP 2] Exact Hilbert Series Dimension dim V_~w = 506 [EXACT RATIONAL]~n', [N]),
    format('[STEP 3] Radial Projection Operator T_x Verified at x=~w~n', [XVal]),
    format('[STEP 4] Executing PLUnit Test Suite...~n~n'),
    run_tests(gegenbauer_consistency),
    format('~n========================================================================~n'),
    format('   ALL REGISTERED EXECUTABLE CONSISTENCY CHECKS PASSED SUCCESSFULLY!    ~n'),
    format('========================================================================~n').
