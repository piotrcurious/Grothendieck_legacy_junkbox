"""Reference Gegenbauer FIR designer for real, odd-length, Type-I linear-phase FIRs.

Scope
-----
* FIR order is even: order = 2*M; tap count = 2*M + 1.
* Real, symmetric taps: h[k] == h[2*M-k].
* Low-pass weighted-L2 design on [0, wp] and [ws, pi].
* Gegenbauer endpoint-normalized basis (lambda > 0), with the lambda -> 0
  Chebyshev-T limit supported explicitly.
* Optional DC-gain equality and Sturm-Liouville spectral regularization.
* Conversion to symmetric taps, Q-format quantization, and C header export.

Important: the optimizer is a weighted-L2 designer, not a minimax/Remez solver.
The reported grid/extremum metrics are numerical diagnostics, not a formal
continuous-frequency certificate. See the accompanying framework document.

Dependencies: Python >= 3.10, NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, lgamma, log, pi, sin, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FilterSpec:
    """Low-pass specification for an odd-length Type-I FIR.

    `order` is the polynomial/filter order (must be even), not tap count.
    `wp` and `ws` are angular frequencies in rad/sample, 0 < wp < ws < pi.
    `delta_p` and `delta_s` are validation limits; weighted-L2 optimization
    does not, by itself, guarantee these inequalities.
    """

    order: int
    wp: float
    ws: float
    lam: float = 1.5
    delta_p: float = 1.0e-3
    delta_s: float = 1.0e-3
    passband_weight: float = 1.0
    stopband_weight: float = 1.0
    regularization: float = 0.0
    spectral_power: float = 1.0
    quadrature_order: int = 256
    dc_gain: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.order, int) or self.order < 2 or self.order % 2:
            raise ValueError("order must be an even integer >= 2 for Type-I symmetry")
        if not (0.0 < self.wp < self.ws < pi):
            raise ValueError("require 0 < wp < ws < pi")
        if self.lam < 0.0:
            raise ValueError("lam must be >= 0; lam=0 selects the Chebyshev-T limit")
        if self.delta_p <= 0.0 or self.delta_s <= 0.0:
            raise ValueError("delta_p and delta_s must be positive")
        if self.passband_weight <= 0.0 or self.stopband_weight <= 0.0:
            raise ValueError("band weights must be positive")
        if self.regularization < 0.0:
            raise ValueError("regularization must be >= 0")
        if self.spectral_power <= 0.0:
            raise ValueError("spectral_power must be > 0")
        if self.quadrature_order < 16:
            raise ValueError("quadrature_order must be >= 16")


@dataclass
class FilterResult:
    spec: FilterSpec
    gegenbauer_coeffs: FloatArray  # a_n in A(x) = sum a_n phi_n(x)
    chebyshev_coeffs: FloatArray   # b_r in A(x) = sum b_r T_r(x)
    taps: FloatArray               # h[0 .. order], symmetric
    sampled_passband_error: float
    sampled_stopband_peak: float
    sampled_grid_points: int
    solver: str = "weighted_l2"
    converged: bool = True
    iterations: int = 1

    @property
    def order(self) -> int:
        return self.spec.order

    @property
    def tap_count(self) -> int:
        return self.taps.size

    def amplitude(self, omega: FloatArray | float) -> FloatArray:
        """Evaluate real zero-phase amplitude A(cos(omega))."""
        w = np.asarray(omega, dtype=np.float64)
        x = np.cos(w)
        basis = gegenbauer_basis_matrix(x, self.order // 2, self.spec.lam)
        return basis @ self.gegenbauer_coeffs

    def response(self, omega: FloatArray | float) -> NDArray[np.complex128]:
        """Evaluate H(exp(jw)) = exp(-j*M*w) A(cos(w))."""
        w = np.asarray(omega, dtype=np.float64)
        return np.exp(-1j * (self.order // 2) * w) * self.amplitude(w)

    def report(self) -> str:
        return (
            f"Type-I FIR: order={self.order}, taps={self.tap_count}, lam={self.spec.lam:g}\n"
            f"sampled passband max |A-1| = {self.sampled_passband_error:.8g} "
            f"(limit {self.spec.delta_p:.8g})\n"
            f"sampled stopband max |A|   = {self.sampled_stopband_peak:.8g} "
            f"(limit {self.spec.delta_s:.8g})\n"
            f"solver = {self.solver}; converged = {self.converged}; iterations = {self.iterations}\n"
            f"grid points = {self.sampled_grid_points}; these are numerical diagnostics, "
            "not a formal continuous-band proof."
        )


def gegenbauer_basis_matrix(x: FloatArray, degree: int, lam: float) -> FloatArray:
    """Return columns phi_n(x), n=0..degree, with phi_n(1)=1.

    For lam=0, use the continuous normalized Gegenbauer limit T_n(x).
    For lam>0, phi_n = C_n^(lam)(x) / C_n^(lam)(1).
    The three-term recurrence avoids monomial expansion.
    """
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    if lam < 0.0:
        raise ValueError("lam must be >= 0")
    out = np.empty((xx.size, degree + 1), dtype=np.float64)
    out[:, 0] = 1.0
    if degree == 0:
        return out
    out[:, 1] = xx
    if lam == 0.0:
        for n in range(1, degree):
            out[:, n + 1] = 2.0 * xx * out[:, n] - out[:, n - 1]
    else:
        for n in range(1, degree):
            out[:, n + 1] = (
                (2.0 * (n + lam) / (n + 2.0 * lam)) * xx * out[:, n]
                - (n / (n + 2.0 * lam)) * out[:, n - 1]
            )
    return out


def _normalized_basis_norms(degree: int, lam: float) -> FloatArray:
    """g_n = integral_{-1}^1 phi_n(x)^2 (1-x^2)^(lam-1/2) dx.

    The result is divided by g_0. This common scaling only changes the
    numerical scale of the regularization parameter, not its eigenspaces.
    """
    if lam == 0.0:
        g = np.full(degree + 1, 0.5, dtype=np.float64)
        g[0] = 1.0
        return g
    if lam <= 0.0:
        raise ValueError("Gegenbauer norm formula requires lam > 0")

    n = np.arange(degree + 1, dtype=np.float64)
    log_h = (
        (1.0 - 2.0 * lam) * log(2.0)
        + log(pi)
        + np.array([lgamma(float(k + 2.0 * lam)) for k in n])
        - np.log(n + lam)
        - np.array([lgamma(float(k + 1.0)) for k in n])
        - 2.0 * lgamma(lam)
    )
    log_c1 = (
        np.array([lgamma(float(k + 2.0 * lam)) for k in n])
        - lgamma(2.0 * lam)
        - np.array([lgamma(float(k + 1.0)) for k in n])
    )
    log_g = log_h - 2.0 * log_c1
    log_g -= log_g[0]
    return np.exp(log_g)


def _spectral_penalty_diagonal(degree: int, lam: float, power: float) -> FloatArray:
    """Diagonal of ||L_lam^(power/2) A||^2 in endpoint-normalized basis."""
    n = np.arange(degree + 1, dtype=np.float64)
    rho = n * (n + 2.0 * lam)
    g = _normalized_basis_norms(degree, lam)
    return np.power(rho, power) * g


def _gauss_legendre_interval(a: float, b: float, count: int) -> tuple[FloatArray, FloatArray]:
    z, w = np.polynomial.legendre.leggauss(count)
    omega = 0.5 * (b - a) * z + 0.5 * (a + b)
    weights = 0.5 * (b - a) * w
    return omega.astype(np.float64), weights.astype(np.float64)


def _chebyshev_coeffs_from_values(values_at_lobatto: FloatArray) -> FloatArray:
    """DCT-I interpolation on x_j=cos(pi*j/M), with endpoint halving."""
    y = np.asarray(values_at_lobatto, dtype=np.float64)
    m = y.size - 1
    if m == 0:
        return y.copy()
    j = np.arange(m + 1, dtype=np.float64)
    b = np.empty(m + 1, dtype=np.float64)
    for k in range(m + 1):
        terms = y[1:m] * np.cos(pi * k * j[1:m] / m)
        s = y[0] + ((-1.0) ** k) * y[m] + 2.0 * np.sum(terms)
        b[k] = s / m
    b[0] *= 0.5
    b[m] *= 0.5
    return b


def _gegenbauer_to_chebyshev(a: FloatArray, lam: float) -> FloatArray:
    """Convert sum a_n phi_n^(lam) to Chebyshev coefficients b_r.

    Uses exact three-term basis recurrence in Chebyshev coordinates; no
    monomial expansion is formed.
    """
    coeffs = np.asarray(a, dtype=np.float64)
    degree = coeffs.size - 1
    if lam == 0.0:
        return coeffs.copy()

    phi_prev = np.array([1.0], dtype=np.float64)  # phi_0 = T_0
    total = np.zeros(degree + 1, dtype=np.float64)
    total[:1] += coeffs[0] * phi_prev
    if degree == 0:
        return total

    phi_cur = np.array([0.0, 1.0], dtype=np.float64)  # phi_1 = x = T_1
    total[:2] += coeffs[1] * phi_cur

    for n in range(1, degree):
        # Multiply phi_n by x, using x*T_0=T_1 and
        # x*T_k=(T_(k+1)+T_(k-1))/2 for k>=1.
        xphi = np.zeros(n + 2, dtype=np.float64)
        for k, value in enumerate(phi_cur):
            if k == 0:
                xphi[1] += value
            else:
                xphi[k - 1] += 0.5 * value
                xphi[k + 1] += 0.5 * value

        alpha = (n + 2.0 * lam) / (2.0 * (n + lam))
        beta = n / (2.0 * (n + lam))
        phi_next = (xphi - beta * np.pad(phi_prev, (0, xphi.size - phi_prev.size))) / alpha
        total[: phi_next.size] += coeffs[n + 1] * phi_next
        phi_prev, phi_cur = phi_cur, phi_next

    return total


def _taps_from_chebyshev(b: FloatArray) -> FloatArray:
    degree = b.size - 1
    taps = np.zeros(2 * degree + 1, dtype=np.float64)
    taps[degree] = b[0]
    for r in range(1, degree + 1):
        taps[degree - r] = 0.5 * b[r]
        taps[degree + r] = 0.5 * b[r]
    return taps


def design_lowpass(spec: FilterSpec, diagnostic_grid: int = 131073) -> FilterResult:
    """Design a weighted-L2 low-pass Type-I FIR in Gegenbauer coordinates."""
    degree = spec.order // 2

    wp, qwp = _gauss_legendre_interval(0.0, spec.wp, spec.quadrature_order)
    ws, qws = _gauss_legendre_interval(spec.ws, pi, spec.quadrature_order)

    xp = np.cos(wp)
    xs = np.cos(ws)
    Bp = gegenbauer_basis_matrix(xp, degree, spec.lam)
    Bs = gegenbauer_basis_matrix(xs, degree, spec.lam)

    Cp = np.sqrt(qwp * spec.passband_weight)[:, None] * Bp
    Cs = np.sqrt(qws * spec.stopband_weight)[:, None] * Bs
    C = np.vstack((Cp, Cs))
    y = np.concatenate((np.sqrt(qwp * spec.passband_weight), np.zeros_like(qws)))

    if spec.regularization > 0.0:
        rdiag = _spectral_penalty_diagonal(degree, spec.lam, spec.spectral_power)
        Rrows = np.diag(np.sqrt(spec.regularization * rdiag))
        C = np.vstack((C, Rrows))
        y = np.concatenate((y, np.zeros(degree + 1, dtype=np.float64)))

    if spec.dc_gain is None:
        a, *_ = np.linalg.lstsq(C, y, rcond=None)
    else:
        # Exact equality sum(a_n)=A(1)=dc_gain, solved through an SVD
        # nullspace so no penalty or finite weight can relax the constraint.
        e = np.ones((1, degree + 1), dtype=np.float64)
        _, _, vh = np.linalg.svd(e, full_matrices=True)
        Z = vh[1:, :].T
        a0 = np.full(degree + 1, spec.dc_gain / (degree + 1), dtype=np.float64)
        if Z.shape[1] == 0:
            a = a0
        else:
            z, *_ = np.linalg.lstsq(C @ Z, y - C @ a0, rcond=None)
            a = a0 + Z @ z

    b = _gegenbauer_to_chebyshev(a, spec.lam)
    taps = _taps_from_chebyshev(b)

    # Dense-grid diagnostics. These are intentionally labelled sampled.
    grid = np.linspace(0.0, pi, diagnostic_grid, dtype=np.float64)
    amplitude = np.polynomial.chebyshev.chebval(np.cos(grid), b)
    p_mask = grid <= spec.wp
    s_mask = grid >= spec.ws
    pass_error = float(np.max(np.abs(amplitude[p_mask] - 1.0)))
    stop_peak = float(np.max(np.abs(amplitude[s_mask])))

    return FilterResult(
        spec=spec,
        gegenbauer_coeffs=np.asarray(a, dtype=np.float64),
        chebyshev_coeffs=np.asarray(b, dtype=np.float64),
        taps=taps,
        sampled_passband_error=pass_error,
        sampled_stopband_peak=stop_peak,
        sampled_grid_points=diagnostic_grid,
        solver="weighted_l2",
        converged=True,
        iterations=1,
    )


def quantize_symmetric_taps(taps: FloatArray, frac_bits: int, total_bits: int = 32) -> IntArray:
    """Quantize symmetric taps to signed fixed-point integers.

    Symmetry is imposed by quantizing one half and mirroring the integer
    values. Raises on overflow rather than silently clipping coefficients.
    """
    h = np.asarray(taps, dtype=np.float64)
    if h.ndim != 1 or h.size < 3 or h.size % 2 != 1:
        raise ValueError("taps must be a one-dimensional odd-length array")
    if not np.allclose(h, h[::-1], rtol=0.0, atol=1e-12):
        raise ValueError("input taps are not symmetric")
    if not (1 <= total_bits <= 63):
        raise ValueError("total_bits must be in [1, 63]")
    if not (0 <= frac_bits < total_bits):
        raise ValueError("frac_bits must be in [0, total_bits-1]")

    scale = float(1 << frac_bits)
    lo = -(1 << (total_bits - 1))
    hi = (1 << (total_bits - 1)) - 1
    m = h.size // 2
    rounded_left = np.rint(h[:m] * scale)
    rounded_center = float(np.rint(h[m] * scale))
    if (not np.all(np.isfinite(rounded_left)) or not np.isfinite(rounded_center)
            or np.any(rounded_left < lo) or np.any(rounded_left > hi)
            or rounded_center < lo or rounded_center > hi):
        raise OverflowError("quantized coefficient exceeds signed fixed-point range")
    left = rounded_left.astype(np.int64)
    center = int(rounded_center)
    q = np.empty(h.size, dtype=np.int64)
    q[:m] = left
    q[m] = center
    q[m + 1 :] = left[::-1]
    return q


def c_header(
    q_taps: IntArray,
    frac_bits: int,
    symbol: str = "gegenbauer_fir_taps",
    c_type: str = "int32_t",
) -> str:
    """Generate a portable C header containing symmetric fixed-point taps."""
    q = np.asarray(q_taps, dtype=np.int64)
    if q.ndim != 1 or q.size % 2 != 1:
        raise ValueError("q_taps must be a one-dimensional odd-length array")
    if not symbol.isidentifier():
        raise ValueError("symbol must be a valid C identifier")
    if not np.array_equal(q, q[::-1]):
        raise ValueError("q_taps must be exactly symmetric")
    lines = [
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define {symbol.upper()}_COUNT {q.size}",
        f"#define {symbol.upper()}_FRAC_BITS {frac_bits}",
        f"static const {c_type} {symbol}[{q.size}] = {{",
    ]
    for i in range(0, q.size, 4):
        chunk = ", ".join(str(int(v)) for v in q[i : i + 4])
        suffix = "," if i + 4 < q.size else ""
        lines.append(f"    {chunk}{suffix}")
    lines.extend(["};", ""])
    return "\n".join(lines)


def write_c_header(path: str | Path, q_taps: IntArray, frac_bits: int,
                   symbol: str = "gegenbauer_fir_taps", c_type: str = "int32_t") -> None:
    Path(path).write_text(c_header(q_taps, frac_bits, symbol, c_type), encoding="utf-8")


def numerical_stationary_point_diagnostics(result: FilterResult) -> dict[str, FloatArray | float]:
    """Return candidate continuous extrema from numerical Chebyshev roots.

    This is stronger than a uniform grid but is still not a formal certificate:
    polynomial roots are computed in floating point and may be ill-conditioned.
    """
    b = result.chebyshev_coeffs
    db = np.polynomial.chebyshev.chebder(b)
    if db.size <= 1 or np.all(db == 0.0):
        roots = np.empty(0, dtype=np.float64)
    else:
        rr = np.polynomial.chebyshev.chebroots(db)
        roots = np.array(
            [z.real for z in rr if abs(z.imag) <= 1e-9 and -1.0 <= z.real <= 1.0],
            dtype=np.float64,
        )
    xp = float(cos(result.spec.wp))
    xs = float(cos(result.spec.ws))
    p_roots = roots[(roots >= xp) & (roots <= 1.0)]
    s_roots = roots[(roots >= -1.0) & (roots <= xs)]
    p_candidates = np.concatenate(([xp, 1.0], p_roots))
    s_candidates = np.concatenate(([-1.0, xs], s_roots))
    p_vals = np.polynomial.chebyshev.chebval(p_candidates, b)
    s_vals = np.polynomial.chebyshev.chebval(s_candidates, b)
    return {
        "passband_candidate_x": p_candidates,
        "passband_candidate_A": p_vals,
        "stopband_candidate_x": s_candidates,
        "stopband_candidate_A": s_vals,
        "passband_candidate_error": float(np.max(np.abs(p_vals - 1.0))),
        "stopband_candidate_peak": float(np.max(np.abs(s_vals))),
    }


if __name__ == "__main__":
    # A reproducible smoke test; it is not intended to guarantee the requested
    # ripple limits (weighted-L2 and minimax are different optimization goals).
    demo = FilterSpec(
        order=64,
        wp=0.20 * pi,
        ws=0.25 * pi,
        lam=1.5,
        delta_p=1e-3,
        delta_s=1e-3,
        passband_weight=1.0,
        stopband_weight=5.0,
        regularization=0.0,
        quadrature_order=256,
        dc_gain=1.0,
    )
    result = design_lowpass(demo)
    print(result.report())
    print("tap symmetry max error:", float(np.max(np.abs(result.taps - result.taps[::-1]))))
    print("DC gain:", float(np.sum(result.taps)))
    print("stationary-point diagnostics:", numerical_stationary_point_diagnostics(result))


def _alternation_candidates(
    b: FloatArray, spec: FilterSpec
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Find endpoints and numerically computed stationary points in both bands."""
    degree = b.size - 1
    xp = float(cos(spec.wp))
    xs = float(cos(spec.ws))
    db = np.polynomial.chebyshev.chebder(b)
    if db.size <= 1 or np.allclose(db, 0.0):
        roots = np.empty(0, dtype=np.float64)
    else:
        rr = np.polynomial.chebyshev.chebroots(db)
        roots = np.array(
            [z.real for z in rr if abs(z.imag) <= 1e-8 and -1.0 < z.real < 1.0],
            dtype=np.float64,
        )

    pass_x = np.unique(np.concatenate(([xp, 1.0], roots[(roots > xp) & (roots < 1.0)])))[::-1]
    stop_x = np.unique(np.concatenate(([-1.0, xs], roots[(roots > -1.0) & (roots < xs)])))[::-1]
    pass_w = np.arccos(np.clip(pass_x, -1.0, 1.0))
    stop_w = np.arccos(np.clip(stop_x, -1.0, 1.0))
    omega = np.concatenate((pass_w, stop_w))
    x_all = np.cos(omega)
    target = np.concatenate((np.ones(pass_w.size), np.zeros(stop_w.size)))
    weight = np.concatenate((
        np.full(pass_w.size, spec.passband_weight),
        np.full(stop_w.size, spec.stopband_weight),
    ))
    order = np.argsort(omega)
    omega, x_all, target, weight = omega[order], x_all[order], target[order], weight[order]
    A = np.polynomial.chebyshev.chebval(x_all, b)
    error = weight * (A - target)

    # Remove duplicate frequencies caused by roundoff at shared endpoints.
    keep = np.ones(omega.size, dtype=bool)
    if omega.size > 1:
        keep[1:] = np.diff(omega) > 1e-10
    return omega[keep], x_all[keep], error[keep], weight[keep]


def _choose_alternating_extrema(
    omega: FloatArray, x: FloatArray, error: FloatArray, weights: FloatArray, count: int
) -> tuple[FloatArray, FloatArray, FloatArray] | None:
    """Choose count alternating extrema with the largest minimum absolute error.

    For a degree-M polynomial on two disjoint bands, the candidate set consists
    of at most M-1 interior stationary points plus four band endpoints; hence
    it is normally only one point larger than the required M+2 alternants.
    The combinatorial selection is therefore small for this specific problem.
    """
    import itertools

    if omega.size < count:
        return None
    best = None
    best_score = -np.inf
    # Guard against pathological numeric root duplication creating a huge set.
    if omega.size > count + 12:
        # Greedy sign compression as a safe fallback.
        idx = list(range(omega.size))
        compressed: list[int] = []
        for i in idx:
            if not compressed or np.sign(error[i]) != np.sign(error[compressed[-1]]):
                compressed.append(i)
            elif abs(error[i]) > abs(error[compressed[-1]]):
                compressed[-1] = i
        if len(compressed) < count:
            return None
        # Retain the count strongest candidates, then restore order.
        compressed = sorted(compressed, key=lambda i: abs(error[i]), reverse=True)[:count]
        compressed.sort()
        if any(np.sign(error[compressed[i]]) == np.sign(error[compressed[i - 1]])
               for i in range(1, count)):
            return None
        return omega[compressed], x[compressed], weights[compressed]

    for ids in itertools.combinations(range(omega.size), count):
        signs = np.sign(error[list(ids)])
        if np.any(signs == 0.0) or np.any(signs[1:] == signs[:-1]):
            continue
        vals = np.abs(error[list(ids)])
        score = float(np.min(vals))
        if score > best_score:
            best_score = score
            ii = np.asarray(ids, dtype=np.int64)
            best = (omega[ii], x[ii], weights[ii])
    return best


def design_minimax(spec: FilterSpec, max_iterations: int = 80,
                   tolerance: float = 1e-9) -> FilterResult:
    """Numerical Remez-exchange design in the Gegenbauer coordinate basis.

    Solves the weighted minimax problem on [0,wp] union [ws,pi] for a Type-I
    FIR. This implementation leaves DC unconstrained; set `dc_gain=None`.
    An exact DC equality is supported by the weighted-L2 solver, not by this
    unconstrained Remez exchange implementation.

    The solution is basis-independent for fixed order; lambda affects only the
    coordinates and numerical conditioning. This unconstrained Remez variant
    does not impose an exact DC equality; no spectral regularizer is allowed
    because that would change the minimax objective.

    Root extraction is floating-point, so this is a numerical exchange solver,
    not a certified exact-arithmetic implementation of Remez.
    """
    if spec.regularization != 0.0:
        raise ValueError("minimax/Remez objective cannot include this L2 spectral regularizer")
    if spec.dc_gain is not None:
        raise ValueError("for unconstrained minimax/Remez design set dc_gain=None")
    degree = spec.order // 2
    a0 = np.zeros(degree + 1, dtype=np.float64)
    Z = np.eye(degree + 1, dtype=np.float64)

    free_dimension = Z.shape[1]
    count = free_dimension + 1  # polynomial degrees of freedom plus ripple t
    total_measure = spec.wp + (pi - spec.ws)
    u = 0.5 * (1.0 - np.cos(np.linspace(0.0, pi, count)))
    cumulative = total_measure * u
    omega_ext = np.where(
        cumulative <= spec.wp,
        cumulative,
        spec.ws + (cumulative - spec.wp),
    )
    signs = np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    last_t = np.nan
    converged = False
    iterations = 0

    for _iteration in range(max_iterations):
        iterations = _iteration + 1
        x_ext = np.cos(omega_ext)
        basis = gegenbauer_basis_matrix(x_ext, degree, spec.lam)
        target = np.where(omega_ext <= spec.wp + 1e-14, 1.0, 0.0)
        weights = np.where(
            omega_ext <= spec.wp + 1e-14,
            spec.passband_weight,
            spec.stopband_weight,
        )
        # E_i = W_i(A(x_i)-D_i) = s_i*t, with A=a0+Z*z.
        reduced_basis = basis @ Z
        system = np.column_stack((reduced_basis, -signs / weights))
        rhs = target - basis @ a0
        try:
            solution = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError:
            solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
        z = solution[:-1]
        a = a0 + Z @ z
        t = float(solution[-1])
        b = _gegenbauer_to_chebyshev(a, spec.lam)

        cand_w, cand_x, cand_e, cand_weights = _alternation_candidates(b, spec)
        selected = _choose_alternating_extrema(cand_w, cand_x, cand_e, cand_weights, count)
        if selected is None:
            if np.isfinite(last_t) and abs(abs(t) - abs(last_t)) <= tolerance * max(1.0, abs(t)):
                break
            last_t = t
            continue
        new_w, new_x, new_weights = selected
        new_target = np.where(new_w <= spec.wp + 1e-14, 1.0, 0.0)
        new_error = new_weights * (
            np.polynomial.chebyshev.chebval(new_x, b) - new_target
        )
        ripple = np.abs(new_error)
        ripple_spread = float(np.max(ripple) - np.min(ripple))
        stable = new_w.size == omega_ext.size and np.max(np.abs(new_w - omega_ext)) <= 1e-10
        omega_ext = new_w
        signs = np.where(np.sign(new_error) == 0.0, signs, np.sign(new_error))
        if ripple_spread <= tolerance * max(1.0, float(np.max(ripple))):
            converged = True
            break
        if np.isfinite(last_t) and abs(abs(t) - abs(last_t)) <= tolerance * max(1.0, abs(t)) and stable:
            converged = True
            break
        last_t = t

    # Re-solve once on the final exchange set.
    x_ext = np.cos(omega_ext)
    basis = gegenbauer_basis_matrix(x_ext, degree, spec.lam)
    target = np.where(omega_ext <= spec.wp + 1e-14, 1.0, 0.0)
    weights = np.where(omega_ext <= spec.wp + 1e-14,
                       spec.passband_weight, spec.stopband_weight)
    system = np.column_stack((basis @ Z, -signs / weights))
    rhs = target - basis @ a0
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        solution, *_ = np.linalg.lstsq(system, rhs, rcond=None)
    a = a0 + Z @ solution[:-1]
    b = _gegenbauer_to_chebyshev(a, spec.lam)
    taps = _taps_from_chebyshev(b)

    grid = np.linspace(0.0, pi, 131073, dtype=np.float64)
    amp = np.polynomial.chebyshev.chebval(np.cos(grid), b)
    p_mask = grid <= spec.wp
    s_mask = grid >= spec.ws
    return FilterResult(
        spec=spec,
        gegenbauer_coeffs=np.asarray(a, dtype=np.float64),
        chebyshev_coeffs=np.asarray(b, dtype=np.float64),
        taps=taps,
        sampled_passband_error=float(np.max(np.abs(amp[p_mask] - 1.0))),
        sampled_stopband_peak=float(np.max(np.abs(amp[s_mask]))),
        sampled_grid_points=grid.size,
        solver="remez_exchange",
        converged=converged,
        iterations=iterations,
    )

