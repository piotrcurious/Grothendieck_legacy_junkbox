# 2_repair.md: Computational Layer and Pareto Optimization Solver Repair Plan

Verdict

The module has a good framework shape, but in its current form I would not trust the benchmark or the Pareto result. There are several P0/P1 issues where the code either evaluates a different function than the stated Gegenbauer zonal function, or claims to model numerical representations that it does not actually implement.

The most important defect is:

> evaluate_quotient_ring_normal_form() does not produce the Gegenbauer zonal polynomial. It appears to produce \(x^n\) (or a quotient-ring equivalent of \(z_1^n\)), followed by no normalization/projection at all.



That alone invalidates comparisons involving that permutation.


---

1. Critical mathematical correctness issues

P0 — Quotient-ring evaluator is not a Gegenbauer evaluator

You construct

poly = QuadricQuotientPolynomial(d, {(0,) * d: 1.0})
for _ in range(self.n):
    poly = poly.multiply_by_x(0)

Starting from \(1\), this is simply

\[
z_1^n \pmod{z_1^2+\cdots+z_d^2-1}.
\]

But on your chosen point

\[
(z_1,\ldots,z_d)
=
\left(x,\sqrt{\frac{1-x^2}{d-1}},\ldots\right),
\]

this evaluates to

\[
z_1^n=x^n.
\]

The Gegenbauer zonal function is

\[
\phi_n(x)
=
\frac{C_n^{(\lambda)}(x)}
{C_n^{(\lambda)}(1)},
\qquad \lambda=\frac{d-2}{2}.
\]

For example,

\[
\phi_2(x)
=
\frac{2(\lambda+1)x^2-1}{2\lambda+1},
\]

which plainly is not \(x^2\).

So this line:

c_n_1 = c_n_1_val(...)
return self._apply_context_quantization(out)

does not actually normalize anything. You calculate c_n_1, then never use it.

What the quotient-ring layer should be doing

If your algebraic-geometry framework is based on the sphere/quadric quotient, you need the harmonic/spherical projection of \(z_1^n\), not merely the quotient-ring residue of \(z_1^n\).

Equivalently, the relevant object is the degree-\(n\) harmonic component whose restriction to the sphere is proportional to \(C_n^{(\lambda)}(x)\).

That distinction is fundamental:

\[
\text{polynomial quotient}
\neq
\text{harmonic projection}.
\]

This is also exactly where your earlier algebraic-geometry direction becomes useful: the quotient-ring representation should eliminate redundant coordinates/relations, but you still need the representation-theoretic projection onto the irreducible harmonic component.


---

P0 — "Numerical bases" are mostly labels, not implementations

You advertise:

Base 2

Base 10

fixed point

LNS

float32

float64

float128

mpmath


but only a tiny subset is actually simulated.

For example:

elif self.context.precision == PrecisionType.FLOAT32:
    return arr.astype(np.float32).astype(np.float64)
return arr

There is no FLOAT128 branch.

Therefore:

PrecisionType.FLOAT128

behaves exactly like float64.

Likewise:

PrecisionType.ARBITRARY

does nothing whatsoever. mpmath is imported but never used.

This is particularly dangerous because the benchmark output would make it look as though the solver has tested these numerical environments.

Recommended semantics

Treat these as distinct execution backends rather than a single np.ndarray plus post-hoc noise:

FLOAT32      -> actual float32 arithmetic
FLOAT64      -> actual float64 arithmetic
LONGDOUBLE   -> actual np.longdouble where supported
MPMATH       -> actual mp.mpf arithmetic
FIXED_POINT  -> integer representation + explicit scaling
LNS          -> actual log-domain arithmetic

Otherwise rename the feature to something like "quantization/error injection model".


---

2. The LNS model is mathematically wrong

This:

noise = 1.0 + np.random.normal(0, self.context.eps, size=arr.shape)
return arr * noise

is not a logarithmic number system.

It is multiplicative random perturbation.

An LNS stores something conceptually like

\[
x = s\,b^\ell
\]

and transforms multiplication/division into addition/subtraction in log space, while addition requires a special log-sum-exp-like operation.

Your model is therefore:

LNS != multiplicative noise

and it also introduces stochastic benchmark results.

Two runs can produce different Pareto fronts.

For a numerical-method benchmark, that is a major reproducibility problem.

If you want a cheap stochastic perturbation model, call it that.


---

3. Fixed-point is also not actually fixed-point

This:

scale = 65536.0
return np.round(arr * scale) / scale

models quantization, but not fixed-point arithmetic.

It omits:

integer representation,

overflow,

saturation vs wraparound,

intermediate rounding,

multiplication widening,

division behavior,

signed range.


More importantly, quantization is applied only to:

input
↓
algorithm running largely in float64
↓
output

instead of to the arithmetic inside the algorithm.

For numerical stability analysis that difference can be enormous.


---

4. The mpmath backend is completely disconnected

You have:

try:
    import mpmath

but nowhere do you evaluate with it.

Consequently:

NumericalContext.mpmath_arbitrary(100)

is not a 100-digit calculation.

The solver still feeds NumPy/scipy double-precision machinery.

This is especially problematic for the ground truth:

ground_truth = eval_gegenbauer(...)

which is scipy's double precision evaluation regardless of context.


---

5. Ground truth is not ground truth for this purpose

You define:

ground_truth = eval_gegenbauer(
    self.n,
    self.lambda_val,
    domain_x
) / c_n_1

This is a reasonable double-precision reference, but it is not a high-precision reference.

For an error-analysis framework the hierarchy should be:

production computation
        ↓
independent high-precision reference
        ↓
error

not:

production computation
        ↓
SciPy float64 reference

Otherwise any shared numerical pathology can disappear from your error measurement.

For example, when n becomes large, or \(\lambda\) is large, or \(x\) approaches ±1, the reference itself can become the limiting factor.

A much better architecture is:

reference = mpmath evaluation at 100–300 bits

with the precision selected according to (n, lambda, x).


---

6. Your hypergeometric evaluation simplifies unnecessarily

You have:

c_n_1 = c_n_1_val(self.n, self.lambda_val)
h_val = hyp2f1(
    -self.n,
    self.n + 2.0 * self.lambda_val,
    self.lambda_val + 0.5,
    z
)
return self._apply_context_quantization((c_n_1 * h_val) / c_n_1)

The c_n_1 factors cancel.

For the normalized Gegenbauer polynomial,

\[
\frac{C_n^{(\lambda)}(x)}
{C_n^{(\lambda)}(1)}
=
{}_2F_1\!\left(
-n,n+2\lambda;
\lambda+\frac12;
\frac{1-x}{2}
\right).
\]

So this should simply be

return self._apply_context_quantization(h_val)

assuming c_n_1_val() indeed means \(C_n^{(\lambda)}(1)\).

The current version adds unnecessary floating-point operations and potential overflow/underflow.

For large parameters, cancelling two potentially huge numbers numerically is especially undesirable.


---

7. "WKB / Weyl" and "Mehler-Heine" are not globally equivalent methods

Your six entries are presented as interchangeable algebraic permutations:

recurrence
quotient ring
hypergeometric
WKB
Mehler-Heine
composite

but they do not all have the same domain of validity.

That distinction is essential.

Typical regimes

For \(x=\cos\theta\):

recurrence: broad/global, subject to numerical stability;

hypergeometric: exact mathematically, but not necessarily numerically stable everywhere;

interior WKB: \(\theta\) bounded away from \(0,\pi\);

Mehler-Heine: endpoint scaling, e.g. \(\theta=O(1/n)\);

composite: only meaningful if the matching construction is actually asymptotically valid.


Thus a Pareto solver shouldn't merely compare them over an arbitrary domain_x.

It should classify the point into a regime:

\[
(n,\lambda,x)
\rightarrow
\{\text{bulk},\text{endpoint},\text{transition},\text{etc.}\}.
\]

Then compare only formulas whose validity conditions are satisfied.

Otherwise a method can win simply because your chosen interval happens to favor it.


---

8. The benchmark interval hides the most interesting numerical regimes

Your demo uses:

domain = np.linspace(0.5, 0.999, 500)

This is heavily biased toward one endpoint and excludes:

x < 0
x ≈ 0
x ≈ -1
exact endpoints
oscillatory symmetry tests
endpoint transition scales

For Gegenbauer numerics I'd split the benchmark into explicit regimes, for example:

bulk:
    x ∈ [-0.8, 0.8]

north endpoint:
    1-x ∼ O(n^-2)

south endpoint:
    1+x ∼ O(n^-2)

transition:
    progressively scaled endpoint windows

global:
    dense grid on [-1,1]

Even better, parameterize the sampling in \(\theta\), because the asymptotics are naturally expressed there.


---

9. np.clip(..., -0.999999, 0.999999) silently changes the problem

You do:

x_q = self._apply_context_quantization(
    np.clip(x, -0.999999, 0.999999)
)

for WKB.

So if the input contains

x = 1.0

you don't evaluate at \(1\).

You evaluate at \(0.999999\).

The error then includes an artificial domain modification.

That is especially bad for endpoint comparisons, because the endpoint is precisely where you are claiming to use specialized asymptotics.

A safer implementation explicitly declares:

method valid for |x| < 1

and marks endpoint evaluations as invalid / not applicable rather than silently modifying them.


---

10. FLOP estimates are not credible

For example:

return 20 * num_points

for hypergeometric evaluation.

But scipy.special.hyp2f1() is a highly nontrivial numerical routine whose cost is not remotely "20 FLOPs per point".

Likewise:

return 12 * num_points

for WKB, despite involving:

arccos

asymptotic coefficient machinery

trigonometric operations

possibly special-function calculations.


A library call should not be represented by an arbitrary tiny FLOP constant.

Better metric

Measure actual:

wall-clock time;

evaluations/second;

allocations;

optionally hardware performance counters.


If you specifically want symbolic-operation cost, distinguish:

algorithmic arithmetic cost
library kernel cost
measured execution cost

instead of treating them as the same quantity.


---

11. FLOPs and eps are combined into a meaningless error estimate

This:

estimated_error = mean_mix_err + (num_flops * self.context.eps)

doesn't have a defensible numerical-analysis interpretation.

Error does not generally grow as

\[
F\epsilon.
\]

A more realistic first-order model is something like

\[
|\Delta y|
\lesssim
\kappa(x)\,\gamma_k
\]

where

\[
\gamma_k =
\frac{k\epsilon}{1-k\epsilon}
\]

for a suitably defined sequence of floating-point operations, together with algorithm-specific conditioning.

Even that is only a model, not a universal law.

And you're then reporting:

estimated_error

but Pareto selection does not use it.

So currently it is dead information.


---

12. Your mixed_error() metric is useful, but the defaults need reconsideration

You have:

|approx-ref| / (atol + rtol*|ref|)

This is a sensible practical metric.

But:

atol=1e-14
rtol=1e-10

creates a very specific scale.

For a root where ref ≈ 0, an absolute error of \(10^{-12}\) produces

\[
E \approx 100.
\]

That may be intended, but then statements like:

max_error_tol=1e-2

become extremely stringent.

More importantly, max mixed error can be dominated by a single root sample.

You currently compute both:

max_mix_err
mean_mix_err

but use only the max for Pareto selection.

That makes the solver effectively optimize:

> worst sampled point.



That is a legitimate choice, but it should be explicit.

For numerical approximation I would report at least:

max error
95th percentile
median
RMS
endpoint max
bulk max

because otherwise one root can dominate the entire optimization.


---

13. Your Pareto frontier is based on one cost measure but the stated problem has two

The module description says:

> 2D Optimization (Computational Cost) vs Numerical Error



but SolverPerformanceMetrics contains:

num_flops
exec_time_sec

and Pareto uses only:

num_flops
max_mixed_error

So actual execution time is not part of the Pareto plane.

For your stated purpose this is particularly unfortunate because the FLOP estimates are the least trustworthy quantity in the module.

I would use:

\[
(\text{measured latency},\text{error})
\]

as the primary empirical frontier, and optionally calculate a second theoretical frontier using symbolic operation count.


---

14. The constraint logic silently violates the user's constraint

This is subtle and important.

You do:

filtered = [
    m for m in pareto_candidates
    if m.max_mixed_error <= max_error_tol
]
if filtered:
    pareto_candidates = filtered

If nothing satisfies the constraint, you simply keep the original candidates.

So:

solve(..., max_error_tol=1e-30)

can return a solution whose error is 1e5.

That violates the requested constraint.

Same problem here:

if filtered:
    pareto_candidates = filtered

for the FLOP budget.

A constrained optimizer needs three possible states:

feasible candidate exists -> choose among them
no feasible candidate       -> report infeasible

not:

no feasible candidate -> silently ignore constraint

I would make this a hard failure:

raise ValueError("No permutation satisfies max_error_tol")

or return an explicit infeasible result.


---

15. The "optimal" selection rule is inconsistent

You say:

if max_error_tol is not None:
    best = min(pareto_candidates, key=lambda m: m.num_flops)
else:
    best = min(pareto_candidates, key=lambda m: m.max_mixed_error)

This means:

without tolerance

minimize error.

with tolerance

minimize FLOPs.

That's reasonable as a constrained optimization problem.

But the actual implementation ignores:

exec_time_sec
estimated_error

and the name speed/accuracy constraints suggests something richer.

A cleaner formulation is:

\[
\min C(p)
\]

subject to

\[
E(p)\le E_{\max}, \qquad C(p)\le C_{\max}.
\]

Then explicitly define \(C\) as either latency or operation count.


---

16. n == 0 handling is inconsistent

Hypergeometric:

c_n_1 = c_n_1_val(self.n, ...)

while quotient-ring uses:

if self.n > 0 else 1.0

You need one consistent definition for the constant polynomial.

For \(n=0\),

\[
\phi_0(x)=1.
\]

That should be a fast path across all methods.


---

17. Domain validation is missing

There's no validation of:

n >= 0
lambda_val > -1/2
x ∈ [-1,1]

For classical Gegenbauer orthogonality the useful parameter region is normally

\[
\lambda > -\frac12
\]

with special degeneracies around \(\lambda=0\).

Your quotient-ring conversion

d = int(2 * self.lambda_val + 2)

is even more restrictive.

For example:

lambda_val = 1.2

gets

d = 4

although \(2\lambda+2=4.4\), so you've silently changed the dimension.

That is a serious mathematical bug.

Either require:

2 * lambda_val + 2 ∈ integers

for the quotient-ring method, or don't derive d this way.


---

18. The quotient construction assumes a geometric dimension that not every Gegenbauer problem has

The statement

d = int(2 * self.lambda_val + 2)

encodes

\[
\lambda=\frac{d-2}{2}.
\]

That's correct for the standard zonal spherical-function interpretation on \(S^{d-1}\), but arbitrary Gegenbauer polynomials have continuous \(\lambda\).

Your solver therefore actually combines two problem classes:

general Gegenbauer C_n^(λ)

and

spherical harmonic / SO(d)/SO(d−1) zonal functions

which only coincide geometrically when the dimension relation makes sense.

That should be represented explicitly in the API.


---

19. Timing methodology is weak for short functions

You do:

iterations = 10 if num_points < 1000 else 1

This is insufficient for reliable microbenchmarks.

Problems include:

function-call overhead;

cache warmup;

scipy startup;

CPU frequency scaling;

allocation noise;

Python interpreter effects;

stochastic LNS noise;

single-run measurement at 1000+ points.


At minimum:

warmup
multiple repetitions
median
MAD / percentile dispersion

and ideally separate:

setup
allocation
kernel

time.


---

20. The method names overstate what the code demonstrates

This isn't merely naming pedantry.

You call this:

> "Algebraic Geometry Expression Permutations"



but:

recurrence is numerical recurrence;

hypergeometric is special-function representation;

WKB is asymptotic analysis;

Mehler-Heine is asymptotic limit;

quotient-ring is algebraic;

composite is matched asymptotics.


These aren't all "algebraic geometry permutations".

The abstraction is mixing several mathematical layers.

Given the direction of your previous code, I'd restructure the conceptual stack as:

MATHEMATICAL REPRESENTATION
    ├── recurrence
    ├── hypergeometric
    ├── harmonic quotient / projection
    └── asymptotic representations

VALIDITY REGIME
    ├── global
    ├── bulk
    ├── endpoint
    └── transition

NUMERICAL BACKEND
    ├── float32
    ├── float64
    ├── longdouble
    ├── fixed-point
    └── arbitrary precision

COST MODEL
    ├── measured latency
    ├── allocation count
    └── optional symbolic op count

ERROR MODEL
    ├── high-precision reference
    ├── absolute
    ├── relative
    └── scaled mixed error

OPTIMIZER
    └── constrained Pareto selection

That is substantially cleaner.


---

21. There is a major missed opportunity: exploit exact anchors

For normalized Gegenbauer,

\[
\phi_n(1)=1,
\]

and

\[
\phi_n(-1)=(-1)^n.
\]

You should include these as exact invariants in the verification layer.

Also useful:

\[
C_n^{(\lambda)}(x)
\]

has parity

\[
C_n^{(\lambda)}(-x)=(-1)^n C_n^{(\lambda)}(x).
\]

Your benchmark should automatically test:

phi_n(1) = 1
phi_n(-1) = (-1)^n
phi_n(-x) = (-1)^n phi_n(x)

These catch a surprising number of implementation errors.

For example, your current WKB clipping prevents a clean endpoint test by construction.


---

22. The current benchmark can rank a bad method as "optimal"

This is the fundamental consequence of all the above.

Suppose:

quotient-ring = x^n

and your domain is

x ∈ [0.5, 0.999]

For some \(n\), \(x^n\) may happen to resemble the zonal function in portions of the interval.

Then the Pareto solver may see:

cheap
reasonable-looking error

and promote it.

But it is not an alternative numerical implementation of the same formula.

Likewise the asymptotic methods can get penalized outside their validity regime and then discarded, even though they are precisely the correct methods in their intended regions.

So the present Pareto frontier is not yet a mathematical optimization of Gegenbauer evaluation. It is an optimization of a collection of heterogeneous approximations under an imperfect benchmark.
