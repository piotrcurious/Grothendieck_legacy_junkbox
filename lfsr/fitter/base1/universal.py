from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Iterator, Sequence
from itertools import product
from sympy import symbols, groebner, Poly, isprime

def is_prime_int(p: int) -> bool:
    """Check if p is a prime integer."""
    return isinstance(p, int) and p >= 2 and bool(isprime(p))

def mod_inverse(a: int, p: int) -> int:
    """Compute modular inverse of a modulo prime p."""
    a_mod = a % p
    if a_mod == 0:
        raise ZeroDivisionError(f"Zero has no modular inverse modulo {p}")
    return pow(a_mod, p - 2, p)

def recurrence_residual(seq: Sequence[int], coeffs: Sequence[int], m: int, p: int) -> List[int]:
    """
    Canonical predicate computing residuals for sequence `seq` under LFSR recurrence:
        a_{n+m} = sum_{j=0}^{m-1} c_j * a_{n+j} (mod p)

    Residual at index i (0 <= i <= len(seq) - m - 1):
        r_i = (sum_{j=0}^{m-1} c_j * seq[i+j] - seq[i+m]) mod p

    Returns a list of residuals r_i. If all r_i == 0, coeffs is a valid recurrence for seq.
    """
    if not is_prime_int(p):
        raise ValueError(f"Modulus p must be a prime integer, got {p}")
    if len(coeffs) != m:
        raise ValueError(f"Coefficients length ({len(coeffs)}) must equal order m ({m})")
    if len(seq) < m:
        raise ValueError(f"Sequence length ({len(seq)}) must be at least order m ({m})")

    seq_norm = [int(x) % p for x in seq]
    coeffs_norm = [int(c) % p for c in coeffs]

    residuals = []
    for i in range(len(seq_norm) - m):
        predicted = sum(coeffs_norm[j] * seq_norm[i+j] for j in range(m)) % p
        r = (predicted - seq_norm[i+m]) % p
        residuals.append(r)
    return residuals

def generate_lfsr_sequence(
    coeffs: Sequence[int],
    initial_state: Sequence[int],
    total_length: int,
    modulus: int = 2
) -> List[int]:
    """
    Generate a sequence from an LFSR defined by the recurrence:
        a_{n+m} = c_0*a_n + c_1*a_{n+1} + ... + c_{m-1}*a_{n+m-1} (mod modulus)

    Validates:
      - modulus is prime.
      - len(initial_state) == len(coeffs) == m.
      - total_length >= m (rejects total_length < m).
      - normalizes inputs to 0..p-1.
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    m = len(initial_state)
    if len(coeffs) != m:
        raise ValueError(f"Length of initial_state ({m}) must match length of coeffs ({len(coeffs)})")
    if total_length < m:
        raise ValueError(f"total_length ({total_length}) cannot be less than LFSR order m ({m})")

    p = modulus
    state = [int(x) % p for x in initial_state]
    c = [int(x) % p for x in coeffs]

    seq = list(state)
    for i in range(total_length - m):
        next_val = sum(c[j] * seq[i+j] for j in range(m)) % p
        seq.append(next_val)
    return seq

@dataclass
class LFSRSolution:
    """
    Result of LFSR polynomial fitting over GF(p).
    Encapsulates status ('unique', 'underdetermined', or 'inconsistent'),
    the particular solution, free variables and free basis vectors for affine families,
    and the Gröbner basis used.
    """
    status: str  # "unique", "underdetermined", "inconsistent"
    particular: Optional[Tuple[int, ...]] = None
    pivots: List[int] = field(default_factory=list)
    free_variables: List[int] = field(default_factory=list)
    free_vectors: List[Tuple[int, ...]] = field(default_factory=list)
    basis: Any = None
    modulus: int = 2
    order: int = 0

    def evaluate_free_variables(self, params: Dict[int, int]) -> Tuple[int, ...]:
        """
        For an underdetermined solution, evaluate assigning specific values to free variables.
        params: dict mapping free_variable_index -> int value in GF(p).
        """
        if self.status == "inconsistent" or self.particular is None:
            raise ValueError("Cannot evaluate free variables for an inconsistent solution.")
        p = self.modulus
        sol = list(self.particular)
        for f_idx, vec in zip(self.free_variables, self.free_vectors):
            t = params.get(f_idx, 0) % p
            for j in range(self.order):
                sol[j] = (sol[j] + t * vec[j]) % p
        return tuple(sol)

    def all_solutions(self) -> List[Tuple[int, ...]]:
        """
        Return list of all coefficient vectors fitting the sequence over GF(p).
        Returns [] if inconsistent.
        """
        if self.status == "inconsistent" or self.particular is None:
            return []
        if not self.free_variables:
            return [self.particular]
        p = self.modulus
        num_free = len(self.free_variables)
        sols = []
        for param_combo in product(range(p), repeat=num_free):
            params = dict(zip(self.free_variables, param_combo))
            sols.append(self.evaluate_free_variables(params))
        return sols

    def parametric_str(self) -> str:
        """
        Return human-readable parametric string representation of the solution family.
        """
        if self.status == "inconsistent":
            return "Inconsistent (no LFSR fits sequence)"
        if self.status == "unique":
            return f"{self.particular} (mod {self.modulus})"

        p = self.modulus
        param_names = {f: f"t{i}" for i, f in enumerate(self.free_variables)}
        exprs = []
        for j in range(self.order):
            part_val = self.particular[j]
            terms = []
            if part_val != 0 or not param_names:
                terms.append(str(part_val))
            for f_idx, vec in zip(self.free_variables, self.free_vectors):
                coeff = vec[j]
                if coeff != 0:
                    t_name = param_names[f_idx]
                    if coeff == 1:
                        terms.append(t_name)
                    else:
                        terms.append(f"{coeff}*{t_name}")
            if not terms:
                exprs.append("0")
            else:
                exprs.append(" + ".join(terms))

        params_decl = ", ".join(param_names.values())
        return f"({', '.join(exprs)}) (mod {p}), where {params_decl} in GF({p})"

def fit_lfsr_using_groebner(seq: Sequence[int], m: int, modulus: int = 2):
    """
    Given sequence seq and assumed order m, set up linear equations:
        sum_{j=0}^{m-1} c_j*a_{n+j} - a_{n+m} = 0 (mod modulus)
    and compute the Gröbner basis over GF(modulus).

    Returns:
        (Groebner basis G, coeff_syms)
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    if m < 1:
        raise ValueError(f"Order m must be >= 1, got {m}")
    if len(seq) < m:
        raise ValueError(f"Sequence length ({len(seq)}) must be at least order m ({m})")

    p = modulus
    seq_norm = [int(x) % p for x in seq]
    coeff_syms = symbols(f'c0:{m}', integer=True)

    eqs = []
    for i in range(len(seq_norm) - m):
        eq = sum(coeff_syms[j] * seq_norm[i+j] for j in range(m)) - seq_norm[i+m]
        eqs.append(eq)

    if not eqs:
        G = groebner([], *coeff_syms, modulus=p)
    else:
        G = groebner(eqs, *coeff_syms, modulus=p)

    return G, coeff_syms

def extract_linear_system_from_groebner(G, coeff_syms, m: int, p: int) -> Tuple[List[List[int]], List[int]]:
    """
    Extract matrix equation A * c = rhs (mod p) from Gröbner basis G.
    Asserts total_degree(g) <= 1 for all generators g in G.
    """
    A = []
    rhs = []
    for g in G:
        poly = Poly(g, *coeff_syms, modulus=p)
        deg = poly.total_degree()
        if deg > 1:
            raise ValueError(
                f"Invariant violation: generator {g} in Gröbner basis has total degree {deg} > 1. "
                "The recurrence system ideal was expected to be linear."
            )

        row = []
        for sym in coeff_syms:
            coeff_val = poly.coeff_monomial(sym)
            row.append(int(coeff_val) % p)

        const_term = int(poly.coeff_monomial(1)) % p

        A.append(row)
        rhs.append((-const_term) % p)

    return A, rhs

def solve_linear_system_gfp(
    A: List[List[int]],
    rhs: List[int],
    m: int,
    p: int,
    basis: Any = None
) -> LFSRSolution:
    """
    Perform Gauss-Jordan elimination over GF(p) on system A * c = rhs (mod p).
    """
    k = len(A)
    matrix = []
    for r in range(k):
        row = [A[r][j] % p for j in range(m)] + [rhs[r] % p]
        matrix.append(row)

    pivot_cols = []
    pivot_row_map = {}
    pivot_row = 0

    for col in range(m):
        r_found = -1
        for r in range(pivot_row, k):
            if matrix[r][col] % p != 0:
                r_found = r
                break
        if r_found != -1:
            matrix[pivot_row], matrix[r_found] = matrix[r_found], matrix[pivot_row]

            inv = mod_inverse(matrix[pivot_row][col] % p, p)
            matrix[pivot_row] = [(val * inv) % p for val in matrix[pivot_row]]

            for r in range(k):
                if r != pivot_row and matrix[r][col] % p != 0:
                    factor = matrix[r][col] % p
                    matrix[r] = [(matrix[r][j] - factor * matrix[pivot_row][j]) % p for j in range(m + 1)]

            pivot_cols.append(col)
            pivot_row_map[col] = pivot_row
            pivot_row += 1

    for r in range(k):
        lhs_zero = all(matrix[r][j] % p == 0 for j in range(m))
        if lhs_zero and matrix[r][m] % p != 0:
            return LFSRSolution(
                status="inconsistent",
                particular=None,
                pivots=[],
                free_variables=list(range(m)),
                free_vectors=[],
                basis=basis,
                modulus=p,
                order=m
            )

    free_vars = [j for j in range(m) if j not in pivot_cols]

    particular = [0] * m
    for p_col in pivot_cols:
        r = pivot_row_map[p_col]
        particular[p_col] = matrix[r][m] % p

    free_vectors = []
    for f_var in free_vars:
        vec = [0] * m
        vec[f_var] = 1
        for p_col in pivot_cols:
            r = pivot_row_map[p_col]
            vec[p_col] = (-matrix[r][f_var]) % p
        free_vectors.append(tuple(vec))

    status = "unique" if len(pivot_cols) == m else "underdetermined"

    return LFSRSolution(
        status=status,
        particular=tuple(particular),
        pivots=pivot_cols,
        free_variables=free_vars,
        free_vectors=free_vectors,
        basis=basis,
        modulus=p,
        order=m
    )

def fit_lfsr(seq: Sequence[int], m: int, modulus: int = 2) -> LFSRSolution:
    """
    Fits an LFSR of order m to sequence seq over GF(modulus) using Gröbner bases
    and linear generator decoding with Gauss-Jordan elimination over GF(p).
    """
    G, coeff_syms = fit_lfsr_using_groebner(seq, m, modulus)
    A, rhs = extract_linear_system_from_groebner(G, coeff_syms, m, modulus)
    return solve_linear_system_gfp(A, rhs, m, modulus, G)

def fit_lfsr_brute_force(seq: Sequence[int], m: int, modulus: int = 2) -> List[Tuple[int, ...]]:
    """
    Brute-force oracle searching all p^m candidate coefficient tuples.
    Used as a cross-check verification oracle in small test cases.
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    if len(seq) < m:
        raise ValueError(f"Sequence length ({len(seq)}) must be at least order m ({m})")
    p = modulus
    valid_sols = []
    for candidate in product(range(p), repeat=m):
        residuals = recurrence_residual(seq, candidate, m, p)
        if all(r == 0 for r in residuals):
            valid_sols.append(candidate)
    return valid_sols

def coeffs_to_characteristic_poly(coeffs: Sequence[int], modulus: int = 2):
    """
    Convert recurrence coeffs (c_0, ..., c_{m-1}) for a_{n+m} = sum c_j a_{n+j} (mod p)
    to characteristic polynomial P(x) = x^m - sum_{j=0}^{m-1} c_j x^j (mod p).
    Returns a SymPy Poly in x over GF(p).
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    p = modulus
    m = len(coeffs)
    x = symbols('x')
    c = [int(val) % p for val in coeffs]
    poly_expr = x**m - sum(c[j] * x**j for j in range(m))
    return Poly(poly_expr, x, modulus=p)

def characteristic_poly_to_coeffs(poly, m: int, modulus: int = 2) -> Tuple[int, ...]:
    """
    Convert characteristic polynomial P(x) = x^m - sum_{j=0}^{m-1} c_j x^j (mod p)
    back to recurrence coefficient tuple (c_0, c_1, ..., c_{m-1}).
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    p = modulus
    x = symbols('x')
    poly_p = Poly(poly, x, modulus=p)
    deg = poly_p.degree()
    if deg != m:
        raise ValueError(f"Polynomial degree ({deg}) does not match order m ({m})")
    leading_coeff = int(poly_p.coeff_monomial(x**m)) % p
    if leading_coeff != 1:
        inv = mod_inverse(leading_coeff, p)
        poly_p = Poly(poly_p * inv, x, modulus=p)

    coeffs = []
    for j in range(m):
        coeff_xj = int(poly_p.coeff_monomial(x**j)) % p
        c_j = (-coeff_xj) % p
        coeffs.append(c_j)
    return tuple(coeffs)

def coeffs_to_connection_poly(coeffs: Sequence[int], modulus: int = 2):
    """
    Convert recurrence coeffs (c_0, ..., c_{m-1}) to connection (reciprocal) polynomial:
        C(x) = 1 - sum_{j=1}^m c_{m-j} x^j (mod p).
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    p = modulus
    m = len(coeffs)
    x = symbols('x')
    c = [int(val) % p for val in coeffs]
    expr = 1 - sum(c[m-j] * x**j for j in range(1, m+1))
    return Poly(expr, x, modulus=p)

def lfsr_next(state: Sequence[int], coeffs: Sequence[int], modulus: int = 2) -> Tuple[int, ...]:
    """
    Given state tuple (a_n, ..., a_{n+m-1}) and recurrence coeffs (c_0, ..., c_{m-1}),
    compute next state (a_{n+1}, ..., a_{n+m}) where a_{n+m} = sum c_j a_{n+j} (mod p).
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    if coeffs is None:
        raise ValueError("coeffs cannot be None")
    m = len(state)
    if len(coeffs) != m:
        raise ValueError(f"len(state) ({m}) must match len(coeffs) ({len(coeffs)})")
    p = modulus
    st_norm = [int(s) % p for s in state]
    c_norm = [int(c) % p for c in coeffs]
    new_val = sum(c_norm[j] * st_norm[j] for j in range(m)) % p
    return tuple(st_norm[1:]) + (new_val,)

def lfsr_state_transition_iter(
    coeffs: Sequence[int],
    m: int,
    modulus: int = 2
) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """
    Lazy iterator yielding (state, next_state) pairs for all p^m possible states over GF(p).
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    if coeffs is None:
        raise ValueError("coeffs cannot be None")
    if len(coeffs) != m:
        raise ValueError(f"coeffs length ({len(coeffs)}) must equal order m ({m})")
    p = modulus
    c_norm = tuple(int(c) % p for c in coeffs)
    for state in product(range(p), repeat=m):
        next_st = lfsr_next(state, c_norm, modulus=p)
        yield state, next_st

def lfsr_state_transition(
    coeffs: Sequence[int],
    m: int,
    modulus: int = 2,
    max_states: int = 65536
) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build state transition dictionary for all p^m states.
    Raises ValueError if p^m > max_states to prevent memory explosion.
    """
    if not is_prime_int(modulus):
        raise ValueError(f"Modulus must be prime, got {modulus}")
    if coeffs is None:
        raise ValueError("coeffs cannot be None")
    p = modulus
    num_states = p ** m
    if num_states > max_states:
        raise ValueError(
            f"State transition table size p^m = {p}^{m} = {num_states} exceeds "
            f"max_states ({max_states}). Use lfsr_state_transition_iter() for lazy iteration."
        )
    transitions = {}
    for state, next_state in lfsr_state_transition_iter(coeffs, m, modulus=p):
        transitions[state] = next_state
    return transitions

if __name__ == '__main__':
    # Demonstration of the improved architecture
    modulus = 2
    m = 3
    true_coeffs = (1, 1, 0)
    initial_state = (1, 0, 0)
    total_length = 10

    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length, modulus)
    print("Generated sequence:", seq)

    sol = fit_lfsr(seq, m, modulus)
    print("\nFitted LFSR Solution Status:", sol.status)
    print("Particular Solution:", sol.particular)
    print("Parametric Solution:", sol.parametric_str())

    # Verify solution with canonical predicate
    res = recurrence_residual(seq, sol.particular, m, modulus)
    print("Recurrence Residuals:", res)

    # Brute-force oracle comparison
    bf_sols = fit_lfsr_brute_force(seq, m, modulus)
    print("Brute-force oracle solutions:", bf_sols)
    print("Extracted solutions match oracle:", set(sol.all_solutions()) == set(bf_sols))

    # Polynomial conversions
    char_poly = coeffs_to_characteristic_poly(sol.particular, modulus)
    recovered_coeffs = characteristic_poly_to_coeffs(char_poly, m, modulus)
    print("\nCharacteristic Polynomial:", char_poly)
    print("Recovered Coeffs from Polynomial:", recovered_coeffs)

    # State transition table
    transitions = lfsr_state_transition(sol.particular, m, modulus)
    print("\nLFSR state transition table over GF({}):".format(modulus))
    for state, next_state in sorted(transitions.items()):
        print(f"  State {state} -> Next state {next_state}")
