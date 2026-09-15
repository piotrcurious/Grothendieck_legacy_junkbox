import sys
from pathlib import Path
import pytest
from sympy import Poly, symbols

base1_dir = Path(__file__).resolve().parent
if str(base1_dir) not in sys.path:
    sys.path.insert(0, str(base1_dir))

from universal import (
    generate_lfsr_sequence,
    fit_lfsr_using_groebner,
    fit_lfsr,
    fit_lfsr_brute_force,
    recurrence_residual,
    coeffs_to_characteristic_poly,
    characteristic_poly_to_coeffs,
    coeffs_to_connection_poly,
    lfsr_state_transition,
    lfsr_state_transition_iter,
    lfsr_next,
    LFSRSolution,
)

def test_unique_recurrence_gf2():
    true_coeffs = (1, 1, 0)
    initial_state = (1, 0, 0)
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=10, modulus=2)

    sol = fit_lfsr(seq, m=3, modulus=2)
    assert sol.status == "unique"
    assert sol.particular == (1, 1, 0)
    assert sol.free_variables == []
    assert sol.free_vectors == []

    residuals = recurrence_residual(seq, sol.particular, m=3, p=2)
    assert all(r == 0 for r in residuals)

    bf_sols = fit_lfsr_brute_force(seq, m=3, modulus=2)
    assert set(sol.all_solutions()) == set(bf_sols)

def test_unique_recurrence_gf3():
    p = 3
    m = 3
    # Use a full-period / maximal LFSR over GF(3)
    true_coeffs = (2, 1, 0)
    initial_state = (1, 2, 1)
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=12, modulus=p)

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "unique"
    assert sol.particular == true_coeffs

    residuals = recurrence_residual(seq, sol.particular, m=m, p=p)
    assert all(r == 0 for r in residuals)

    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert set(sol.all_solutions()) == set(bf_sols)

def test_p_greater_than_2():
    for p in (3, 5, 7):
        m = 2
        # Use a non-degenerate recurrence over GF(p)
        true_coeffs = (1, 1)  # a_{n+2} = a_n + a_{n+1}
        initial_state = (1, 0)
        seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=8, modulus=p)

        sol = fit_lfsr(seq, m=m, modulus=p)
        assert sol.status == "unique"
        assert sol.particular == true_coeffs

        bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
        assert set(sol.all_solutions()) == set(bf_sols)

def test_m_equals_1():
    for p in (2, 3, 5):
        m = 1
        true_coeffs = (p - 1,)
        initial_state = (1,)
        seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=6, modulus=p)

        sol = fit_lfsr(seq, m=m, modulus=p)
        assert sol.status == "unique"
        assert sol.particular == true_coeffs

        bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
        assert set(sol.all_solutions()) == set(bf_sols)

def test_underdetermined_multiple_recurrences():
    # Sequence length m + 1 provides only 1 constraint for order m=3 over GF(2)
    p = 2
    m = 3
    seq = [1, 0, 1, 1]  # len = 4. Equation: c0*1 + c1*0 + c2*1 - 1 = 0 => c0 + c2 = 1 (mod 2)

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "underdetermined"
    assert len(sol.free_variables) > 0

    extracted_sols = sol.all_solutions()
    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)

    assert len(extracted_sols) == len(bf_sols)
    assert set(extracted_sols) == set(bf_sols)

    for s in extracted_sols:
        res = recurrence_residual(seq, s, m=m, p=p)
        assert all(r == 0 for r in res)

def test_inconsistent_noisy_sequence():
    p = 2
    m = 2
    # Inconsistent sequence for order 2
    seq = [1, 0, 1, 0, 1, 1]

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "inconsistent"
    assert sol.particular is None
    assert sol.all_solutions() == []

    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert bf_sols == []

def test_sequence_length_equals_m():
    p = 2
    m = 3
    seq = [1, 0, 1]  # len == m

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "underdetermined"
    assert len(sol.all_solutions()) == p**m

    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert len(bf_sols) == p**m
    assert set(sol.all_solutions()) == set(bf_sols)

def test_sequence_length_less_than_m():
    with pytest.raises(ValueError, match="cannot be less than LFSR order m"):
        generate_lfsr_sequence((1, 1), (1, 0), total_length=1, modulus=2)

    with pytest.raises(ValueError, match="must be at least order m"):
        fit_lfsr([1], m=2, modulus=2)

def test_zero_sequence():
    p = 3
    m = 2
    seq = [0, 0, 0, 0, 0]

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "underdetermined"

    extracted_sols = sol.all_solutions()
    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert set(extracted_sols) == set(bf_sols)
    assert len(extracted_sols) == p**m  # All c in GF(3)^2 fit zero sequence

def test_zero_coefficients():
    p = 2
    m = 3
    true_coeffs = (0, 0, 0)
    initial_state = (1, 0, 0)
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=8, modulus=p)
    # Generated sequence: [1, 0, 0, 0, 0, 0, 0, 0]

    sol = fit_lfsr(seq, m=m, modulus=p)
    # Note: [1, 0, 0, 0, 0, 0, 0, 0] constrains c0=0, but c1 and c2 can be anything (0 or 1).
    assert sol.status == "underdetermined"
    assert (0, 0, 0) in sol.all_solutions()

    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert set(sol.all_solutions()) == set(bf_sols)

def test_non_binary_initial_states():
    p = 5
    m = 2
    true_coeffs = (3, 4)
    initial_state = (2, 3)  # non-binary initial state elements
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length=10, modulus=p)

    sol = fit_lfsr(seq, m=m, modulus=p)
    assert sol.status == "unique"
    assert sol.particular == true_coeffs

    bf_sols = fit_lfsr_brute_force(seq, m=m, modulus=p)
    assert set(sol.all_solutions()) == set(bf_sols)

def test_modulus_prime_validation():
    with pytest.raises(ValueError, match="must be prime"):
        fit_lfsr([1, 0, 1, 0], m=2, modulus=4)

    with pytest.raises(ValueError, match="must be prime"):
        generate_lfsr_sequence((1, 1), (1, 0), total_length=5, modulus=6)

    with pytest.raises(ValueError, match="must be a prime integer"):
        recurrence_residual([1, 0, 1, 0], (1, 1), m=2, p=1)

def test_polynomial_conversions():
    p = 2
    m = 3
    coeffs = (1, 1, 0)

    char_poly = coeffs_to_characteristic_poly(coeffs, modulus=p)
    assert char_poly.degree() == 3

    recovered_coeffs = characteristic_poly_to_coeffs(char_poly, m=m, modulus=p)
    assert recovered_coeffs == coeffs

    conn_poly = coeffs_to_connection_poly(coeffs, modulus=p)
    assert conn_poly.degree() == 3

def test_state_transition_and_size_guard():
    p = 2
    m = 3
    coeffs = (1, 1, 0)

    transitions = lfsr_state_transition(coeffs, m=m, modulus=p)
    assert len(transitions) == 2**m

    iter_transitions = dict(lfsr_state_transition_iter(coeffs, m=m, modulus=p))
    assert transitions == iter_transitions

    with pytest.raises(ValueError, match="exceeds max_states"):
        lfsr_state_transition(coeffs, m=m, modulus=p, max_states=4)

def test_lfsr_next_validation():
    with pytest.raises(ValueError, match="coeffs cannot be None"):
        lfsr_next((1, 0), None, modulus=2)

    with pytest.raises(ValueError, match="must match len"):
        lfsr_next((1, 0, 0), (1, 1), modulus=2)
