# Example: Fitting an LFSR polynomial to data using Gröbner bases
# and decoding the linear basis via GF(2) Gauss-Jordan elimination.
#
# In this example we work in GF(2) (i.e. mod 2 arithmetic).
# We assume an LFSR of order m with recurrence:
#   a_{n+m} = c_0*a_n + c_1*a_{n+1} + ... + c_{m-1}*a_{n+m-1}   (mod 2)

import sys
from pathlib import Path
from typing import List, Tuple, Dict, Sequence

# Ensure base1 directory is in sys.path for direct script execution
base1_dir = Path(__file__).resolve().parent
if str(base1_dir) not in sys.path:
    sys.path.insert(0, str(base1_dir))

from universal import (
    generate_lfsr_sequence as gen_seq_universal,
    fit_lfsr_using_groebner as fit_groebner_universal,
    fit_lfsr as fit_lfsr_universal,
    lfsr_next as lfsr_next_universal,
    lfsr_state_transition as lfsr_state_transition_universal,
    recurrence_residual,
    fit_lfsr_brute_force,
    LFSRSolution,
)

def generate_lfsr_sequence(
    coeffs: Sequence[int],
    initial_state: Sequence[int],
    total_length: int
) -> List[int]:
    """
    Generate a bit sequence from an LFSR over GF(2).
    """
    return gen_seq_universal(coeffs, initial_state, total_length, modulus=2)

def fit_lfsr_using_groebner(seq: Sequence[int], m: int):
    """
    Compute Gröbner basis over GF(2) for unknown coefficients c0, ..., c_{m-1}.
    """
    return fit_groebner_universal(seq, m, modulus=2)

def fit_lfsr(seq: Sequence[int], m: int) -> LFSRSolution:
    """
    Fit LFSR coefficients using Gröbner basis and GF(2) Gauss-Jordan elimination.
    """
    return fit_lfsr_universal(seq, m, modulus=2)

def lfsr_next(state: Sequence[int], coeffs: Sequence[int]) -> Tuple[int, ...]:
    """
    Compute the next m-bit state given current state and GF(2) coefficients.
    """
    return lfsr_next_universal(state, coeffs, modulus=2)

def lfsr_state_transition(coeffs: Sequence[int], m: int, max_states: int = 65536) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
    """
    Build state transition dictionary for all 2^m states over GF(2).
    """
    return lfsr_state_transition_universal(coeffs, m, modulus=2, max_states=max_states)

if __name__ == '__main__':
    # Step 1. Generate a sequence from a known LFSR over GF(2).
    true_coeffs = (1, 1, 0)        # x^3 + x + 1 recurrence: a_{n+3} = a_n + a_{n+1}
    initial_state = (1, 0, 0)      # initial 3-bit state
    total_length = 10              # generate 10 bits
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length)
    print("Generated sequence:", seq)

    # Step 2. Compute Gröbner basis and decode via GF(2) Gauss-Jordan elimination.
    m = 3  # LFSR order
    G, coeff_syms = fit_lfsr_using_groebner(seq, m)
    print("\nComputed Gröbner basis (over GF(2)):")
    for g in G:
        print(" ", g)

    sol = fit_lfsr(seq, m)
    print("\nFitted solution status:", sol.status)
    print("Particular solution (c0, c1, c2):", sol.particular)
    print("Parametric solution:", sol.parametric_str())

    # Step 3. Verify solution using canonical predicate and brute-force oracle check
    residuals = recurrence_residual(seq, sol.particular, m, 2)
    print("Recurrence residuals:", residuals)

    bf_sols = fit_lfsr_brute_force(seq, m, 2)
    print("Brute-force oracle solutions:", bf_sols)
    print("Extracted solution matches oracle:", set(sol.all_solutions()) == set(bf_sols))

    # Step 4. Use fitted coefficients to build LFSR state transition table.
    transitions = lfsr_state_transition(sol.particular, m)
    print("\nLFSR state transition table:")
    for state, next_state in sorted(transitions.items()):
        print(f"  State {state} -> Next state {next_state}")
