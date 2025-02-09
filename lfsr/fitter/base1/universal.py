from sympy import symbols, groebner
from itertools import product

def generate_lfsr_sequence(coeffs, initial_state, total_length, modulus=2):
    """
    Generate a sequence from an LFSR defined by the recurrence:
         a_{n+m} = c_0*a_n + c_1*a_{n+1} + ... + c_{m-1}*a_{n+m-1}  (mod modulus)
    where m = len(initial_state) and coeffs is a tuple of length m.
    """
    m = len(initial_state)
    seq = list(initial_state)
    for i in range(total_length - m):
        next_val = sum(coeffs[j] * seq[i+j] for j in range(m)) % modulus
        seq.append(next_val)
    return seq

def fit_lfsr_using_groebner(seq, m, modulus=2):
    """
    Given a sequence and an assumed LFSR order m, set up the equations:
         sum_{j=0}^{m-1} c_j*a_{n+j} - a_{n+m} = 0  (mod modulus)
    for n = 0, 1, ..., len(seq)-m-1.
    
    Returns:
      - The computed Gröbner basis (over GF(modulus)) as a list of polynomials.
      - The list of coefficient symbols.
      
    Note: In the general case it may not be straightforward to extract the solution 
    directly from the basis. (For example, when p ≠ 2 the signs matter.)
    """
    coeffs = symbols('c0:%d' % m, integer=True)
    eqs = []
    for i in range(len(seq) - m):
        eq = sum(coeffs[j] * seq[i+j] for j in range(m)) - seq[i+m]
        eqs.append(eq)
    G = groebner(eqs, *coeffs, modulus=modulus)
    return G, coeffs

def lfsr_next(state, coeffs, modulus=2):
    """
    Given a current state (a tuple of m elements) and LFSR coefficients,
    compute the next state. (We compute the new value and then shift.)
    """
    m = len(state)
    new_val = sum(coeffs[j] * state[j] for j in range(m)) % modulus
    return state[1:] + (new_val,)

def lfsr_state_transition(coeffs, m, modulus=2):
    """
    Build the state transition table for an m-element LFSR over GF(modulus).
    There are modulus^m possible states.
    Returns a dictionary mapping each state to its successor state.
    """
    transitions = {}
    for state in product(range(modulus), repeat=m):
        next_state = lfsr_next(state, coeffs, modulus=modulus)
        transitions[state] = next_state
    return transitions

if __name__ == '__main__':
    # === Example Parameters ===
    # Choose the prime modulus (GF(p)). Here we use 2 (i.e. GF(2)) but you can change this.
    modulus = 2
    
    # LFSR order (number of coefficients). For example, m=3.
    m = 3
    
    # Define the "true" coefficients (for the recurrence)
    # Here the recurrence is: a_{n+3} = 1*a_n + 1*a_{n+1} + 0*a_{n+2}  (mod 2)
    true_coeffs = (1, 1, 0)
    
    # Initial state (length m) for generating the sequence.
    initial_state = (1, 0, 0)
    
    # Total length of the generated sequence.
    total_length = 10
    
    # === Generate a Sequence ===
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length, modulus)
    print("Generated sequence:", seq)
    
    # === Fit the LFSR coefficients using a Gröbner basis ===
    G, coeff_syms = fit_lfsr_using_groebner(seq, m, modulus)
    print("\nComputed Gröbner basis over GF({}):".format(modulus))
    for poly in G:
        print(" ", poly)
    
    # === Verify/Extract the Coefficients by Brute Force ===
    # (Extracting from G is not always straightforward.)
    possible_solutions = list(product(range(modulus), repeat=m))
    solution = None
    for sol in possible_solutions:
        valid = True
        for i in range(len(seq)-m):
            if (sum(sol[j] * seq[i+j] for j in range(m)) - seq[i+m]) % modulus != 0:
                valid = False
                break
        if valid:
            solution = sol
            break
    if solution is None:
        print("\nNo valid LFSR recurrence was found.")
    else:
        print("\nFitted coefficients (via brute force):", solution)
    
    # === Build the LFSR State Transition Table ===
    transitions = lfsr_state_transition(solution, m, modulus)
    print("\nLFSR state transition table over GF({}):".format(modulus))
    for state, next_state in sorted(transitions.items()):
        print("  State {} -> Next state {}".format(state, next_state))
