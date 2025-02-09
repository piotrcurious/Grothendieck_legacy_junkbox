# Example: Fitting an LFSR polynomial to data using Gröbner bases
# and then building the LFSR shift register transition table.
#
# In this example we work in GF(2) (i.e. mod 2 arithmetic).
# We assume an LFSR of order 3 with recurrence:
#   a_{n+3} = c0*a_n + c1*a_{n+1} + c2*a_{n+2}   (mod 2)
#
# We will generate a sequence using the "true" coefficients:
#   (c0, c1, c2) = (1, 1, 0)   (which corresponds to x^3 + x + 1)
#
# Then we set up the equations based on the sequence and compute
# a Gröbner basis (with modulus=2) to recover the coefficients.
# Finally, we create the LFSR shift register transition table.

from sympy import symbols, groebner

def generate_lfsr_sequence(coeffs, initial_state, total_length):
    """
    Generate a bit sequence from an LFSR defined by the recurrence
       a_{n+3} = c0*a_n + c1*a_{n+1} + c2*a_{n+2}   (mod 2)
    :param coeffs: tuple (c0, c1, c2) with values 0 or 1.
    :param initial_state: tuple of 3 bits giving the starting state.
    :param total_length: total length of the output sequence.
    :return: list of bits.
    """
    m = len(initial_state)
    seq = list(initial_state)
    # Generate additional bits using the recurrence.
    for i in range(total_length - m):
        new_bit = (coeffs[0]*seq[i] + coeffs[1]*seq[i+1] + coeffs[2]*seq[i+2]) % 2
        seq.append(new_bit)
    return seq

def fit_lfsr_using_groebner(seq, m):
    """
    Given a bit sequence (list of 0's and 1's) and an assumed LFSR order m,
    set up the equations
         seq[i+m] + c0*seq[i] + c1*seq[i+1] + ... + c_{m-1}*seq[i+m-1] = 0 (mod 2)
    and compute the Groebner basis for these equations over GF(2).
    :param seq: list of bits.
    :param m: order of the LFSR.
    :return: (Groebner basis, list of coefficient symbols)
    """
    # Define unknown coefficients c0, c1, ..., c_{m-1}
    coeffs = symbols('c0:%d' % m, integer=True)
    eqs = []
    # For every valid index, create an equation
    for i in range(len(seq) - m):
        eq = seq[i+m] + sum(coeffs[j]*seq[i+j] for j in range(m))
        eqs.append(eq)
    # Compute the Groebner basis modulo 2.
    G = groebner(eqs, *coeffs, modulus=2)
    return G, coeffs

def lfsr_next(state, coeffs):
    """
    Given a current state (tuple of m bits) and LFSR coefficients,
    compute the next state. (The new bit is computed as the sum over GF2
    of the product of the coefficients with the state bits.)
    """
    m = len(state)
    new_bit = sum(coeffs[i]*state[i] for i in range(m)) % 2
    # Shift left: drop the leftmost bit and append the new bit at the right.
    new_state = state[1:] + (new_bit,)
    return new_state

def lfsr_state_transition(coeffs, m):
    """
    Build the state transition table for an m-bit LFSR.
    There are 2^m states (including the all-zero state).
    :param coeffs: tuple of LFSR coefficients.
    :param m: order of the LFSR.
    :return: dictionary mapping state -> next state.
    """
    transitions = {}
    for s in range(2**m):
        # Create an m–bit state (as a tuple). The bits are taken from
        # the binary representation of s.
        state = tuple((s >> i) & 1 for i in reversed(range(m)))
        next_state = lfsr_next(state, coeffs)
        transitions[state] = next_state
    return transitions

if __name__ == '__main__':
    # Step 1. Generate a sequence from a known LFSR.
    true_coeffs = (1, 1, 0)        # our true coefficients: x^3 + x + 1
    initial_state = (1, 0, 0)        # initial 3-bit state
    total_length = 10              # generate 10 bits
    seq = generate_lfsr_sequence(true_coeffs, initial_state, total_length)
    print("Generated sequence:", seq)
    
    # Step 2. Set up the equations for the recurrence and compute the Groebner basis.
    m = 3  # LFSR order
    G, coeff_syms = fit_lfsr_using_groebner(seq, m)
    print("\nComputed Groebner basis (over GF(2)):")
    for g in G:
        print(" ", g)
    # For this example one finds that the basis is something like:
    #    c0 + 1,   c1 + 1,   c2
    # which tells us that c0 = 1, c1 = 1, and c2 = 0.
    
    # (Optionally, we can also verify the solution by brute force.)
    possible_solutions = [(a, b, c) for a in [0,1] for b in [0,1] for c in [0,1]]
    solution = None
    for sol in possible_solutions:
        valid = True
        for i in range(len(seq)-m):
            if (seq[i+m] + sol[0]*seq[i] + sol[1]*seq[i+1] + sol[2]*seq[i+2]) % 2 != 0:
                valid = False
                break
        if valid:
            solution = sol
            break
    print("\nFitted coefficients (via brute force check):", solution)
    
    # Step 3. Use the fitted coefficients to build the LFSR state transition table.
    transitions = lfsr_state_transition(solution, m)
    print("\nLFSR state transition table:")
    for state, next_state in sorted(transitions.items()):
        print("  State {} -> Next state {}".format(state, next_state))
