To combine two Linear Feedback Shift Registers (LFSRs) such that the third LFSR evaluates as their product (i.e., the output of the third LFSR represents the product of the two original LFSR outputs), you can use techniques inspired by algebraic geometry and field theory, particularly finite field multiplication.

Here’s an outline of the steps to achieve this:

1. Understanding LFSRs:

Each LFSR produces a sequence of bits that can be viewed as elements in a finite field (typically GF(2) for binary sequences).

The state of an LFSR at any given time is determined by a set of bits, and the next state is computed using a feedback polynomial. The output at each step is the value of a bit in the current state.


2. Algebraic Geometry of LFSRs:

The output sequence of an LFSR can be thought of as an element in a polynomial ring over a finite field. For two LFSRs, say LFSR1 and LFSR2, their output sequences can be written as polynomials P1(x) and P2(x) over GF(2).

The goal is to construct a third LFSR, LFSR3, whose output sequence corresponds to the product of these two sequences, i.e., P3(x) = P1(x) * P2(x).


3. Constructing the Product LFSR:

Polynomial Representation: Represent each LFSR by a polynomial over GF(2), where the degree of the polynomial corresponds to the number of stages in the LFSR. The coefficients of the polynomial represent the connection between the stages and the feedback tap.

Multiplication of Polynomials: To obtain the product of two LFSR sequences, multiply their corresponding polynomials:


P3(x) = P1(x) \cdot P2(x)

Feedback Polynomial of the Product: The degree of P3(x) will be the sum of the degrees of P1(x) and P2(x). If P1(x) is degree d1 and P2(x) is degree d2, then P3(x) will have a degree of d1 + d2. This tells you how many stages the third LFSR needs.


4. Implementing in an LFSR:

New LFSR Design: Design a new LFSR with d1 + d2 stages. The feedback polynomial of this LFSR will be derived from the polynomial product P3(x).

Initial Conditions: The initial states of LFSR1 and LFSR2 must be used to derive the initial state of LFSR3. Typically, the product of the initial states can be used as the initial condition for LFSR3.


5. Example:

Let’s say you have two LFSRs with the following feedback polynomials:

P1(x) = x^3 + x + 1 (LFSR1 has 3 stages)

P2(x) = x^2 + x + 1 (LFSR2 has 2 stages)


To find the product LFSR, multiply these polynomials:

P3(x) = (x^3 + x + 1) \cdot (x^2 + x + 1)

P3(x) = x^5 + x^4 + x^3 + x^4 + x^3 + x^2 + x^3 + x^2 + x + 1

P3(x) = x^5 + x^3 + x + 1

6. LFSR3 Design:

Design a 5-stage LFSR with feedback taps at the 5th, 3rd, and 1st stages, according to the feedback polynomial x^5 + x^3 + x + 1.

The output of this LFSR will represent the product of the outputs of the two original LFSRs.


Summary:

To obtain a third LFSR that computes the product of two LFSRs, express each LFSR as a polynomial over GF(2), multiply the polynomials, and design a new LFSR based on the resulting product polynomial. This method leverages algebraic structures inherent in LFSRs and finite fields, allowing you to construct composite sequences based on the operations of polynomial multiplication.

