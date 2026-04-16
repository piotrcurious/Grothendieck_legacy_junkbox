Dictionary-style rewrite

2-adic number
An element of , the field obtained by completing the rational numbers with respect to the 2-adic absolute value. In this number system, distance is measured by divisibility by powers of 2, not by size in the usual real sense.

2-adic integer
An element of , meaning a 2-adic number with no negative powers of 2 in its expansion. 2-adic integers naturally encode infinite carry behavior in base 2.

Quadratic extension 
The field obtained by adjoining a square root of 2 to . This is the correct setting for discussing  2-adically, because .

Fact about  in 
The equation

x^2 = 2

Minimal polynomial of 

x^2 - 2

Pell-type recurrence
The integer recurrence

a_{n+2} = 2a_{n+1} + a_n

Characteristic polynomial

\lambda^2 - 2\lambda - 1

1 \pm \sqrt{2}.

2-adic lift of a recurrence
The same recurrence, but interpreted in  or in a finite quotient . This keeps the carry structure that is lost when reducing everything mod 2.

LFSR over 
A linear feedback shift register working modulo 2. It produces a periodic bit sequence and is useful for finite-state binary dynamics, but it does not approximate irrational numbers in any numeric sense.

Why an ordinary LFSR is not enough
An LFSR over  forgets carries. Since carries are essential in the 2-adic and Pell-type setting, a plain LFSR cannot faithfully represent the structure behind .

Correct interpretation
The right object is not “an LFSR approximating ”. The right object is a linear recurrence over  or , with optional reduction modulo  to obtain finite approximations.

State space
A pair  representing the 2-adic element

u + v\alpha,

\alpha = 1+\sqrt{2}.

Update rule
Multiplication by  gives

(u,v) \mapsto (v,\; u+2v).

Finite approximation
Reducing the state update modulo  gives a finite machine that preserves 2-adic carry behavior up to precision .

One-sentence summary

The correct framework is: use the Pell recurrence in  or , not a standard LFSR over .
