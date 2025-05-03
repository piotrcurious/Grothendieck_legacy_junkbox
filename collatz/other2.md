We can regard the usual “decimal” Collatz problem as nothing more mysterious than the study of rational points on a single algebraic‑geometric object, and then ask what happens when we replace the base field ℚ by larger fields—even those containing transcendental elements.  Here is how one sets this up in the language of schemes and field‑theory.


---

1. The Collatz map as a morphism of schemes

1. Model the parity‐branches algebraically.  Introduce an auxiliary “parity coordinate”  satisfying



y^2 \;=\;1,

X \;=\;\Spec\;A,\qquad A \;=\;\Q[x,y]/(y^2-1).

T(n)=\begin{cases}n/2,&n\equiv0\pmod2,\3n+1)/2,&n\equiv1\pmod2\end{cases}

T^*\!:A\;\longrightarrow\;A

\begin{aligned}
T^*(x)&=\frac{x}{2}\,\frac{1+y}{2}\;+\;\frac{3x+1}{2}\,\frac{1-y}{2},\\
T^*(y)&=\;(-1)^{\,T(x)}\;=\;\text{an explicit polynomial in }x,y.
\end{aligned}

T\colon X\;\longrightarrow\;X

> References:

The Collatz (3+1) map and its 2‑adic extension are discussed in detail on Wikipedia .

The notion of a scheme and morphism of schemes can be found on Wikipedia and in the Stacks Project .





---

2. Base‑change to arbitrary fields, including transcendental ones

Once you have  defined over , you may base‑change to any field .  Write

X_K \;=\;X\times_{\Spec\Q}\Spec K,

T_K\;=\;T\times\mathrm{id}_{\Spec K}\colon X_K\;\longrightarrow\;X_K.

If  is a transcendental extension—e.g.\  or —then  contains points whose -coordinate is a transcendental number.  One can now study the orbit


x,\;T_K(x),\;T_K^2(x),\;\dots

Because  is an honest algebraic morphism, all the tools of algebraic dynamics on schemes apply: you can form the th‑iterate fixed‑point locus

\{\,P\in X_K:\;T_K^n(P)=P\},

> Key point:  the “decimal integers” are just the special case  with the extra integrality condition.  For any larger field , including those obtained by adjoining transcendental elements, one obtains a family of Collatz–like dynamical systems .




---

3. Transcendental versus algebraic points

Algebraic (ℚ‑rational) points on  correspond to the classical Collatz problem: one asks whether every algebraic point with  eventually lands in the cycle .

Transcendental points (e.g.\ , , or ) give orbits in the field .  Because  is defined by rational formulas, the orbit remains in .  One can then ask:

Do transcendental orbits ever hit an algebraic point?

Are there transcendental cycles?  (I.e.\ periodic –points whose -coordinate is transcendental.)

What is the interaction between the Galois group of the extension  and the dynamics of ?



In many cases (e.g.\ function‑field analogues ) one can prove finiteness of cycles via Diophantine or geometric methods—offering a stark contrast to the number‑field case .


---

4. Analytic‑transcendental interpolation

An alternative way to bring in transcendental functions is via an analytic interpolation of the two branches: for example one can show 

f(x)\;=\;\frac{x}{2}\cos^2\!\bigl(\tfrac{\pi x}{2}\bigr)\;+\;\frac{3x+1}{2}\sin^2\!\bigl(\tfrac{\pi x}{2}\bigr)


---

Outlook

By packaging the Collatz rule into the single scheme‑morphism  over , one sees clearly:

Integers are just one fiber of the family of dynamics obtained by base‑change.

Transcendental numbers arise naturally when you study  for larger fields .

Questions about cycles become questions about rational points on algebraic varieties, amenable to tools from Diophantine geometry, Galois theory, and the theory of heights.


This field‑theoretic/algebraic‑geometric viewpoint thus both unifies all “base‑” Collatz problems and suggests powerful new invariants (moduli of periodic orbits, height‑growth under , Galois‑action on function‑field points) that may one day crack the classical conjecture.

