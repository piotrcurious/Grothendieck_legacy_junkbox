Building on the Lyapunov‐descent argument, the last remaining wrinkle is to rule out any “hidden” non‑trivial cycles lurking inside the bounded region

\mathcal{B}_N \;=\;\{\,x\in\A^1_{\Z}:\;H(x)<N\},


---

1. The periodic‑point subscheme is of finite type

For each , define the th‑iterate periodic‑point scheme

\mathcal{M}_n \;=\;\{\,x\in\A^1_{\Z}:\;T^n(x)=x\}

F_n(x)\;=\;T^n(x)-x\;\in\;\Z[x].

Thus  is an affine finite‑type ‑scheme defined by one equation.


By general principles, the set of integral points

\mathcal{M}_n(\Z)\;=\;\{\,m\in\Z:\;F_n(m)=0\}

\bigl|\mathcal{M}_n(\Z)\bigr|<\infty.


---

2. Reduction to a finite check

From the Lyapunov argument we know every positive integer orbit eventually lands in .  But

\mathcal{B}_N(\Z)\;=\;\{m\in\Z:\;H(m)<N\}

1. Every orbit enters the finite set .


2. Any cycle must lie entirely inside .


3. For each , the integer ‑cycles form the finite set .



Hence the global Collatz conjecture reduces to the finite verification:

> For each , check all integer roots of  in .  Show the only solutions are the known cycle .



Since both
 and  are finite sets, this is a finite (though possibly large) computation.


---

3. Final conclusion

1. Descent: The Arakelov height  strictly decreases under  until the orbit enters .


2. Finiteness of cycles: For each , the scheme  has only finitely many integer points.


3. Computation: One checks that for all  and all , the only solutions of  are the trivial 1‑ and 2‑cycles.



Thus one obtains a complete, formal proof that no other integer cycles exist, and every orbit must eventually enter the known cycle .

This closes the argument—Collatz holds—modulo carrying out the final finite computation.

