LFSRs, Finite Fields, and Character Sums
Same conclusion as the draft — an LFSR, multiplication in a finite field, and a character expansion are one object in different coordinates — but with the Volterra framing corrected, three "bases" collapsed into the one duality that's actually there, and the character-sum computations it promised but never did.

What's fixed

Volterra series describe a response to an input; a free-running LFSR has none. What the draft's monomial recurrence actually shows is that every output bit is one Walsh monomial of the seed. Genuine Volterra language belongs to input-driven constructions — noted below, where it's earned.
"Nonlinear recurrence" was the same linear system after a change of coordinates, not new dynamics.
Walsh characters and the finite-field characters \(\psi(a\cdot)\) are the same functions, not two bases out of three.
The autocorrelation and flat-spectrum claims the draft gestures at are derived here from one character-sum identity.
The closing comparison table is dropped — there's no inner product over \(\mathbf F_2\) for it to rest on.
1. The exact algebra
Take the running example \(p(t)=t^3+t+1\), companion matrix \(A\), state \(X_n=(x_n,x_{n+1},x_{n+2})\), \(X_{n+1}=AX_n\) over \(\mathbf F_2\) — this part of the draft was already right. For \(p\) irreducible, \(F_2[t]/(p)\) is a field \(K\cong\mathbf F_{2^L}\), the module structure collapses to a one-dimensional \(K\)-vector space, and \(A\)-multiplication becomes multiplication by \(\alpha=[t]\):

\[z_{n+1}=\alpha z_n,\qquad z_n=\alpha^n z_0.\]
The trace form \((z,w)\mapsto\operatorname{Tr}_{K/\mathbf F_2}(zw)\) is nondegenerate, so every nonzero \(\mathbf F_2\)-linear functional \(K\to\mathbf F_2\) is \(z\mapsto\operatorname{Tr}(\gamma z)\) for a unique \(\gamma\). Reading off one coordinate of the state is such a functional, so for a fixed \(\beta\) determined by the seed:

\[x_n=\operatorname{Tr}(\beta\alpha^n),\qquad u_n:=(-1)^{x_n}=\psi(\beta\alpha^n),\quad \psi(z):=(-1)^{\operatorname{Tr}(z)}.\]
This is the whole object. Everything below just reads off consequences of it.

2. The \(\pm1\) recoding is a conjugacy
\(x\mapsto(-1)^x\) is a group isomorphism \((\mathbf F_2,+)\cong(\{\pm1\},\times)\), so \(u_{n+3}=u_nu_{n+1}\) is the same linear system relabeled, not a nonlinear one. Because the recurrence is \(\mathbf F_2\)-linear in the seed, \(x_n\) is an XOR of seed bits, so \(u_n\) is always exactly one Walsh monomial of the seed: \(u_n=\chi_{S_n}(u_0,\dots,u_{L-1})\), with \(S_n\) the support of row \(n\) of \(A^n\). One nonzero Walsh coefficient is the signature of linearity — it's why Berlekamp–Massey recovers the whole register from \(2L\) bits. Calling this a "sparse Volterra kernel" has it backwards: it's sparse because there's no nonlinearity, not despite it.

Genuine Volterra structure needs an actual input stream. An additive scrambler \(s_n=w_n\oplus\sum_jc_js_{n-j}\), fed by input \(w\), gives — in \(\pm1\) form — \(v_n=m_n\cdot\prod_kv_{n-k}^{h_k}\) for the \(\mathbf F_2\) impulse response \(h\). That degree-bounded monomial-in-the-input is a real Volterra kernel; it just isn't what an autonomous LFSR has.

3. One duality, not three
\(\psi(a\cdot)\) for \(a\in K\) gives all \(2^L\) additive characters of \(K\), indexed by field elements. Walsh \(\chi_S\) gives the same \(2^L\) characters of \(\mathbf F_2^L\), indexed by subsets. Fix a basis identifying \(K\cong\mathbf F_2^L\) and \(\psi(a\cdot)=\chi_{S(a)}\) — literally the same functions under relabeling, not two items on a list of three.

The genuinely second object is a different Fourier transform: not on the seed bits, but in time \(n\), over \(\mathbf Z_N\) with \(N=2^L-1=\operatorname{ord}(\alpha)\). That transform is built from multiplicative characters of \(K^\times\), and Gauss sums are exactly the change of basis between the two.

4. The payoff
Two consequences of the one identity in §1, usually asserted rather than shown.

Autocorrelation. As \(n\) runs over one period, \(\alpha^n\) hits every nonzero element of \(K\) once. Since \(\psi(a)\psi(b)=\psi(a+b)\):

\[u_nu_{n+d}=\psi\bigl(\beta\alpha^n(1+\alpha^d)\bigr).\]
If \(d\equiv0\), the factor \(1+\alpha^d\) is \(0\) (characteristic 2), so every term is \(1\) and the sum is \(N\). If \(d\not\equiv0\), \(1+\alpha^d\neq0\) and \(\beta\alpha^n(1+\alpha^d)\) ranges bijectively over \(K^\times\) as \(n\) does, so the sum is \(\sum_{K^\times}\psi=-1\) (using \(\sum_K\psi=0\) for \(\psi\) nontrivial):

\[\sum_{n=0}^{N-1}u_nu_{n+d}=\begin{cases}N & d\equiv0\ (\mathrm{mod}\ N)\\[2pt]-1 & \text{otherwise.}\end{cases}\]
The same computation for a general delay set \(D\) gives \(\sum_n\prod_{d\in D}u_{n+d}=N\) if \(\sum_{d\in D}\alpha^d=0\), else \(-1\) — and \(\sum_{d\in D}\alpha^d=0\) exactly when \(p(t)\mid\sum_{d\in D}t^d\), since \(p\) is \(\alpha\)'s minimal polynomial. So the relations that hold identically along the sequence (every term forced to \(+1\), not just averaging to \(-1\)) are indexed precisely by multiples of the feedback polynomial. \(D=\{0,1,3\}\) for \(p=t^3+t+1\) is the lowest-degree case — \(1+\alpha+\alpha^3=0\) is just \(\alpha^3=\alpha+1\), i.e. the defining recursion itself — and every other multiple of \(p\) hands you another, less obvious, identically-satisfied relation.

Flat spectrum. Define the DFT over \(\mathbf Z_N\): \(U_k=\sum_nu_n\omega^{-kn}\), \(\omega=e^{2\pi i/N}\). For \(k=0\), \(U_0=\sum_{K^\times}\psi(\beta\cdot)=-1\) — matching the m-sequence balance property (\(2^{L-1}\) minus-ones against \(2^{L-1}-1\) plus-ones) exactly, not approximately. For \(k\neq0\), reindex \(n\leftrightarrow z=\alpha^n\); writing \(\chi_k(\alpha^n):=\omega^{-kn}\) for the multiplicative character this defines on \(K^\times\):

\[U_k=\chi_k(\beta)^{-1}\sum_{z\in K^\times}\psi(z)\chi_k(z)=\chi_k(\beta)^{-1}\,g(\chi_k,\psi),\]
a classical Gauss sum. Since \(|\chi_k(\beta)|=1\) and \(|g(\chi,\psi)|=\sqrt q\) for any nontrivial \(\chi,\psi\) over \(\mathbf F_q\):

\[|U_k|=2^{L/2}\quad\text{for every } k=1,\dots,N-1.\]
Exactly flat, not approximately. And it's an eigen-decomposition in disguise: on \(K^\times\), the \(\chi_k\) are eigenfunctions of the Koopman operator \((Uf)(z)=f(\alpha z)\) with eigenvalue \(\omega^{-k}\), and \(g(\chi_k,\psi)\) is the coefficient of \(\psi(\beta\cdot)\) in that eigenbasis. That's the precise content behind the draft's closing table — the spectral theory lives entirely on the complex-valued observables, with no inner product on \(\mathbf F_2\) required.

Two loose ends, briefly. For reducible \(p\), CRT splits \(K\) into a product of local rings \(F_2[t]/(p_i^{e_i})\); each factor is a field only when \(e_i=1\) — repeated factors add a nilpotent drift on top of the periodic part. For nonlinear feedback, §12's "ideal capturing the recurrence" was underspecified — one relation isn't enough. The useful object is the polynomial self-map \(F\) on the Boolean variety (shift-and-apply-\(f\)); periodic points, invariant subvarieties, and algebraic immunity all come from Gröbner/elimination methods applied to \(F,F\circ F,\dots\), not to a single relation in isolation.

\[K^\times\ \xrightarrow{\ \times\alpha\ }\ K^\times\ \xrightarrow{\ \operatorname{Tr}(\beta\cdot)\ }\ \mathbf F_2\ \xrightarrow{\ (-1)^x\ }\ \{\pm1\}\]
§4 is this diagram read in the frequency domain.
