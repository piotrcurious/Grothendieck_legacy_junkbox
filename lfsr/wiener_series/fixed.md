# Spectral Properties and Walsh-Wiener Theory of Linear Feedback Shift Registers

## Executive Summary & Theoretical Synthesis

This document presents a rigorous, fully cross-checked algebraic, spectral, and functional analysis of Linear Feedback Shift Registers (LFSRs). We resolve previous ambiguities regarding Volterra expansions, Walsh character dualities, and finite-field spectral decompositions.

The central insight is that an LFSR over $\mathbb{F}_2$, multiplication in an extension field $\mathbb{F}_{2^L}$, Walsh characters on the Boolean hypercube $(\{-1, +1\}^L, \times)$, and multiplicative-character Gauss sums over $\mathbb{F}_{2^L}^\times$ are **a single mathematical object viewed in different coordinates**:

$$
\boxed{
\begin{array}{ccccccc}
\text{State Evolution} & \xrightarrow{\sim} & \text{Extension Field} & \xrightarrow{\sim} & \text{Trace Representation} & \xrightarrow{\sim} & \text{Gauss Sum / Koopman} \\
X_{n+1} = A X_n & & z_{n+1} = \alpha z_n & & u_n = \psi(\beta \alpha^n) & & U_k = \chi_k(\beta)^{-1} g(\chi_k, \psi)
\end{array}
}
$$

---

## 1. Exact Finite Field Dynamics and Trace Representation

Let $p(t) = t^L + c_{L-1}t^{L-1} + \dots + c_1 t + c_0 \in \mathbb{F}_2[t]$ be a primitive polynomial of degree $L$.
The vector space state $X_n = (x_n, x_{n+1}, \dots, x_{n+L-1})^T \in \mathbb{F}_2^L$ evolves according to the companion matrix $A$:

$$X_{n+1} = A X_n, \quad A = \begin{pmatrix} 0 & 1 & 0 & \dots & 0 \\ 0 & 0 & 1 & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ c_0 & c_1 & c_2 & \dots & c_{L-1} \end{pmatrix}$$

Since $p(t)$ is irreducible, the quotient ring $K = \mathbb{F}_2[t]/(p(t))$ is isomorphic to the Galois extension field $\mathbb{F}_{2^L}$. Under this isomorphism, vector space states correspond to field elements $z_n \in \mathbb{F}_{2^L}$, and multiplication by companion matrix $A$ corresponds to field multiplication by the primitive root $\alpha = [t] \in \mathbb{F}_{2^L}^\times$:

$$z_{n+1} = \alpha z_n \implies z_n = \alpha^n z_0$$

The field trace map $\operatorname{Tr}_{K/\mathbb{F}_2} : \mathbb{F}_{2^L} \to \mathbb{F}_2$ is defined by:
$$\operatorname{Tr}(z) = \sum_{i=0}^{L-1} z^{2^i}$$

By non-degeneracy of the trace bilinear form $(z, w) \mapsto \operatorname{Tr}(zw)$, every linear functional on $\mathbb{F}_{2^L}$ can be represented as $z \mapsto \operatorname{Tr}(\beta z)$ for a uniquely determined initial state parameter $\beta \in \mathbb{F}_{2^L}^\times$. Thus, the sequence bit $x_n \in \mathbb{F}_2$ and its bipolar sign counterpart $u_n = (-1)^{x_n} \in \{-1, +1\}$ are given exactly by:

$$\boxed{x_n = \operatorname{Tr}(\beta \alpha^n), \qquad u_n = \psi(\beta \alpha^n)}$$

where $\psi(z) := (-1)^{\operatorname{Tr}(z)}$ is the canonical additive character of $\mathbb{F}_{2^L}$.

---

## 2. Conjugacy of $\pm 1$ Recoding and Walsh Monomial Sparsity

The mapping $x \mapsto (-1)^x$ is a group isomorphism $(\mathbb{F}_2, +) \cong (\{-1, +1\}, \times)$.
Under this transformation, addition modulo 2 becomes real multiplication:
$$x + y \pmod 2 \longrightarrow (-1)^{x+y} = (-1)^x (-1)^y$$

For example, the Fibonacci recurrence $x_{n+3} = x_{n+1} + x_n \pmod 2$ transforms into the multiplicative monomial recurrence:
$$u_{n+3} = u_{n+1} u_n$$

### Clarification on Volterra vs. Walsh Extensions
1. **Conjugacy, Not Nonlinearity**: The $\pm 1$ transformation is an exact algebraic conjugacy. The dynamics remain linear over $\mathbb{F}_2$; the multiplicative recurrence over $\{-1, +1\}$ is simply the same linear dynamics written in multiplicative coordinates.
2. **1-Sparsity in the Seed**: Because $x_n$ is a linear combination of initial seed bits $(x_0, \dots, x_{L-1})$, $u_n$ is **exactly one Walsh monomial** of the initial bipolar seed $u^{(0)} = (u_0, \dots, u_{L-1})$:
$$u_n = \chi_{S_n}(u^{(0)}) = \prod_{j \in S_n} u_j, \qquad S_n = \operatorname{supp}\left(e_1^T A^n\right)$$
In a discrete Wiener-Walsh chaos expansion $f(u) = \sum_{S \subseteq \{0,\dots,L-1\}} \hat{f}(S) \chi_S(u)$, the LFSR output at time $n$ has spectrum $\hat{u}_n(S) = \delta_{S, S_n}$. It is 1-sparse. This extreme sparsity reflects the absence of true non-linear interactions across seed bits, which is precisely why the Berlekamp-Massey algorithm can recover the entire state generator from just $2L$ stream bits.
3. **Genuine Volterra Systems**: Genuine Volterra series describe nonlinear responses to an external input stream $w_n$. An input-driven additive scrambler $s_n = w_n \oplus \sum_{j=1}^L c_j s_{n-j}$ yields in bipolar form $v_n = (-1)^{w_n} \prod_{k} v_{n-k}^{h_k}$, where $h_k$ is the impulse response over $\mathbb{F}_2$. Autonomous LFSRs are free-running systems and thus represent pure monomial orbit trajectories rather than multi-order input kernels.

---

## 3. Character Duality and Unification

There are not three independent bases; rather, there is a fundamental duality:

1. **Additive Characters / Walsh Basis**: The $2^L$ additive characters of $\mathbb{F}_{2^L}$ are given by $\psi_a(z) = \psi(a z) = (-1)^{\operatorname{Tr}(a z)}$ for $a \in \mathbb{F}_{2^L}$. Under a choice of basis $\mathbb{F}_{2^L} \cong \mathbb{F}_2^L$, these are identical to the $2^L$ Walsh characters $\chi_S(u) = \prod_{j \in S} u_j$ on the Boolean hypercube.
2. **Multiplicative Characters / Time-Domain Fourier Basis**: Time-domain spectral analysis over one period $N = 2^L - 1$ operates on the cyclic group $\mathbb{Z}_N \cong \mathbb{F}_{2^L}^\times$. The multiplicative characters of $\mathbb{F}_{2^L}^\times$ are $\chi_k(\alpha^n) = e^{-2\pi i k n / N} = \omega^{-kn}$, where $\omega = e^{2\pi i / N}$.

The change of basis between the additive character domain ($\mathbb{F}_{2^L}$) and the multiplicative character domain ($\mathbb{F}_{2^L}^\times$) is mediated precisely by **Gauss sums**:

$$g(\chi_k, \psi) = \sum_{z \in \mathbb{F}_{2^L}^\times} \chi_k(z) \psi(z)$$

---

## 4. Derived Theorems & Spectral Proofs

### Theorem 1: Exact Two-Valued Autocorrelation
For an $m$-sequence generated by a primitive polynomial of degree $L$ with period $N = 2^L - 1$, the periodic autocorrelation $R(d) = \sum_{n=0}^{N-1} u_n u_{n+d}$ satisfies:

$$\boxed{R(d) = \sum_{n=0}^{N-1} u_n u_{n+d} = \begin{cases} N & \text{if } d \equiv 0 \pmod N \\ -1 & \text{if } d \not\equiv 0 \pmod N \end{cases}}$$

*Proof*:
Using $u_n = \psi(\beta \alpha^n)$ and character additivity $\psi(a)\psi(b) = \psi(a+b)$:
$$u_n u_{n+d} = \psi(\beta \alpha^n) \psi(\beta \alpha^{n+d}) = \psi\left(\beta \alpha^n (1 + \alpha^d)\right)$$
As $n$ ranges over $0, 1, \dots, N-1$, the element $z = \alpha^n$ traverses all non-zero elements of $\mathbb{F}_{2^L}^\times$ exactly once.
- Case 1: If $d \equiv 0 \pmod N$, then $\alpha^d = 1$. In characteristic 2, $1 + \alpha^d = 1 + 1 = 0$. Thus $\psi(0) = (-1)^0 = 1$ for all $n$, giving $R(0) = \sum_{n=0}^{N-1} 1 = N$.
- Case 2: If $d \not\equiv 0 \pmod N$, then $\gamma = \beta(1 + \alpha^d) \neq 0$. As $\alpha^n$ ranges over $\mathbb{F}_{2^L}^\times$, $\gamma \alpha^n$ ranges bijectively over $\mathbb{F}_{2^L}^\times$.
Using the orthogonality of additive characters $\sum_{z \in \mathbb{F}_{2^L}} \psi(z) = 0$:
$$\sum_{n=0}^{N-1} u_n u_{n+d} = \sum_{z \in \mathbb{F}_{2^L}^\times} \psi(\gamma z) = \left(\sum_{z \in \mathbb{F}_{2^L}} \psi(z)\right) - \psi(0) = 0 - 1 = -1 \quad \blacksquare$$

### Theorem 2: Higher-Order Correlation Selection Rules
For a set of delays $D = \{d_1, d_2, \dots, d_m\}$, the higher-order product sum over one period satisfies:

$$\sum_{n=0}^{N-1} \prod_{d \in D} u_{n+d} = \begin{cases} N & \text{if } p(t) \mid \sum_{d \in D} t^d \text{ in } \mathbb{F}_2[t] \\ -1 & \text{otherwise} \end{cases}$$

*Proof*:
$$\prod_{d \in D} u_{n+d} = \psi\left(\beta \alpha^n \sum_{d \in D} \alpha^d\right)$$
The term $\sum_{d \in D} \alpha^d = 0$ in $\mathbb{F}_{2^L}$ if and only if $\alpha$ is a root of the polynomial $Q(t) = \sum_{d \in D} t^d$. Since $p(t)$ is the minimal polynomial of $\alpha$, $\alpha$ is a root if and only if $p(t) \mid Q(t)$.
When $p(t) \mid Q(t)$, every summand is $\psi(0) = 1$, yielding $N$. Otherwise, the argument ranges over $\mathbb{F}_{2^L}^\times$, yielding $-1$. $\blacksquare$

### Theorem 3: Flat Discrete Fourier Spectrum and Gauss Sum Identification
Let $U_k = \sum_{n=0}^{N-1} u_n \omega^{-kn}$ be the $N$-point Discrete Fourier Transform of the sequence $u_n$, where $\omega = e^{2\pi i / N}$. Then:
1. $U_0 = -1$ (exact balance property).
2. For all $k \in \{1, 2, \dots, N-1\}$:
$$\boxed{U_k = \chi_k(\beta)^{-1} g(\chi_k, \psi)}$$
3. The power spectrum is perfectly flat:
$$\boxed{|U_k| = 2^{L/2} \implies |U_k|^2 = 2^L = N + 1 \quad \forall k \neq 0}$$

*Proof*:
1. For $k = 0$, $U_0 = \sum_{n=0}^{N-1} u_n = \sum_{z \in \mathbb{F}_{2^L}^\times} \psi(\beta z) = -1$.
2. For $k \neq 0$, substitute $u_n = \psi(\beta \alpha^n)$ and $\omega^{-kn} = \chi_k(\alpha^n)$:
$$U_k = \sum_{n=0}^{N-1} \psi(\beta \alpha^n) \chi_k(\alpha^n)$$
Change variables $z = \beta \alpha^n \implies \alpha^n = \beta^{-1} z$. Since $\chi_k$ is a multiplicative character:
$$\chi_k(\alpha^n) = \chi_k(\beta^{-1} z) = \chi_k(\beta)^{-1} \chi_k(z)$$
Substituting this back into the sum:
$$U_k = \sum_{z \in \mathbb{F}_{2^L}^\times} \psi(z) \chi_k(\beta)^{-1} \chi_k(z) = \chi_k(\beta)^{-1} \sum_{z \in \mathbb{F}_{2^L}^\times} \chi_k(z) \psi(z) = \chi_k(\beta)^{-1} g(\chi_k, \psi)$$
3. Since $\chi_k(\beta)$ lies on the complex unit circle, $|\chi_k(\beta)| = 1$. By the classical field theory property of non-trivial Gauss sums over $\mathbb{F}_q$ ($q = 2^L$), $|g(\chi_k, \psi)| = \sqrt{q} = 2^{L/2}$.
Therefore, $|U_k| = 2^{L/2}$, and $|U_k|^2 = 2^L = N + 1$. $\blacksquare$

---

## 5. Koopman Operator and Spectral Decomposition

The Koopman operator $\mathcal{K}$ acts on functions $f : \mathbb{F}_{2^L}^\times \to \mathbb{C}$ via state space transition:
$$(\mathcal{K} f)(z) = f(\alpha z)$$

The multiplicative characters $\chi_k$ are the exact eigenfunctions of $\mathcal{K}$:
$$(\mathcal{K} \chi_k)(z) = \chi_k(\alpha z) = \chi_k(\alpha) \chi_k(z) = \omega^{-k} \chi_k(z)$$

with eigenvalue $\lambda_k = \omega^{-k} = e^{-2\pi i k / N}$.

The observable sequence function $f_\beta(z) = \psi(\beta z)$ expands in this Koopman eigenbasis as:
$$\psi(\beta z) = \frac{1}{N} \sum_{k=0}^{N-1} g(\bar{\chi}_k, \psi) \chi_k(\beta z) = \frac{1}{N} \sum_{k=0}^{N-1} g(\chi_{-k}, \psi) \chi_k(\beta) \chi_k(z)$$

Evaluating along the orbit $z_n = \alpha^n$ yields the exact spectral decomposition of the LFSR sequence in time, confirming that the Gauss sum $g(\chi_{-k}, \psi)$ is precisely the spectral projection weight onto the $k$-th Koopman mode.

---

## 6. Generalizations: Non-Primitive Polynomials, Reducible Rings, and NFSRs

1. **Non-Primitive Irreducible Polynomials**: If $p(t)$ is irreducible of degree $L$ but non-primitive, its root $\alpha$ generates a sub-group of $\mathbb{F}_{2^L}^\times$ of order $M = \operatorname{ord}(\alpha) < 2^L - 1$, where $M \mid (2^L - 1)$. The output period is $M$, and the spectrum consists of impulses scaled by sub-field character sums.
2. **Reducible Polynomials & Ring Decomposition**: When $p(t) = \prod_{i=1}^m p_i(t)^{e_i}$, the Chinese Remainder Theorem splits the state ring:
$$R = \mathbb{F}_2[t]/(p(t)) \cong \bigoplus_{i=1}^m \mathbb{F}_2[t]/(p_i(t)^{e_i})$$
Factors with $e_i = 1$ correspond to pure periodic orbits over subfields $\mathbb{F}_{2^{\deg p_i}}$. Repeated factors ($e_i > 1$) introduce nilpotent elements $N^{e_i} = 0$, generating polynomial-modulated polynomial drift terms (e.g., $n^k \alpha^n$) on top of periodic motion.
3. **Nonlinear Feedback Shift Registers (NFSRs)**: For nonlinear update functions $x_{n+L} = f(x_n, \dots, x_{n+L-1})$, the state map $F : \mathbb{F}_2^L \to \mathbb{F}_2^L$ is a polynomial map on the Boolean variety $V(x_0^2 - x_0, \dots, x_{L-1}^2 - x_{L-1})$. Spectral analysis proceeds via the full Koopman matrix or Walsh transform of $F$. Orbit structure, cycle decompositions, and algebraic immunity are determined by the Gröbner basis of the iteration ideal $I_k = \langle F^{\circ k}(x) - x \rangle$.

---

## Summary Matrix of LFSR Duality

| Domain / Coordinate System | Underlying Set | Basic Operation | Spectral Basis | Key Quantity |
| :--- | :--- | :--- | :--- | :--- |
| **State Vector Space** | $\mathbb{F}_2^L$ | $X_{n+1} = A X_n$ | Canonical unit vectors | State transition $A^n$ |
| **Extension Field** | $\mathbb{F}_{2^L}$ | $z_{n+1} = \alpha z_n$ | Trace map $\operatorname{Tr}(\beta z)$ | Primitive root $\alpha$ |
| **Bipolar Monomials** | $\{-1, +1\}^L$ | $u_{n+3} = u_{n+1} u_n$ | Walsh characters $\chi_S$ | 1-sparse index $S_n$ |
| **Time Domain (DFT)** | $\mathbb{Z}_N$ | Shift $n \to n+1$ | Fourier modes $e^{2\pi i k n / N}$ | Spectrum $|U_k| = 2^{L/2}$ |
| **Koopman / Characters** | $\mathbb{F}_{2^L}^\times$ | $(\mathcal{K}f)(z) = f(\alpha z)$ | Multiplicative $\chi_k$ | Gauss Sum $g(\chi_k, \psi)$ |
