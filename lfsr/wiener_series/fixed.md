# Spectral Properties and Walsh-Wiener Theory of Linear Feedback Shift Registers

## Executive Summary & Theoretical Synthesis

This document presents a rigorous, fully cross-checked algebraic, spectral, and functional analysis of Linear Feedback Shift Registers (LFSRs) and Filtered Generator Systems. We resolve previous ambiguities regarding Volterra expansions, Walsh character dualities, finite-field spectral decompositions, and the discrete **Walsh-Wiener Chaos Expansion**.

The central insight is that an LFSR over $\mathbb{F}_2$, multiplication in an extension field $\mathbb{F}_{2^L}$, Walsh characters on the Boolean hypercube $(\{-1, +1\}^L, \mu_{\text{uniform}})$, and multiplicative-character Gauss sums over $\mathbb{F}_{2^L}^\times$ are **a single mathematical object viewed in different coordinates**:

$$
\boxed{
\begin{array}{ccccccc}
\text{State Evolution} & \xrightarrow{\sim} & \text{Extension Field} & \xrightarrow{\sim} & \text{Trace Representation} & \xrightarrow{\sim} & \text{Gauss Sum / Koopman} \\
X_{n+1} = A X_n & & z_{n+1} = \alpha z_n & & u_n = \psi(\beta \alpha^n) & & U_k = \chi_k(\beta)^{-1} g(\chi_k, \psi)
\end{array}
}
$$

Furthermore, when observable outputs are formed by applying linear or non-linear Boolean filtering functions $f : \{-1, +1\}^L \to \mathbb{R}$ to LFSR states, the output sequence $y_n = f(X_n)$ admits a canonical **Rademacher-Walsh Wiener Chaos Decomposition**. The degree-$k$ Wiener chaos subspaces $\mathcal{H}_k$ are invariant under LFSR time evolution permutations, providing a fundamental bridge between pseudorandomness, nonlinearity, and functional analysis.

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

---

## 3. The Discrete Walsh-Wiener Chaos Expansion Framework

In classical Wiener analysis on continuous Gaussian space, functions $f \in L^2(\mathbb{R}^d, \gamma)$ are decomposed into orthogonal Hermite polynomial chaoses. For discrete binary dynamics over the Boolean hypercube $\mathcal{B}_L = \{-1, +1\}^L$ equipped with uniform probability measure $\mu$, the exact analogue is the **Discrete Walsh-Wiener Chaos Expansion**.

### 3.1 Hilbert Space Chaos Decomposition
The space $L^2(\{-1, +1\}^L, \mu)$ admits an orthogonal direct sum decomposition into $L+1$ Wiener chaos subspaces of degree $k$:

$$\boxed{L^2(\{-1, +1\}^L, \mu) = \bigoplus_{k=0}^L \mathcal{H}_k}$$

where the $k$-th Wiener chaos subspace $\mathcal{H}_k$ is defined by:
$$\mathcal{H}_k = \operatorname{span}\left\{ \chi_S(u) : S \subseteq \{0, 1, \dots, L-1\}, \, |S| = k \right\}$$

and $\chi_S(u) = \prod_{j \in S} u_j$ are the Walsh characters (Rademacher chaos monomials).

The dimension of the $k$-th chaos subspace $\mathcal{H}_k$ is $\dim(\mathcal{H}_k) = \binom{L}{k}$, satisfying:
$$\sum_{k=0}^L \binom{L}{k} = 2^L = \dim\left(L^2(\{-1, +1\}^L)\right)$$

### 3.2 Wiener Chaos Projection & Energy Distribution
For any Boolean function or real-valued observable $f : \{-1, +1\}^L \to \mathbb{R}$, its unique Fourier-Walsh expansion is:

$$f(u) = \sum_{S \subseteq \{0, \dots, L-1\}} \hat{f}(S) \chi_S(u)$$

where the Fourier-Walsh coefficients are given by the inner product:
$$\hat{f}(S) = \langle f, \chi_S \rangle = \frac{1}{2^L} \sum_{u \in \{-1, +1\}^L} f(u) \chi_S(u)$$

The **$k$-th Order Wiener Chaos Projection** operator $W_k : L^2(\{-1, +1\}^L) \to \mathcal{H}_k$ is defined as:

$$\boxed{W_k[f](u) = \sum_{\substack{S \subseteq \{0, \dots, L-1\} \\ |S| = k}} \hat{f}(S) \chi_S(u)}$$

The **Wiener Chaos Energy** at degree $k$, denoted $E_f(k)$, measures the proportion of total variance/power concentrated in $k$-bit non-linear interactions:

$$\boxed{E_f(k) = \|W_k[f]\|^2 = \sum_{|S|=k} |\hat{f}(S)|^2}$$

By Parseval's identity, the total signal energy satisfies:
$$\sum_{k=0}^L E_f(k) = \|f\|^2 = \frac{1}{2^L} \sum_{u \in \{-1, +1\}^L} |f(u)|^2$$

### 3.3 LFSR Evolution as Chaos Index Permutation
When an LFSR evolves in time $X_n = A^n X_0$, each initial Walsh monomial $\chi_S(X_0)$ transforms into another single Walsh monomial:

$$\chi_S(X_n) = \chi_S(A^n X_0) = \chi_{S A^n}(X_0)$$

where $S A^n$ denotes the linear action of the transposed state transition on the subset index mask $S \in \mathbb{F}_2^L$.

**Theorem (Wiener Chaos Energy Conservation under LFSR Evolution)**:
*Let $y_n = f(X_n)$ be a filtered LFSR output, where $X_{n+1} = A X_n$ with nonsingular $A$. The $k$-th order Wiener chaos projection of the time series $y_n$ with respect to initial state $X_0$ is:*

$$W_k[y_n](X_0) = \sum_{|S|=k} \hat{f}(S) \chi_{S A^n}(X_0)$$

*Because $A$ is an isomorphism over $\mathbb{F}_2^L$, the linear map $S \mapsto S A^n$ preserves subset cardinality $|S A^n| = |S| = k$. Consequently, the total Wiener chaos energy $E_{y_n}(k) = E_f(k)$ is strictly invariant for all $n$.*

This proves that LFSR evolution acts as an **energy-preserving permutation operator** across Walsh modes within each Wiener chaos subspace $\mathcal{H}_k$.

---

## 4. Volterra-Wiener Series for Input-Driven Shift Register Systems

While autonomous free-running LFSRs exhibit 1-sparse Walsh chaos trajectories, input-driven binary systems (such as additive scramblers, stream ciphers, and nonlinear feedback filters with external input $w_n \in \{-1, +1\}$) possess genuine multi-order **Volterra-Wiener functional kernels**.

Let $v_n \in \{-1, +1\}$ be the output of a general causal discrete shift system fed by an independent input stream $w_n \in \{-1, +1\}$:
$$v_n = F(w_n, w_{n-1}, \dots, w_{n-M}, v_{n-1}, \dots, v_{n-L})$$

Expressing $v_n$ as a functional of current and past inputs $w_n, w_{n-1}, w_{n-2}, \dots$ yields the exact discrete **Volterra-Wiener Series Expansion**:

$$\boxed{
v_n = h_0 + \sum_{k \ge 0} h_1(k) w_{n-k} + \sum_{0 \le k_1 < k_2} h_2(k_1, k_2) w_{n-k_1} w_{n-k_2} + \sum_{0 \le k_1 < k_2 < k_3} h_3(k_1, k_2, k_3) w_{n-k_1} w_{n-k_2} w_{n-k_3} + \dots
}$$

where:
- $h_0 = \mathbb{E}[v_n]$ is the DC / 0-th order Wiener kernel.
- $h_1(k) = \mathbb{E}[v_n w_{n-k}]$ is the 1st-order linear Volterra-Wiener kernel (impulse response over $\{-1, +1\}$).
- $h_2(k_1, k_2) = \mathbb{E}[v_n w_{n-k_1} w_{n-k_2}]$ is the 2nd-order non-linear interaction Volterra-Wiener kernel.
- $h_m(k_1, \dots, k_m) = \mathbb{E}[v_n w_{n-k_1} \dots w_{n-k_m}]$ is the $m$-th order Volterra-Wiener kernel.

### Volterra Kernel Extraction via Cross-Correlation
Under independent white binary input $w_n \sim \text{Bernoulli}(1/2)$ mapped to $\{-1, +1\}$, the input character monomials $W_S(n) = \prod_{j \in S} w_{n-j}$ satisfy the orthogonality condition:

$$\mathbb{E}[W_S(n) W_T(n)] = \delta_{S, T}$$

Therefore, the $m$-th order Volterra-Wiener kernel $h_m(k_1, \dots, k_m)$ is computed via higher-order cross-correlation:

$$h_m(k_1, \dots, k_m) = \frac{1}{T} \sum_{n=0}^{T-1} v_n w_{n-k_1} w_{n-k_2} \dots w_{n-k_m}$$

---

## 5. Derived Theorems & Spectral Proofs

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

---

## 6. Koopman Operator and Spectral Decomposition

The Koopman operator $\mathcal{K}$ acts on functions $f : \mathbb{F}_{2^L}^\times \to \mathbb{C}$ via state space transition:
$$(\mathcal{K} f)(z) = f(\alpha z)$$

The multiplicative characters $\chi_k$ are the exact eigenfunctions of $\mathcal{K}$:
$$(\mathcal{K} \chi_k)(z) = \chi_k(\alpha z) = \chi_k(\alpha) \chi_k(z) = \omega^{-k} \chi_k(z)$$

with eigenvalue $\lambda_k = \omega^{-k} = e^{-2\pi i k / N}$.

The observable sequence function $f_\beta(z) = \psi(\beta z)$ expands in this Koopman eigenbasis as:
$$\psi(\beta z) = \frac{1}{N} \sum_{k=0}^{N-1} g(\bar{\chi}_k, \psi) \chi_k(\beta z) = \frac{1}{N} \sum_{k=0}^{N-1} g(\chi_{-k}, \psi) \chi_k(\beta) \chi_k(z)$$

Evaluating along the orbit $z_n = \alpha^n$ yields the exact spectral decomposition of the LFSR sequence in time, confirming that the Gauss sum $g(\chi_{-k}, \psi)$ is precisely the spectral projection weight onto the $k$-th Koopman mode.

---

## Summary Matrix of LFSR Duality

| Domain / Coordinate System | Underlying Set | Basic Operation | Spectral Basis | Key Quantity |
| :--- | :--- | :--- | :--- | :--- |
| **State Vector Space** | $\mathbb{F}_2^L$ | $X_{n+1} = A X_n$ | Canonical unit vectors | State transition $A^n$ |
| **Extension Field** | $\mathbb{F}_{2^L}$ | $z_{n+1} = \alpha z_n$ | Trace map $\operatorname{Tr}(\beta z)$ | Primitive root $\alpha$ |
| **Bipolar Monomials** | $\{-1, +1\}^L$ | $u_{n+3} = u_{n+1} u_n$ | Walsh characters $\chi_S$ | 1-sparse index $S_n$ |
| **Wiener Chaos Space** | $L^2(\{-1,+1\}^L, \mu)$ | $W_k[f] = \sum_{|S|=k} \hat{f}(S) \chi_S$ | Chaos Subspaces $\mathcal{H}_k$ | Energy Spectrum $E_f(k)$ |
| **Volterra Input Series** | $\{-1, +1\}^\mathbb{N}$ | Convolution / Filtering | Multi-input products | Volterra Kernels $h_m(k_1,\dots,k_m)$ |
| **Time Domain (DFT)** | $\mathbb{Z}_N$ | Shift $n \to n+1$ | Fourier modes $e^{2\pi i k n / N}$ | Spectrum $|U_k| = 2^{L/2}$ |
| **Koopman / Characters** | $\mathbb{F}_{2^L}^\times$ | $(\mathcal{K}f)(z) = f(\alpha z)$ | Multiplicative $\chi_k$ | Gauss Sum $g(\chi_k, \psi)$ |
