# Spectral Properties and Walsh-Wiener Theory of Linear Feedback Shift Registers

## Executive Summary & Theoretical Synthesis

This document presents a rigorous, fully cross-checked algebraic, spectral, and functional analysis of Linear Feedback Shift Registers (LFSRs) and Filtered Generator Systems. We resolve previous ambiguities regarding Volterra expansions, Walsh character dualities, finite-field spectral decompositions, discrete **Walsh-Wiener Chaos Expansions**, and **LFSR Pair Spectral Synthesis**.

The central insight is that an LFSR over $\mathbb{F}_2$, multiplication in an extension field $\mathbb{F}_{2^L}$, Walsh characters on the Boolean hypercube $(\{-1, +1\}^L, \mu_{\text{uniform}})$, and multiplicative-character Gauss sums over $\mathbb{F}_{2^L}^\times$ are **a single mathematical object viewed in different coordinates**:

$$
\boxed{
\begin{array}{ccccccc}
\text{State Evolution} & \xrightarrow{\sim} & \text{Extension Field} & \xrightarrow{\sim} & \text{Trace Representation} & \xrightarrow{\sim} & \text{Gauss Sum / Koopman} \\
X_{n+1} = A X_n & & z_{n+1} = \alpha z_n & & u_n = \psi(\beta \alpha^n) & & U_k = \chi_k(\beta)^{-1} g(\chi_k, \psi)
\end{array}
}
$$

Furthermore, when observable outputs are formed by applying linear or non-linear Boolean filtering functions $f : \{-1, +1\}^L \to \mathbb{R}$ to LFSR states or combining outputs of LFSR pairs $(A, B)$, the output sequence $y_n$ admits a canonical **Rademacher-Walsh Wiener Chaos Decomposition** and an exact **Spectral Composition Algebra**.

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

---

## 4. Volterra-Wiener Series for Input-Driven Shift Register Systems

For input-driven binary systems fed by an independent input stream $w_n \in \{-1, +1\}$, the output sequence $v_n$ expands into the exact discrete **Volterra-Wiener Series Expansion**:

$$\boxed{
v_n = h_0 + \sum_{k \ge 0} h_1(k) w_{n-k} + \sum_{0 \le k_1 < k_2} h_2(k_1, k_2) w_{n-k_1} w_{n-k_2} + \sum_{0 \le k_1 < k_2 < k_3} h_3(k_1, k_2, k_3) w_{n-k_1} w_{n-k_2} w_{n-k_3} + \dots
}$$

where $h_m(k_1, \dots, k_m) = \mathbb{E}[v_n w_{n-k_1} \dots w_{n-k_m}]$ are the $m$-th order Volterra-Wiener interaction kernels.

---

## 5. LFSR Pair Spectral Synthesis Engine

To synthesize arbitrary target spectral compositions (e.g. non-flat spectral profiles, specific multi-line discrete harmonics, customizable cross-correlation peaks, or targeted Wiener chaos energy distributions), we construct combined dynamics using **LFSR Pairs**.

### 5.1 Joint Period & Pair Combination Algebra
Consider two independent primitive LFSRs $A$ and $B$ with extension degrees $L_A$ and $L_B$, primitive polynomials $p_A(t), p_B(t)$, and periods $N_A = 2^{L_A}-1$ and $N_B = 2^{L_B}-1$.
Let $u_n^{(A)} = \psi_A(\beta_A \alpha_A^n)$ and $u_n^{(B)} = \psi_B(\beta_B \alpha_B^n)$ be their respective bipolar output sequences.

The combined sequence $y_n$ is synthesized via a combination function $g : \{-1, +1\} \times \{-1, +1\} \to \mathbb{R}$:

$$y_n = g\left(u_n^{(A)}, u_n^{(B)}\right)$$

Common algebraic combinations include:
1. **Multiplicative Combination (XOR in binary)**: $y_n = u_n^{(A)} \cdot u_n^{(B)}$
2. **Additive Combination (Gold / Kasami type)**: $y_n = c_A u_n^{(A)} + c_B u_n^{(B)}$
3. **Multiplexed / Switching Combination**: $y_n = \frac{1 + u_n^{(C)}}{2} u_n^{(A)} + \frac{1 - u_n^{(C)}}{2} u_n^{(B)}$

The period of the synthesized output $y_n$ is:

$$\boxed{N_{\text{joint}} = \operatorname{lcm}(N_A, N_B) = \operatorname{lcm}\left(2^{L_A}-1, 2^{L_B}-1\right)}$$

When $\gcd(N_A, N_B) = 1$ (e.g., $L_A$ and $L_B$ coprime), $N_{\text{joint}} = N_A N_B$.

### 5.2 Synthesized Fourier Spectrum and Convolution Formula
The $N_{\text{joint}}$-point Discrete Fourier Transform $Y_m$ of the combined sequence $y_n$ decomposes into explicit tensor products of individual Gauss sum spectra:

1. **For Multiplicative Combination $y_n = u_n^{(A)} u_n^{(B)}$**:
By independence of $u^{(A)}$ and $u^{(B)}$, the joint DFT $Y_m$ is the circular convolution of the individual $N_{\text{joint}}$-extended spectra $U^{(A)}$ and $U^{(B)}$:

$$\boxed{Y_m = \frac{1}{N_{\text{joint}}} \sum_{k=0}^{N_{\text{joint}}-1} U_k^{(A)} U_{m-k \pmod{N_{\text{joint}}}}^{(B)}}$$

When $k \not\equiv 0 \pmod{N_A}$ and $m-k \not\equiv 0 \pmod{N_B}$, the spectral line magnitude satisfies:

$$|Y_m| = \sqrt{2^{L_A} \cdot 2^{L_B}} = 2^{(L_A + L_B)/2}$$

2. **For Additive Combination $y_n = c_A u_n^{(A)} + c_B u_n^{(B)}$**:
The joint spectrum consists of two distinct sets of spectral lines located at integer multiples of $\frac{N_{\text{joint}}}{N_A}$ and $\frac{N_{\text{joint}}}{N_B}$:

$$Y_m = c_A \frac{N_{\text{joint}}}{N_A} U_{m \bmod N_A}^{(A)} \cdot \delta_{\frac{N_{\text{joint}}}{N_A} \mid m} + c_B \frac{N_{\text{joint}}}{N_B} U_{m \bmod N_B}^{(B)} \cdot \delta_{\frac{N_{\text{joint}}}{N_B} \mid m}$$

This provides an exact **Spectral Synthesis Engine**: by choosing parameters $(L_A, p_A, \beta_A)$ and $(L_B, p_B, \beta_B)$ along with weights $c_A, c_B$, one can synthesize arbitrary multi-tiered discrete spectral power distributions.

### 5.3 Wiener Chaos Allocation of LFSR Pairs
For the pair combination space $\mathcal{B}_{L_A + L_B} = \{-1, +1\}^{L_A} \times \{-1, +1\}^{L_B}$, the combined function $g(u_A, u_B)$ distributes energy into joint chaos degrees $k = k_A + k_B$:

$$\boxed{E_{\text{pair}}(k) = \sum_{k_A + k_B = k} E_A(k_A) \cdot E_B(k_B)}$$

Multiplicative combination $y_n = u_n^{(A)} u_n^{(B)}$ shifts energy from degree 1 to degree 2, creating controlled higher-order chaos correlations.

---

## 6. Derived Theorems & Spectral Proofs

### Theorem 1: Exact Two-Valued Autocorrelation
For an $m$-sequence generated by a primitive polynomial of degree $L$ with period $N = 2^L - 1$, the periodic autocorrelation $R(d) = \sum_{n=0}^{N-1} u_n u_{n+d}$ satisfies:

$$\boxed{R(d) = \sum_{n=0}^{N-1} u_n u_{n+d} = \begin{cases} N & \text{if } d \equiv 0 \pmod N \\ -1 & \text{if } d \not\equiv 0 \pmod N \end{cases}}$$

### Theorem 2: Higher-Order Correlation Selection Rules
For a set of delays $D = \{d_1, d_2, \dots, d_m\}$, the higher-order product sum over one period satisfies:

$$\sum_{n=0}^{N-1} \prod_{d \in D} u_{n+d} = \begin{cases} N & \text{if } p(t) \mid \sum_{d \in D} t^d \text{ in } \mathbb{F}_2[t] \\ -1 & \text{otherwise} \end{cases}$$

### Theorem 3: Flat Discrete Fourier Spectrum and Gauss Sum Identification
Let $U_k = \sum_{n=0}^{N-1} u_n \omega^{-kn}$ be the $N$-point Discrete Fourier Transform of the sequence $u_n$, where $\omega = e^{2\pi i / N}$. Then:
1. $U_0 = -1$ (exact balance property).
2. For all $k \in \{1, 2, \dots, N-1\}$:
$$\boxed{U_k = \chi_k(\beta)^{-1} g(\chi_k, \psi)}$$
3. The power spectrum is perfectly flat:
$$\boxed{|U_k| = 2^{L/2} \implies |U_k|^2 = 2^L = N + 1 \quad \forall k \neq 0}$$

---

## Summary Matrix of LFSR Duality & Synthesis

| Domain / Coordinate System | Underlying Set | Basic Operation | Spectral Basis | Key Quantity |
| :--- | :--- | :--- | :--- | :--- |
| **State Vector Space** | $\mathbb{F}_2^L$ | $X_{n+1} = A X_n$ | Canonical unit vectors | State transition $A^n$ |
| **Extension Field** | $\mathbb{F}_{2^L}$ | $z_{n+1} = \alpha z_n$ | Trace map $\operatorname{Tr}(\beta z)$ | Primitive root $\alpha$ |
| **Bipolar Monomials** | $\{-1, +1\}^L$ | $u_{n+3} = u_{n+1} u_n$ | Walsh characters $\chi_S$ | 1-sparse index $S_n$ |
| **Wiener Chaos Space** | $L^2(\{-1,+1\}^L, \mu)$ | $W_k[f] = \sum_{|S|=k} \hat{f}(S) \chi_S$ | Chaos Subspaces $\mathcal{H}_k$ | Energy Spectrum $E_f(k)$ |
| **LFSR Pair Synthesis** | $\mathbb{F}_{2^{L_A}} \times \mathbb{F}_{2^{L_B}}$ | $y_n = g(u_n^{(A)}, u_n^{(B)})$ | Tensor Gauss Product | Joint Period $N_{\text{joint}} = \operatorname{lcm}(N_A, N_B)$ |
| **Time Domain (DFT)** | $\mathbb{Z}_N$ | Shift $n \to n+1$ | Fourier modes $e^{2\pi i k n / N}$ | Spectrum $|U_k| = 2^{L/2}$ |
| **Koopman / Characters** | $\mathbb{F}_{2^L}^\times$ | $(\mathcal{K}f)(z) = f(\alpha z)$ | Multiplicative $\chi_k$ | Gauss Sum $g(\chi_k, \psi)$ |
