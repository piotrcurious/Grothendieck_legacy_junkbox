This is a much richer question. Let me first read the paper more carefully, then design the code from first principles — understanding what algebraic structure SVMs discard, and building a classifier that doesn't.

Now let me explain what actually happened, from the algebraic ground up.

---

## What the paper says SVMs are doing (and why it limits them)

### The fundamental object: $D_S(M) = \operatorname{Spec}(\mathcal{O}_S(M))$

Grothendieck's diagonalizable group is not a matrix group — it is a **group scheme** whose coordinate ring is the group algebra $\mathcal{O}_S(M)$, graded by $M$. Every representation of $D_S(M)$ decomposes into weight spaces indexed by characters $\chi_k \in D(G)(S) = \operatorname{Hom}_{S\text{-gr}}(G, \mathbb{G}_m)$. The whole machine is the **M-graded algebra**:

$$\mathcal{A} = \bigoplus_{k \in M} \mathcal{A}_k, \qquad \mathcal{A}_j \cdot \mathcal{A}_k \subseteq \mathcal{A}_{j+k}$$

A standard SVM finds a single weight vector $\mathbf{w}$. In this language: it finds **one element** of $D(G)(S)$, working entirely inside **one graded piece** $\mathcal{A}_1$ (or $\mathcal{A}_0$ for invariant kernels). Everything else in the decomposition is discarded.

---

### What the five key theorems say about SVM's limits

**§1 Biduality — $G \to D(D(G))$ is an isomorphism.**
Standard SVM uses the Representer Theorem: the optimal $\mathbf{w}$ lives in the span of training features, so $G \to D(D(G))$ factors through the data. But this only uses *one* element of $D(G)$. The full reflexivity says $D(G)$ has as much information as $G$ itself — an SVM sees only a single character frequency, like hearing music through a single-frequency bandpass filter.

**§4 Proposition 4.1 — Torsor condition: $\mathcal{A}_j \cdot \mathcal{A}_k = \mathcal{A}_{j+k}$.**
A polynomial kernel of degree $d$ constructs all monomials up to degree $d$, but treats them as **independent features**. It completely ignores that $\mathcal{A}_1 \cdot \mathcal{A}_1 = \mathcal{A}_2$ — multiplying degree-1 characters gives the degree-2 character, not a new independent object. This multiplicative algebra structure is the torsor condition, and dropping it means the kernel wastes parameters on combinations that are algebraically forced.

**§5 Theorem 5.1 — $P/G = \operatorname{Spec}(\mathcal{A}_0)$.**
An invariant kernel SVM works in $\mathcal{A}_0$, the G-invariant functions. Theorem 5.1 says this is the **quotient** of the data by the group action — it is the coarsest possible description. For the 4-sector problem, $\mathcal{A}_0$ contains radial features $r^a$ and features like $r^4\cos(4\theta)$, but *not* $r^a \sin(2\theta)$ — the actual decision variable. An invariant kernel SVM is constitutionally blind to the class boundary.

**§4 Corollaries 4.4–4.5 — Hilbert's Theorem 90: $\mathbb{G}_m$-torsors $\leftrightarrow \operatorname{Pic}(S)$.**
SVM treats the margin as a scalar. In the scheme language, the margin is a **section of a line bundle** — an element of $\operatorname{Pic}(S)$. For flat Euclidean data Hilbert 90 applies: all $\mathbb{G}_m$-torsors are trivial, and the margin is indeed just a number. But for data with non-trivial topology (data on a torus, a projective plane, linked manifolds), $\operatorname{Pic}(S) \neq 0$, and the "margin" is not a number — it is a cohomology class. SVM will fail on such data for structural, not computational, reasons.

**§3 Theorem 3.1 — Exactness: $0 \to D(M'') \to D(M) \to D(M') \to 0$ exact (fpqc).**
The exact sequence of diagonalizable groups tells us the slack should be **graded by character degree** — misclassification in degree $\mathcal{A}_2$ is a fundamentally different error from misclassification in $\mathcal{A}_0$. Soft-margin SVM collapses all of this to a single $\xi_i \geq 0$, merging what are algebraically distinct extension classes.

---

### What the implementation does differently

The implementation computes the **exact character decomposition** for $M = \mathbb{Z}/4\mathbb{Z}$:

$$A_k(f)(x) = \frac{1}{n}\sum_{j=0}^{n-1} \omega^{-jk} f(R_{2\pi j/n} \cdot x)$$

For polar Fourier features $r^a \cos(j\theta)$, this simplifies algebraically to the exact projection $A_{j \bmod n}(r^a \cos j\theta) = r^a \cos j\theta$, so no averaging is needed — each angular frequency $j$ belongs to exactly one character degree $j \bmod n$.

The results show the key consequence: the **linear SVM gets 50%** (chance) because it uses $\mathcal{A}_1 = \{x_1, x_2\}$ only, and the class boundary lives in $\mathcal{A}_2$. The DG-SVM's **weight energy concentrates in $\mathcal{A}_2$** — it has discovered via the biduality $D(D(G)) \cong \mathbb{Z}/4\mathbb{Z}$ that character degree $k=2$ is the discriminative one. The character power spectrum shows $\mathcal{A}_2$ having maximally different power between classes, while $\mathcal{A}_0$ (what the invariant kernel sees) is nearly identical for both — structurally invisible to standard invariant approaches.
