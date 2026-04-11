This is Grothendieck's **SGA3, Exposé VIII — "Groupes Diagonalisables"** (Diagonalizable Group Schemes). It is a deep piece of algebraic geometry in French. Below I'll unpack each major section and then build a genuine structural analogy to modern **Support Vector Machine (SVM) solvers** — the pairing is unusual but remarkably fertile, since both theories share a skeleton of **duality, graded decomposition, free actions, and representability**.

---

## § 1 — Biduality: D(G) and Reflexivity

### The algebraic geometry

The paper opens by defining the **character-dual functor**:

$$D(G) = \operatorname{Hom}_{\text{gr}}(G,\, \mathbb{G}_m)$$

where $\mathbb{G}_m$ is the multiplicative group scheme. For an ordinary abelian group $M$, the **diagonalizable group** is $D(M_S) = \operatorname{Spec} \mathcal{O}_S(M)$, the spectrum of the group algebra of $M$ over the base scheme $S$.

The central concept is **reflexivity**: there is a canonical map $G \to D(D(G))$, and $G$ is reflexive when this is an isomorphism — i.e., $G$ knows itself completely through its characters. Grothendieck proves that constant group schemes $M_S$ and all diagonalizable groups are reflexive, giving an **anti-equivalence** of categories.

### The SVM analogy: primal/dual and the kernel trick

In SVM theory, the primal problem optimises over a weight vector $\mathbf{w}$ in feature space; the **dual** problem optimises over support vectors $\alpha_i$ in data space. The move $\mathbf{w} \leftrightarrow \alpha$ is exactly a **character duality**: each data point $x_i$ plays the role of a "character" evaluated against the model $\mathbf{w}$.

Reflexivity corresponds to the **Representer Theorem**: the optimal $\mathbf{w}$ lives in the span of the data, so $\mathbf{w} = \sum_i \alpha_i k(\cdot, x_i)$. The double-dual map $G \to D(D(G))$ being an isomorphism is the algebraic avatar of this theorem — you can reconstruct the primal solution entirely from dual characters.

In scalable solvers like **LIBSVM**, **ThunderSVM**, and **cuSVM**, this biduality is exploited concretely: instead of working in a potentially infinite-dimensional feature space (the group algebra $\mathcal{O}_S(M)$), the solver works entirely with the $n \times n$ kernel (Gram) matrix — the dual side of the equivalence.

---

## § 2 — Schematic Properties of Diagonalizable Groups

### The algebraic geometry

Grothendieck catalogues flatness, finiteness, and smoothness of $D(M_S)$ in terms of $M$:

- $M$ finitely generated $\Leftrightarrow$ $D(M_S)$ of finite presentation over $S$
- $M$ a torsion group $\Leftrightarrow$ $D(M_S)$ is integral over $S$
- $M$ finitely generated with torsion of order coprime to the residual characteristics of $S$ $\Leftrightarrow$ $D(M_S)$ is **smooth** over $S$

The smoothness condition is the most important for the sequel: it governs whether the group scheme behaves like a classical Lie group fibrewise.

### The SVM analogy: kernel matrix conditioning

The kernel (Gram) matrix $K_{ij} = k(x_i, x_j)$ in an SVM is the algebraic incarnation of $\mathcal{O}_S(M)$. The schematic properties map as follows:

| Grothendieck | SVM / Kernel Methods |
|---|---|
| $M$ finitely generated | Finite training set, finite-rank kernel |
| $D(M_S)$ flat over $S$ | Kernel matrix is positive semi-definite |
| $D(M_S)$ smooth over $S$ | Kernel matrix is **strictly** positive definite (invertible) |
| Torsion in $M$ | Degenerate (rank-deficient) kernel directions |

In solvers like **sklearn's SVC** with an RBF kernel, the smoothness condition corresponds to the numerical stability of the QP solver: if the kernel matrix is degenerate (torsion analogue), the solver must regularise (add $\epsilon I$ to the diagonal), exactly as one passes from $D(M_S)$ to $D(M_{\text{free}})$ by killing the torsion subgroup.

---

## § 3 — Exactness of the Functor $D_S$

### The algebraic geometry

Given an exact sequence of abelian groups $0 \to M' \to M \to M'' \to 0$, the transposed sequence of diagonalizable groups:

$$0 \to D_S(M'') \to D_S(M) \xrightarrow{u^t} D_S(M') \to 0$$

is exact in the **faithfully flat quasi-compact (fpqc) topology**. The key point is that $u^t$ is faithfully flat, and $D_S(M')$ represents the fpqc quotient sheaf $D_S(M)/D_S(M'')$. Monomorphisms of groups dualize to faithfully flat maps; epimorphisms dualize to closed immersions.

### The SVM analogy: soft-margin SVMs and slack variables

The passage from a hard-margin SVM (exact separability: $0 \to M' \to M \to M'' \to 0$ splits cleanly) to a **soft-margin SVM** is the computational analogue of working with a non-split exact sequence. The slack variables $\xi_i \geq 0$ in:

$$y_i(\mathbf{w} \cdot x_i + b) \geq 1 - \xi_i$$

form the "cokernel" $M''$ of the inclusion of the correctly-classified margin into the full constraint group. The $C$-regularization parameter controls how much of this cokernel you tolerate — it is the quantitative measure of how far the sequence fails to split.

In **ADMM-based distributed SVM solvers** (e.g., those built on consensus ADMM), the exact-sequence structure is made explicit: the global constraint $Ax + Bz = c$ decomposes the optimization across nodes, and the dual variable $\lambda$ tracks the "obstruction to splitting" — exactly the extension class of $0 \to M' \to M \to M'' \to 0$.

---

## § 4 — Torsors under a Diagonalizable Group

### The algebraic geometry

A **$G$-torsor** (principal homogeneous space) over $S$ with $G = D_S(M)$ is classified by an $M$-graded algebra $\mathcal{A} = \bigoplus_{m \in M} \mathcal{A}_m$ where each $\mathcal{A}_m$ is an invertible sheaf (line bundle) and the multiplication $\mathcal{A}_m \otimes \mathcal{A}_{m'} \xrightarrow{\sim} \mathcal{A}_{m+m'}$ is an isomorphism.

The central special case $M = \mathbb{Z}$, $G = \mathbb{G}_m$ gives **Corollary 4.4**: $\mathbb{G}_m$-torsors are classified by $\operatorname{Pic}(S) = H^1(S, \mathcal{O}_S^\times)$, and every such torsor is Zariski-locally trivial (**Hilbert 90**). The remark 4.5.1 notes that for $\mu_n$ (roots of unity), non-trivial torsors over $\mathbb{R}$ exist, with $H^1_{\text{ét}}(\mathbb{R}, S^1) \cong \mathbb{Z}/2\mathbb{Z}$.

### The SVM analogy: the margin as a torsor; kernel feature spaces

The SVM **margin** $\gamma = 2/\|\mathbf{w}\|$ and the decision hyperplane $\mathbf{w} \cdot x + b = 0$ together define a structure that is precisely a $\mathbb{G}_m$-torsor over the solution manifold. Rescaling $(\mathbf{w}, b) \mapsto (\lambda \mathbf{w}, \lambda b)$ gives the same classifier — the classifier is not a vector but an element of a **principal $\mathbb{R}^\times$-bundle**.

The $M$-graded algebra $\mathcal{A} = \bigoplus_{m} \mathcal{A}_m$ has a direct analogue in the **feature space decomposition** of kernel methods: for a polynomial kernel $k(x,y) = (x \cdot y + c)^d$, the RKHS decomposes as:

$$\mathcal{H} = \bigoplus_{|\alpha| \leq d} \mathcal{H}_\alpha$$

where $\alpha$ ranges over multi-indices (playing the role of $M$), and the grading condition $\mathcal{A}_m \otimes \mathcal{A}_{m'} \cong \mathcal{A}_{m+m'}$ corresponds to the multiplicative structure of monomials. The **Nyström approximation** used in scalable solvers like **Falkon** or **EigenPro** is precisely a truncation of this graded decomposition to the most important "degrees" $m$.

---

## § 5 — Quotient of an Affine Scheme by a Freely Acting Diagonalizable Group

### The algebraic geometry

**Theorem 5.1** is the central structural result: if $G = D_S(M)$ acts freely on an affine $S$-scheme $P = \operatorname{Spec} \mathcal{A}$ (where $\mathcal{A}$ is $M$-graded), then the quotient $X = P/G$ exists as an affine scheme, and:

$$X = \operatorname{Spec}(\mathcal{A}_0), \quad \text{the degree-0 part of } \mathcal{A}.$$

Moreover, $P$ is a $G_X$-torsor over $X$. The proof hinges on showing that free action forces $\mathcal{A}_m \cdot \mathcal{A}_{-m} = \mathcal{A}_0$ for all $m \in M$ — the degree components are invertible and the grading is "perfect." The key technical lemma is that the graph morphism $P \times_S G \to P \times_S P$ is a closed immersion.

### The SVM analogy: the quotient as the invariant feature space; separating hyperplane

The quotient $X = P/G$ taking $\operatorname{Spec}(\mathcal{A}) \to \operatorname{Spec}(\mathcal{A}_0)$ is the algebraic counterpart of **projecting a data representation onto its $G$-invariant subspace**.

In **invariant kernel SVMs** (used for symmetry-aware classification, e.g., in molecular property prediction or image recognition), one explicitly builds a kernel that is invariant under a group $G$ acting on input space:

$$k_G(x, y) = \frac{1}{|G|} \sum_{g \in G} k(g \cdot x, y)$$

The resulting classifier lives in $\mathcal{A}_0 = \mathcal{A}^G$, the invariant subalgebra — exactly the quotient of Theorem 5.1. Solvers like **GeoSVM** or kernel methods on **group-equivariant neural networks** implement this quotient structure. The "free action" condition translates to the requirement that the symmetry group acts **without fixed points on the training data**, ensuring the kernel matrix restricted to the quotient remains invertible (no degenerate orbit structure).

The corollary that **the graph morphism is a closed immersion** (5.5) corresponds to the fact that in a freely-acting, separable configuration, the **support vectors are isolated** — no two training points collapse to the same point in the quotient space.

---

## § 6 — Essentially Free Morphisms and Representability

### The algebraic geometry

A morphism $f: X \to S$ is **essentially free** if locally over $S$ (after a faithfully flat base change), the coordinate ring of $X$ is a **free module** over that of $S$. Grothendieck proves that diagonalizable groups are essentially free (since their algebra is literally a free $\mathcal{O}_S$-module with basis $\{e_m\}_{m \in M}$).

The main **Theorem 6.4** says: if $Z \to S$ is essentially free and $Y \hookrightarrow Z$ is a closed subscheme, then the functor

$$F(S') = \begin{cases} \{\mathrm{id}\} & \text{if } Z_{S'} = Y_{S'} \\ \emptyset & \text{otherwise} \end{cases}$$

is **representable by a closed subscheme of** $S$. This is a powerful existence result for "loci where two things agree."

The applications in §6 include: the **centralizer** $C_G(Z)$ and **normalizer** $N_G(H)$ of subgroup schemes are closed subschemes when $G$ is essentially free — a result used repeatedly in the structure theory of reductive groups.

### The SVM analogy: active set methods and working set selection

In large-scale SVM solvers (SMO algorithm as in **LIBSVM**, or chunking algorithms), the key computational step is **working set selection**: identify the subset $\mathcal{W} \subseteq \{1,\ldots,n\}$ of indices where the KKT conditions are violated. The functor $F$ above — detecting the locus where $Z_{S'} = Y_{S'}$ — is the algebraic version of this: finding where the constraint is **tight** (active).

The representability theorem says this locus is a closed subscheme: in SVM terms, the active set is a **well-defined, finitely-describable subset** of the training data, not some pathological infinite object. Solvers like **LASVM** and the **decomposition method** exploit this: at each step, the working set is a closed, representable object whose boundary (KKT conditions) controls the next update.

The centralizer/normalizer constructions correspond to **structured SVMs** (e.g., for sequence labeling or parse trees), where the "group of symmetries" of the output space must be computed and quotiented out to get tractable prediction problems.

---

## § 7 — Appendix: Monomorphisms of Group Schemes

### The algebraic geometry

The final section asks: when is a **monomorphism** $u: G \to H$ of group schemes actually a **closed immersion**? Over a field (or in characteristic 0), monomorphisms of finite-type group schemes are automatically closed immersions (**Proposition 7.2**). In characteristic $p$, Grothendieck uses a careful analysis of the quotient $Q = H/G$ (a purely infinitesimal group) and shows that after passing to the $p^\nu$-power, the sequence:

$$0 \to p^\nu G \to p^\nu H \to Q \to 0$$

becomes exact for large $\nu$, implying the monomorphism is an immersion under mild flatness and separability assumptions.

### The SVM analogy: degenerate kernels and injective feature maps

An injective feature map $\phi: \mathcal{X} \to \mathcal{H}$ (the SVM analogue of a monomorphism $G \to H$) need not give a **closed embedding** of the data. If the kernel $k(x,y) = \langle \phi(x), \phi(y) \rangle$ is **not strictly positive definite**, the Gram matrix is rank-deficient — the map $\phi$ is injective but not a "closed immersion" into feature space; distinct points collide in kernel space.

The characteristic-$p$ phenomenon corresponds to **finite-field arithmetic** in SVMs over non-real fields (used in cryptographic ML), where injectivity of the feature map requires careful control of $p$-power torsion — exactly the $p^\nu$-analysis in §7.

In practice, solvers handle this via:
- **Jitter regularization** ($K \mapsto K + \varepsilon I$): kills the infinitesimal $Q$
- **Incomplete Cholesky factorisation** (used in **ICD-SVM**): detects rank deficiency and works in the image of the "immersion"
- **Random features** (Rahimi & Recht): replace the possibly non-injective exact feature map with a stochastic approximation that is injective with high probability

---

## Summary Table: SGA3 Exp. VIII ↔ SVM Concepts

| SGA3 Concept | SVM / Kernel Solver Analogue |
|---|---|
| Diagonalizable group $D(M_S)$ | Kernel (Gram) matrix; group algebra of features |
| Biduality $G \to D(D(G))$ | Representer Theorem; primal↔dual |
| Reflexivity | Kernel PCA / double dual embedding |
| $M$-graded algebra $\mathcal{A}$ | Feature space decomposition by degree |
| Degree-0 part $\mathcal{A}_0$ | Invariant subspace; projection to quotient |
| Flatness of $D(M_S)$ | PSD kernel matrix |
| Smoothness of $D(M_S)$ | Strictly PD kernel (invertible Gram matrix) |
| Torsion in $M$ | Degenerate / rank-deficient kernel directions |
| $\mathbb{G}_m$-torsor | SVM margin as principal bundle; scale invariance |
| Hilbert 90 (Cor. 4.5) | All $\mathbb{G}_m$-torsors trivial = RBF kernels are universal |
| Free action of $G$ on $P$ | Separable data configuration; isolated support vectors |
| Quotient $P/G = \operatorname{Spec}(\mathcal{A}_0)$ | Invariant kernel / quotient feature space |
| Essentially free morphism | Freely-expandable basis; Nyström approximation |
| Representability (Thm 6.4) | Active set is well-defined; KKT working set |
| Centralizer $C_G(Z)$ | Structured SVM output symmetry group |
| Monomorphism = closed immersion (§7) | Injective kernel feature map (strict PD) |
| $p$-power torsion / $Q = H/G$ | Rank deficiency in finite-precision kernel solvers |

The deepest unifying idea is this: both theories are fundamentally about **when a dual representation faithfully recovers the primal object** (reflexivity / the Representer Theorem), and **when a free action produces a clean, representable quotient** (Theorem 5.1 / separability + invariant kernels). Grothendieck's machinery makes this precise in the language of schemes; the SVM literature rediscovers it in the language of optimisation and functional analysis.
