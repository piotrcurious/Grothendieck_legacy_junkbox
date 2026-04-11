That is a better angle.

The clean way to read SGA3 VIII is not “diagonalizable groups are like SVMs,” but rather:

an SVM is what you get after collapsing geometry to a chosen feature algebra, whereas SGA3 keeps track of the full grading and the quotient by a diagonalizable symmetry.

So the paper is richer than standard SVMs. Standard SVMs usually start after a simplification has already happened: you have chosen a feature map or kernel, and then the classifier only sees inner products. In the language of the paper, that means you have already forgotten most of the grading data and retained only a compressed shadow of the representation.

The right lens

Take a diagonalizable group

G = D(M)

A = \bigoplus_{m\in M} A_m.

That grading is the real structure.

An ordinary SVM, by contrast, usually works with a function

f(x) = \mathrm{sign}\Big(\sum_i \alpha_i k(x_i,x) + b\Big),

So the more faithful interpretation is:

SGA3 viewpoint: keep the grading, then quotient by the symmetry, then classify on the invariant part.

Standard SVM viewpoint: choose a feature map, then classify in the compressed space, with little explicit record of why that compression is legitimate.


That is why SVMs are often limited: they work best when the decision boundary is already well expressed in the chosen collapsed feature space. If the true geometry lives in different weight spaces, or if the relevant symmetry is not the one the kernel respects, the SVM has no built-in mechanism to recover that structure.

A diagonalizable-group SVM

Here is a real toy model where the group is doing the work.

Let

G = \mathbb G_m

t\cdot (x,y) = (tx,t^{-1}y).

This is a diagonalizable action. The coordinate ring is

k[x,y],

\deg(x)=1,\qquad \deg(y)=-1.

The invariant subring is the degree-zero part:

k[x,y]^G = k[xy].

So the quotient is one-dimensional, with coordinate

z = xy.

That means a classifier that is invariant under this group must factor through . The decision rule becomes

f(x,y)=\mathrm{sign}(w\,xy+b).

That is an SVM on the quotient space.

Why this is the right model

The group action changes  but keeps  fixed:

(t x)(t^{-1} y)=xy.

This is not just a metaphor. It is literally the quotient construction of Exposé VIII.


---

Working example

Below is a complete example where:

the raw data are not linearly separable in ,

the invariant feature  makes them separable,

a linear SVM on  succeeds,

a linear SVM on raw  struggles.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)

# Generate data on orbits of G_m acting by (x,y) -> (t x, t^{-1} y)
# Class is determined by the invariant z = x*y.
n = 400
z_pos = rng.uniform(0.5, 2.0, size=n // 2)
z_neg = -rng.uniform(0.5, 2.0, size=n // 2)
z = np.concatenate([z_pos, z_neg])

# random orbit parameter t > 0
t = np.exp(rng.uniform(-1.0, 1.0, size=n))  # log-uniform

x = np.sqrt(np.abs(z)) * t
y = np.sign(z) * np.sqrt(np.abs(z)) / t

X = np.column_stack([x, y])
labels = (z > 0).astype(int)

# Shuffle
perm = rng.permutation(n)
X = X[perm]
labels = labels[perm]
z = z[perm]

# Baseline: linear SVM on raw coordinates
clf_raw = SVC(kernel="linear", C=1e6)
clf_raw.fit(X, labels)
pred_raw = clf_raw.predict(X)
acc_raw = accuracy_score(labels, pred_raw)

# Diagonalizable-group-aware feature: invariant z = x*y
Z = z.reshape(-1, 1)
clf_inv = SVC(kernel="linear", C=1e6)
clf_inv.fit(Z, labels)
pred_inv = clf_inv.predict(Z)
acc_inv = accuracy_score(labels, pred_inv)

print("Accuracy on raw (x,y):", acc_raw)
print("Accuracy on invariant z=xy:", acc_inv)
print("Raw separator w,b:", clf_raw.coef_, clf_raw.intercept_)
print("Invariant separator w,b:", clf_inv.coef_, clf_inv.intercept_)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(X[:, 0], X[:, 1], c=labels, s=18, alpha=0.8)
axes[0].set_title("Data in original coordinates (x,y)")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")

z_grid = np.linspace(Z.min() - 0.5, Z.max() + 0.5, 300).reshape(-1, 1)
decision = clf_inv.decision_function(z_grid)
axes[1].scatter(Z[:, 0], np.zeros_like(Z[:, 0]), c=labels, s=18, alpha=0.8)
axes[1].plot(z_grid[:, 0], decision, linewidth=2)
axes[1].axhline(0, linestyle="--")
axes[1].set_title("Invariant coordinate z = x y")
axes[1].set_xlabel("z")
axes[1].set_yticks([])

plt.tight_layout()
plt.show()

What this demonstrates

The group action creates orbits:

(x,y)\sim (tx,t^{-1}y).

The classifier should not depend on where you are on the orbit.
The invariant coordinate  is exactly the quotient coordinate.

So the SVM becomes a linear separator on the quotient:

\operatorname{Spec}(k[x,y]^G)=\operatorname{Spec}(k[xy]).

That is the diagonalizable-group version of “feature engineering.”


---

What standard SVMs simplify away

From this perspective, a standard SVM usually simplifies in three ways:

1. It forgets the grading

SGA3 says a representation decomposes into weight spaces.
A standard kernel method usually does not remember which monomials or features came from which weights.

2. It keeps only pairwise geometry

The classifier only sees the Gram matrix .
That is much weaker than keeping the full graded algebra and its quotient structure.

3. It assumes the right quotient was chosen already

If the true decision rule is invariant under a diagonalizable symmetry, then quotienting is good.
If not, the compression loses signal.

That is the real limitation of the SVM side: it is powerful only after you have guessed the right invariant or feature algebra.


---

A more honest “SGA3-style SVM”

If you want a genuine algebraic formulation, it looks like this.

Choose a diagonalizable group  acting on a feature algebra

A=\bigoplus_{m\in M} A_m.

Then build the classifier from the invariant part:

A_0 = A^G.

Train a linear SVM on a finite list of generators of , or on a truncated collection of invariant monomials.

For the example above, , so the whole classifier reduces to one invariant variable .

That is the cleanest “working SVM using diagonalizable groups” you can build.


---

The punchline

Standard SVMs are not the full theory; they are already a quotient of a quotient.
SGA3 VIII gives the more structural viewpoint:

diagonalizable groups act by weights,

quotients are invariant degree-zero parts,

a good classifier should be built on the quotient only when the label is orbit-invariant.


So the paper does not merely “look like” SVMs. It explains when a classifier should descend to a quotient at all.

If you want, I can turn this into a compact mini-lecture with a second example using -symmetry and polynomial features, which is even closer to the grading language in Exposé VIII.
