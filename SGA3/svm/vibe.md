Diagonalizable Groups: a Standalone Introduction

This note rewrites the main ideas of Grothendieck’s Exposé VIII: Groupes diagonalisables into a more modern, self-contained introduction. The original text is extremely concise and assumes the reader already knows a lot of group-scheme language. Here the goal is different: explain the objects, show why they are natural, and give computational examples that make the algebra concrete.

A caution up front: the subject is algebraic geometry and commutative group theory, so there are not really “numerical methods” in the usual PDE or optimization sense. What modern computation can do is help us experiment with the algebra behind diagonalizable groups: finite abelian groups, character lattices, graded rings, and quotient constructions. The code below is therefore illustrative rather than approximate.

1. The guiding idea

A diagonalizable group is the group-scheme version of the fact that a commutative group can often be understood through its characters.

For an ordinary abelian group , its character group is

\operatorname{Hom}(M, \mathbb{G}_m),

where  is the multiplicative group. Grothendieck packages this as a functor

D(M) = \operatorname{Hom}(M, \mathbb{G}_m).

When  is commutative, this functor is represented by an affine group scheme. That representing object is the diagonalizable group attached to .

The slogan is:

diagonalizable groups are exactly those commutative group schemes that are built from characters.

This viewpoint is the algebraic-geometry analogue of decomposing a representation into eigenspaces. Over a field, a diagonalizable group acts by splitting objects into weight spaces.

2. Definition

Let  be a base scheme and let  be an abelian group.

The diagonalizable group scheme associated with  is written

D_S(M) = \operatorname{Hom}_{S\text{-grp}}(M_S, \mathbb{G}_{m,S}),

where  is the constant group scheme attached to .

A group scheme  over  is diagonalizable if it is isomorphic to some .

Typical examples:

 itself, corresponding to .

, the group of -th roots of unity, corresponding to .

Products such as , corresponding to direct sums of groups.


A diagonalizable group is locally diagonalizable if it becomes diagonalizable after restricting to an open cover of the base.

3. Why the duality works

The original paper’s first major point is a duality principle.

A morphism from a group  into a diagonalizable group is the same thing as a compatible family of characters of . Abstractly, Grothendieck encodes this using a contravariant functor  and shows that diagonalizable objects are reflexive: applying the dual functor twice gives you back the original group scheme.

In the simplest useful cases, this is the familiar idea that the character lattice determines the torus or finite diagonalizable group.

For example:








This is the algebraic-geometric version of “diagonalize by weights.”

4. The module-theoretic viewpoint

One of the cleanest ways to understand a diagonalizable group is through gradings.

Suppose . Then a representation of  on a quasi-coherent module is the same as an -grading. In the affine case, a -action on a ring  corresponds to a decomposition

A = \bigoplus_{m \in M} A_m

such that multiplication respects degrees:

A_m \cdot A_n \subseteq A_{m+n}.

This is the key bridge from group actions to explicit algebra.

Example

If , then a -action on an affine scheme is the same as a -grading.

If , then an action of  is the same as a cyclic grading modulo .

5. A computational toy model

Here is a tiny Python model for a finite diagonalizable group in the easiest case: the group of characters of .

from cmath import exp, pi

# n-th roots of unity as complex numbers

def roots_of_unity(n):
    return [exp(2j * pi * k / n) for k in range(n)]

# Characters of Z/nZ are determined by an integer k mod n:
# chi_k(m) = exp(2πi k m / n)

def character(n, k, m):
    return exp(2j * pi * (k * m % n) / n)

n = 5
print("5th roots of unity:")
for z in roots_of_unity(n):
    print(z)

print("\nCharacter table values for Z/5Z:")
for k in range(n):
    row = [character(n, k, m) for m in range(n)]
    print(k, row)

This is not algebraic geometry yet, but it is the same underlying structure: the group is controlled by its characters, and those characters form a dual object.

6. Graded rings in code

A diagonalizable action often becomes a grading problem. We can model a graded ring by storing each homogeneous piece separately.

from collections import defaultdict

class ZnGradedRing:
    def __init__(self, n):
        self.n = n
        self.pieces = defaultdict(list)

    def add_homogeneous(self, degree, element):
        self.pieces[degree % self.n].append(element)

    def degrees(self):
        return sorted(self.pieces.keys())

R = ZnGradedRing(4)
R.add_homogeneous(0, "1")
R.add_homogeneous(1, "x")
R.add_homogeneous(2, "x^2")
R.add_homogeneous(3, "x^3")

print(R.degrees())
print(R.pieces)

The point of this toy example is conceptual: the group action is encoded by how elements are sorted into degree classes.

7. Torsors under diagonalizable groups

Grothendieck’s Section 4 studies torsors under diagonalizable groups.

A torsor is a space that locally looks like the group, but may be globally twisted. For diagonalizable groups, torsors can be described very explicitly using graded invertible modules.

Roughly speaking, a -torsor over a base scheme  corresponds to a decomposition of a sheaf of algebras into homogeneous pieces with strong invertibility properties. In the affine case, this means the torsor is controlled by a graded algebra whose pieces behave like line bundles.

This is a very modern-feeling idea: the geometry of the torsor is reduced to algebraic data that can often be checked piece by piece.

Computational analogy

For finite groups, one can test whether a cocycle defines a twisted action by checking a small number of compatibility equations. In a computer algebra system, this usually means validating multiplication rules on basis elements of graded components.

# Symbolic check of a cyclic grading law

def degree_mul(a, b, n):
    return (a + b) % n

n = 6
for a in range(n):
    for b in range(n):
        assert degree_mul(a, b, n) == (a + b) % n

print("Cyclic grading law verified.")

8. Quotients by free actions

Section 5 of the original note proves a quotient theorem: if a diagonalizable group acts freely on an affine scheme, then the quotient exists and the original scheme is a torsor over it.

This is one of the most useful structural results in the paper.

In ordinary language:

free action,

affine scheme,

diagonalizable group,

good quotient exists,

quotient map is a torsor.


The result is powerful because diagonalizable groups are exactly the kind of groups for which invariant theory behaves cleanly. The grading again does the work: the quotient is recovered from the degree-zero part, while the other homogeneous parts describe the torsor structure.

A useful computational model is to think of quotienting by a finite diagonalizable group as projecting onto invariant monomials.

# Very small monomial-invariant toy model
# Monomial x^a y^b has weight a - b mod n under a cyclic action.

def weight(a, b, n):
    return (a - b) % n

n = 3
invariants = []
for a in range(5):
    for b in range(5):
        if weight(a, b, n) == 0:
            invariants.append((a, b))

print("Invariant monomials x^a y^b with a-b ≡ 0 mod 3:")
print(invariants)

9. What the appendix contributes

The appendix on monomorphisms is auxiliary but useful. It gives criteria for when a monomorphism of group schemes is an immersion or a closed immersion. For diagonalizable groups, these statements become especially transparent because everything can be translated into lattice maps.

In the diagonalizable world, a morphism is often determined by a map of character groups in the opposite direction. This turns geometric questions into elementary algebra:

injectivity of a map of group schemes becomes a surjectivity question on character lattices,

kernels and cokernels become explicit,

quotient behavior becomes computable.


10. A modern mental model

Here is the most practical way to remember the whole story.

A diagonalizable group is what you get when a group can be read from its characters. Its actions are gradings. Its torsors are twisted gradings. Its quotients are invariant pieces. Most of the geometry is invisible until you choose a decomposition into weights.

That makes diagonalizable groups unusually friendly to computation:

in finite settings, you can enumerate characters directly;

in affine settings, you can work with graded rings;

in quotient problems, invariants are often the degree-zero part;

in representation problems, weight decomposition is the organizing principle.


11. Suggested code directions

If you want to turn this introduction into a computational companion, the most natural next steps are:

implement finite abelian groups and their character tables,

use a CAS to manipulate graded rings,

model torus actions on polynomial rings by weights,

compute invariant monomials and quotient generators,

experiment with torsors using cocycles in finite abelian groups.


For a serious working environment, SageMath is the most natural choice, but even plain Python is enough for many toy examples.

12. From diagonalizable groups to SVM intuition

At first glance, diagonalizable groups and Support Vector Machines (SVMs) live in completely different worlds: algebraic geometry vs. statistical learning. But there is a useful bridge:

feature maps and kernels behave like weight decompositions.

An SVM works by embedding data into a (possibly high-dimensional) feature space and then finding a linear separator. When a kernel is used, this feature space is implicit, but mathematically it is still there.

A diagonalizable group action decomposes functions into weight spaces indexed by a lattice (or finite group). This is structurally similar to decomposing a feature space into basis functions indexed by "frequencies" or characters.

Key analogy

Algebraic geometry	Machine learning

character (chi_m)	feature map component
grading by (M)	feature decomposition
invariant part (degree 0)	kernel invariance / symmetry
weight decomposition	basis expansion


In particular, for finite abelian groups, characters are exactly discrete Fourier modes. Many kernel methods implicitly rely on such decompositions.

13. Toy example: characters as features

We can use characters of a finite cyclic group as explicit features for an SVM.

import numpy as np
from cmath import exp, pi
from sklearn import svm

# Characters of Z/nZ as feature map

def character_features(x, n):
    return np.array([exp(2j * pi * k * x / n) for k in range(n)])

# Real-valued embedding (split real/imag parts)
def real_features(x, n):
    feats = character_features(x, n)
    return np.concatenate([feats.real, feats.imag])

# Dataset: classify integers mod n into two classes
n = 7
X = np.array([real_features(x, n) for x in range(n)])
y = np.array([1 if x < n//2 else -1 for x in range(n)])

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print("Predictions:", clf.predict(X))

What is happening?

We map each input (x) into a vector of characters (Fourier features).

This is exactly a decomposition into weight spaces.

The SVM then finds a linear separator in this feature space.


This is conceptually identical to working with a diagonalizable group: the structure is revealed after decomposition into characters.

14. Kernel viewpoint

Instead of explicitly computing features, SVMs often use kernels.

For cyclic groups, a natural kernel is:

[ K(x, y) = sum_{m in M} chi_m(x) overline{chi_m(y)}. ]

This is a Fourier kernel, and it corresponds to an inner product in the character space.

# Kernel induced by characters

def kernel(x, y, n):
    return sum(
        exp(2j * pi * k * (x - y) / n)
        for k in range(n)
    ).real

# Build kernel matrix
K = np.array([[kernel(x, y, n) for y in range(n)] for x in range(n)])

clf = svm.SVC(kernel='precomputed')
clf.fit(K, y)

print("Kernel SVM predictions:", clf.predict(K))

Interpretation

This kernel is invariant under the group structure (depends only on (x - y)).

That invariance is exactly what diagonalizable groups encode: symmetry under characters.

15. Connection to real SVM practice

This toy model connects to real techniques:

Fourier features / random Fourier features approximate kernels using characters of (mathbb{R}^n).

Group-invariant kernels explicitly enforce symmetries (translations, rotations).

Polynomial kernels correspond to graded decompositions (degree = weight).


From the diagonalizable group perspective:

choosing a kernel = choosing a decomposition into weight spaces,

training an SVM = finding a linear functional on those weights,

invariance = restricting to degree-zero (or symmetric) components.


16. Why this perspective is useful

The benefit of this algebraic viewpoint is conceptual clarity:

it explains why Fourier features work (they are characters),

it clarifies invariance as a grading constraint,

it connects kernel methods to representation theory,

it suggests new kernels by choosing different groups and characters.


In more advanced settings, one replaces finite groups by tori or even more general diagonalizable group schemes, leading toward harmonic analysis on algebraic groups.

17. Closing summary

Grothendieck’s Exposé VIII shows that diagonalizable groups are the group-scheme incarnation of character theory. The duality is clean, the representation theory is grading theory, and the quotient theory is invariant theory in disguise. Once those three viewpoints are in place, the subject becomes much easier to navigate.

The original text is beautifully compact. This rewrite expands the same core ideas into a form that can be read independently and used as a bridge to computation.

I tried to directly extend your document in the canvas with a full SVM section (feature maps + kernel formulation + code), but the update failed due to a pattern-matching issue in the editor.

Instead of retrying blindly, here’s the extension cleanly laid out—you can paste it right after Section 12 in your document:


---

12. From diagonalizable groups to SVM intuition

At first glance, diagonalizable groups and Support Vector Machines (SVMs) live in completely different worlds: algebraic geometry vs. statistical learning. The bridge is this:

feature maps and kernels behave like weight decompositions.

An SVM embeds data into a feature space and finds a linear separator. A diagonalizable group decomposes objects into character (weight) components. These are mathematically the same kind of operation: turning structure into coordinates where linear methods work.


---

13. Characters as feature maps (Fourier SVM)

For a finite abelian group (e.g. ), characters are discrete Fourier modes. These can be used directly as features.

import numpy as np
from cmath import exp, pi
from sklearn import svm

def character_features(x, n):
    return np.array([exp(2j * pi * k * x / n) for k in range(n)])

def real_features(x, n):
    feats = character_features(x, n)
    return np.concatenate([feats.real, feats.imag])

# simple dataset on Z/nZ
n = 7
X = np.array([real_features(x, n) for x in range(n)])
y = np.array([1 if x < n//2 else -1 for x in range(n)])

clf = svm.SVC(kernel='linear')
clf.fit(X, y)

print("Predictions:", clf.predict(X))

Interpretation

Each character = one “frequency feature”

The feature map = decomposition into weights

The SVM = linear functional on weights


This is exactly how diagonalizable groups act: by splitting into eigenspaces.


---

14. Kernel = inner product in character space

Instead of explicit features, we can define a kernel:

K(x, y) = \sum_{m \in M} \chi_m(x)\overline{\chi_m(y)}

def kernel(x, y, n):
    return sum(
        exp(2j * pi * k * (x - y) / n)
        for k in range(n)
    ).real

K = np.array([[kernel(x, y, n) for y in range(n)] for x in range(n)])

clf = svm.SVC(kernel='precomputed')
clf.fit(K, y)

print("Kernel SVM predictions:", clf.predict(K))

What this shows

Kernel depends only on  → group invariance

This is exactly the structure diagonalizable groups encode

You're doing harmonic analysis without naming it



---

15. Deep connection (this is the important part)

Diagonalizable groups → characters → Fourier decomposition
SVM → kernel → implicit feature decomposition

These are the same pipeline in disguise:

1. Choose a symmetry (group )


2. Use characters → basis functions


3. Build inner product → kernel


4. Solve linear separation → SVM




---

16. Why this actually matters

This isn’t just analogy—it explains real ML techniques:

Random Fourier Features = sampling characters of 

Translation-invariant kernels = characters of additive groups

Polynomial kernels = graded algebra (degree = weight)

Equivariant ML = representation theory of groups



---

17. If you want to push this further

The genuinely powerful direction (where your math background pays off):

Replace  with tori 

Use continuous characters → exponentials

Recover:

Fourier transforms

kernel methods

spectral learning



At that point, you’re essentially rebuilding kernel methods from SGA-level algebra.


---

If you want next step, I can:

turn this into a **full SVM solver from scratch using characters (no sklearn)**

or show how **random Fourier features approximate RBF kernels from this viewpoint**

or go deeper: connect diagonalizable groups → Pontryagin duality → kernels


That’s where it starts getting genuinely powerful.
