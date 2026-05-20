Here is what was corrected, and why each correction matters.

**The central error in all previous versions** was stating `err = f(x,y)`. This is wrong. The correct theorem, proven by induction over all 8 octants and verified computationally before a line of React was written, is:

```
err + sx·sy·f(x,y)  =  adx − ady  =  constant
```

The sign factor `sx·sy` is essential — it flips for lines going in opposite diagonal directions — and the additive constant `adx − ady` is not zero in general. The visualization now has a dedicated column that shows this sum at every pixel and marks it `✓/✗`.

**The Farey accumulator invariant** also has a sign that was wrong before. There are two distinct forms:

```
x-major:  acc  =  ⌊(adx−1)/2⌋  +  sx·sy·f(x,y)
y-major:  acc  =  ⌊(ady−1)/2⌋  −  sx·sy·f(x,y)
```

The sign flip in the y-major case comes from the fact that the major-axis step (y) decreases `F = sx·sy·f`, so `acc` and `F` move in opposite directions. This was verified across all 8 octants before being written into the code.

**The nearest-neighbour proof (T2)** is now stated precisely. Bresenham selects whichever of the two candidate pixels has smaller `|f|`. The selection condition `f > (adx − 2·ady)/2` is algebraically identical to the Bresenham step condition `e2 < adx`. These are not approximately the same — they are exactly the same inequality rewritten.

**The discriminant bound (T4)** follows directly from the Farey invariant: since `acc ∈ [0, major−1]` throughout, `|f| ≤ ⌊(major−1)/2⌋ + 1`. This confirms the algorithm stays within half a pixel of the ideal line in the perpendicular direction, which is the definition of nearest-integer rasterization.

**On the Grothendieck framing**: the theory tab now says explicitly what is genuine (integer arithmetic on ℤ, Ehrhart theory, Sturmian words, continued fractions) and what is theatrical (claiming Bresenham "uses schemes" when it predates scheme theory by 15 years). The real algebraic geometry is in Pick's theorem via toric varieties, where the Ehrhart polynomial of a lattice polygon is genuinely the Hilbert polynomial of the associated projective toric variety, and Pick's theorem is a specialisation of Hirzebruch-Riemann-Roch. That connection is stated precisely and proved to hold computationally via the `interior.length === pick.I` verification shown in the polygon tab.
