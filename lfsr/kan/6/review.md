The current RangeTraverser is fundamentally doing:

GF(2^n)^\times \rightarrow \{1,\ldots,2^n-1\}

followed by rejection of values outside .

That gives a pseudorandom subsequence, but not a true permutation of the interval.

If you want a mathematically guaranteed traversal of

\{0,1,\ldots,M\}

exactly once, you need something stronger.


---

Method 1: Orbit + Ranking (simple and rigorous)

Build the full field orbit:

1,\alpha,\alpha^2,\ldots,\alpha^{2^n-2}

Then retain only elements whose integer representation is ≤ M.

Example:

field orbit:
13 7 29 3 18 1 24 10 4 31 ...

keep only <= 20:

13 7 3 18 1 10 4 ...

Store them in a vector.

Now:

class RangeTraverser {
    std::vector<uint64_t> perm;
    size_t pos = 0;

public:
    uint64_t next() {
        uint64_t x = perm[pos];
        pos = (pos + 1) % perm.size();
        return x;
    }
};

Properties:

every value appears once,

reproducible,

mathematically correct.


Drawback:

Memory O(M).

For M=10^9 impossible.


---

Method 2: Discrete Log Permutation

Much more interesting.

Observe:

GF(2^n)^\times

already provides a cyclic ordering.

Let

\phi(i)=\alpha^i

for

i=0,\ldots,2^n-2.

Then define

S=\{x\in GF(2^n)^\times : x\le M\}.

Instead of walking the field and rejecting values, compute the rank of every accepted element in the cyclic order.

Then:

rank 0 -> value 13
rank 1 -> value 7
rank 2 -> value 3
...

You obtain a true cyclic permutation of the interval.

This is essentially a finite-geometry version of a perfect shuffle.

Still requires storing M+1 entries.


---

Method 3: Affine permutation modulo N

This is what people actually use.

Let

N=M+1.

Choose

a

coprime to N.

Define

f(i)=(ai+b)\bmod N.

Then:

uint64_t next() {
    return (a * counter++ + b) % N;
}

Theorem:

If gcd(a,N)=1 then f is a permutation.

Thus every number appears exactly once.

Memory:

O(1)

Speed:

O(1)

This is often superior to LFSRs.


---

Method 4: Feistel permutation on interval

This is probably closest to your "constructive geometry" ambitions.

Choose

k = ceil(log2(N))

and build a Feistel network on k bits.

A Feistel network is automatically bijective:

x -> Feistel(x)

Then use cycle-walking:

do {
    x = Feistel(x);
} while (x >= N);

Result:

exact permutation of [0,N-1]

no storage

random-access

reversible


Used in format-preserving encryption.


---

Method 5: Primitive polynomial + quotient geometry

This is the most algebraic redesign.

Current code traverses

GF(2^n)^\times

which has size

2^n-1.

Suppose

N=M+1.

Find n such that

N \mid (2^n-1).

Then:

H=\langle \alpha^N\rangle

is a subgroup.

The quotient

GF(2^n)^\times/H

has exactly N elements.

Now traversal becomes

H,\alpha H,\alpha^2 H,\ldots,\alpha^{N-1}H.

This is beautiful because:

no rejection sampling,

exact size N,

every step is field multiplication,

genuinely derived from finite geometry.


In categorical language this is much closer to the "Kan-extension-generated traversal" idea you were discussing earlier, because the interval is obtained as a quotient of a cyclic field action rather than by filtering an orbit.


---

What I would actually implement

For a practical replacement of your code:

class ExactRangeTraverser {
    uint64_t N;
    uint64_t a;
    uint64_t b;
    uint64_t i;

public:
    ExactRangeTraverser(uint64_t max_val,
                        uint64_t seed)
        : N(max_val + 1),
          i(0)
    {
        std::mt19937_64 gen(seed);

        do {
            a = gen();
        } while(std::gcd(a,N)!=1);

        b = gen();
    }

    uint64_t next() {
        return (a * (i++) + b) % N;
    }
};

This gives a provable one-pass traversal of [0,max_val].

If your goal is specifically to preserve the finite-field/LFSR flavor, then the subgroup-quotient construction (Method 5) is the mathematically natural replacement: instead of rejecting unwanted field states, you construct a cyclic action whose orbit size is exactly . That turns the range traversal from a heuristic into an actual theorem.
