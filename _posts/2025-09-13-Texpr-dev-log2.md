---
layout: post
title: Texpr-dev-log2
---

In my compiler I want to be able to optimize coalesced memory accesses, for example if the user wrote 

```
par N |i| {
    ... = A[f(i)]
}
```
where `f(i)` is some arbitrary quasi-affine function, then the compiler would assign a schedule on the logical worker elements `i`, such that memory accesses are coalesced. 
What this means is that the compiler would pick a function `σ(t, s) -> N`, a bijective function that transforms the loop above into:
```
for t in 0..T {
    for s in 0..S {
        ... = A[f(σ(t, s))]
    }
}
==> also known as this:
for s in 0..S {
    ... = A[f(σ(threadIdx.x, s))]
}
```
For the sake of simplicity we can assume that N factors into T and S so there exists such a bijection from $$\Z_{T} \times \Z_{S} \to \Z_N$$.

This is an unexpectedly difficult problem as I metaphorically bang my head against the wall, because we both want to find such a permutation and extract a small expression out of it. To more formally state the problem, this is the representation of `f` stated inductively:

```
quastExpr e = e + e | e - e | CONST * e | e // CONST | PID | CONST 
```
- `e` is another quasi-affine expression
- `PID` is a symbolic constant (that represents the block id)
- `CONST` is a constant
- `//` is floor division

we want to solve: 

$$\min_{\sigma} \sum_{j=0}^{T // 32} \sum_{t=0}^30 C(f(\sigma(t+1 + 32j, s), f(\sigma(t + 32j, s))))$$
Where the cost function $$C$$ on consecutive thread elements can be $$C(x, y) = |x - y|$$ or more precisely:
- C(x, y) = 0 if x + 1 = y
- C(x, y) = |x - y| otherwise.

There are other more complicated cost functions that models the cost dynamics of coalesced memory access, such as vectorization, minimum number of 32byte transactions, cache friendliness, etc. But even with such simple absolute cost functions we are searching over the infinite space of quasi-affine functions, which cannot be formulated as a linear integer program.

One approach is to consider a parameterized family of permutations with optimizable parameters. One such family that comes to mind is what I call the permute/reshape family. It is the family composed of permutations achieved via the pytorch operations `reshape([...])` and `permute([...])`. Imagine you initially have a pytorch tensor that represents a permutation, initially the identity permutation `arange(0, N)`, then you can subsequently apply `.reshape([...]).permute([...]).reshape([...]) ...` on that tensor to finally get a tensor of shape `[T, S]` where each value `tensor[t, s]` represents the place it is sent. One can parameterize a finite number of such reshape/permutes and a maximum number of dimensions to allow for each reshape/permute. However, this results in a program that is extremely non-linear, and it imposes one to set a bound for the parameterization. 

Another approach is to somehow deduce from the quasi-affine expression what would be good, a non-parameteric approach where the expression size can vary. To even start doing this I noticed that all quasi-affine expressions can be canonicalized into a flattened representation 

$$f(x) = \sum_{i=1}^n \kappa_i \lfloor \frac{a_i x + b_i}{d_i} \rfloor$$. 

To show that this is possible, all we need is to remove one nested level of floor, as the other cases of quasi-affine expression follows naturally.

Suppose we have $$F(x) = \lfloor \sum_{i=1}^n \kappa_i \lfloor \frac{a_i x + b_i}{d_i} \rfloor \rfloor$$, then let $$r_i = a_i x + b_i \pmod d_i$$. So we can rewrite to
$$F(x) = \lfloor \sum_{i=1}^n \kappa_i \frac{a_i x + b_i - r_i}{d_i} \rfloor$$. Let $$L$$ be the lcm of all the $$d_i$$s, then multiplying out by $$L$$ we get 
$$F(x) = \lfloor \sum_{i=1}^n \frac{a_i' x + b_i' - \gamma_i r_i}{L} \rfloor$$. Notice $$F(x + L) = F(x) + P$$ where $$P = \sum_i a_i'$$. Hence $$G(x) = F(x) - P \lfloor \frac{x}{L} \rfloor$$ is a $$L$$ periodic function.

By a fact from discrete calculi we can write $$G(x) = G(-1) + \sum_{i=0}^{L-1} (G(i) - G(i-1)) \lfloor \frac{x - i}{L} \rfloor$$. So $$F(x) = G(-1) + \sum_{i=0}^{L-1} (G(i) - G(i-1)) \lfloor \frac{x - i}{L} \rfloor + P \lfloor \frac{x}{L} \rfloor$$. 
