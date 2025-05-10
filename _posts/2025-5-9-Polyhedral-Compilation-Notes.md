---
layout: post
title: Polyhedral Compilation Notes
---

I have been wanting to learn about polyhedral compilation for the longest time, and finally had some time recently to wrap my head around it. I wanted to see its applicability into tensor programming DSLs, particular for my planned language texpr, the idea of which is not full automatic parallelization like tensor comprehensions, which does use polyhedral techniques, but make CUDA programming easier and more composable, ideally with a hierarchical IR which exposes incrementally more hardware details the lower down you go.


### What is polyhedral compilation / optimization

Polyhedral techniques represent loop-nests in terms of convex polyhedrons, which form a kind of minimal description of a loop-nest, there are three primary objects of interest: 
1. The iteration domain $$\mathcal D$$
2. The schedule $$\theta$$
3. The dependence relation $$\mathcal P$$

```
for i = 0..N
    for j = 0..M
        C[i] += A[i, j] * b[j]    # S(i, j)
```

Each statement is abstracted as a named integer tuple, and each specific instantiation of an iteration is represented by a d-dimension integer vector/tuple.
The set of all iterations is the set of all possible values that the integer vector can take, and is represented by a polyhedron called the iteration domain, $$\mathcal D \subseteq \mathbb Z^n$$. In the example above $$\mathcal D = \{ (i, j) : 0 \leq i < N, 0 \leq j < M \}$$. If there are symbolic constants in the program, we also include them in the iteration domain, eg. $$\mathcal D = \{ (i, j, n, m) : 0 \leq i < N, 0 \leq j < M, 1 \geq n, 1 \geq m \}$$. More generally, we can write the iteration domain as 

$$\mathcal D = \{ (x, n) : A(x, n)^T + b \geq 0 \}$$

Where $$x$$ are the loop indices and $$n$$ the vector of symbolic constants.

The schedule is a function $$\theta: \mathcal D \to \mathbb Z^m$$, which represents the ordering of the execution. In the example above, $$\theta(0, 0, n, m) << \theta(0, 1, n, m) << ... << \theta(0, m-1, n, m) << \theta(1, 0, n, m) ...$$. Where $$<<$$ is interpreted as lexicographic ordering (standard ordering on the integers if $$m=1$$). The schedule implied by the example is $$\theta(i, j, n, m) = (j, i)$$. 


The dependence relation gives the ordering restrictions on the iteration domain that arise from data-dependencies. Let $$x, y \in \mathcal D$$, if $$x$$ must come before $$y$$, then $$(x, y) \in \mathcal P$$. We can write the dependence relation as

$$\mathcal P = \{ (x, y, n) : C(x, y, n)^T + d \geq 0 \}$$

where in this instance I included the symbolic constants as well, for generality.


Polyhedral compilation has 3 phases as an optimization pipeline
1. lifting the AST into the polyhedral abstractions (getting iteration domain $$\mathcal D$$, original schedule $$\theta_0$$ and dependence relation $$\mathcal P$$)
2. finding a better schedule $$\theta$$ obeying constraints (data-dependence, etc.)
3. lower the new found schedule back into an AST (codegen)


### 1. Lifting the AST

In polyhedral optimization, the only loops that can be operated on are loops where the lower, upper and increments are affine expressions of constants, upper loop indices, symbolic constants, and all array accesses are also affine expressions of (symbolic) constants and loop indices. 

```
affineExpr e = e + e | e - e | c * e | floor(e / c)
```
Note that `c` is a constant and floor division by a constant is allowed.

This is because many operations we want to perform in the optimization stage become undecidable if general polynomials are allowed. Whereas operations on polyhedra and affine expressions belong to a formalism known as presburger arithmetic that is decidable (with exponential algorithms lol).

```
for i = 0..N
    C[i] = 0    # U(i)
    for j = 0..M
        C[i] += A[i, j] * b[j]    # S(i, j)
```

There are several representations for loop-nests with multiple statements. One such representation associates a dimension to each statement ordered syntaxically, and 0s for empty iteration indices. In the example above $$\mathcal D_U = \{ (0, i, 0) : 0 \leq i < N \}$$, $$\mathcal D_S = \{ (1, i, j) : 0 \leq i < N, 0 \leq j < M \}$$. $$\mathcal P = \{ ((0, i, 0), (1, i, 0)) : 0 \leq i < N \} \cup \{ ((1, i, j), (1, i, j+1)) : 0 \leq i < N, 0 \leq j < M \}$$. (symbolic constants not included)

In practice, several other encodings are possible, such as schedule trees, dependence graphs.

I did not look into this stage of the pipeline that much since it seems to be the most straightforward of all the stages. 


### 2. Finding a better schedule
For this stage all we need is the iteration domain and dependence relation. The goal is to find a better schedule that maximizes some objective while satisfying the data-dependencies or other legality constraints. 

To optimize the over schedules, we must first find a parameterization of it. From what I can find the most general schedule parameterization is in the form $$\theta(x) = Tx + t$$, since we have to obey the affine restriction. (In Bastoul's 2004 [paper](https://icps.u-strasbg.fr/~bastoul/research/papers/Bas04-PACT.pdf) there's a more general parameterization $$\theta(x) = (Tx + t) / d$$, but for simplicity let's focus on the previous one.)

If we include symbolic constants, then $$\theta(x, n) = T(x, n)^T + t$$.

Let's first consider the 1 dimensional case $$\theta: \mathcal D \to \mathbb Z$$ and later generalize to multidimensional schedules with lexicographic ordering.

To be a valid schedule, it has to obey 2 constaints:
1. $$\forall (x, n) \in \mathcal D, \theta(x, n) \geq 0$$
- we disallow negative time
2. $$\forall (x, y, n) \in \mathcal P, \theta(x, n) < \theta(y, n)$$
- schedules must obey data dependencies

If we have a valid schedule satisfying the dependences, we can choose to optimize the schedule, for example, maximizing its parallelism by minimizing the maximum time it takes: $$\min_{T, t} \theta(x, n) \text{ s.t. 1 and 2 hold}$$, the schedule can assign the two iteration vectors to the same time stamp, which implies that the two iterations can be scheduled in parallel.

To make the optimization over the constaints more tractable, we can apply farkas lemma:

Let $$P = \{ x \in \mathbb R^n | Ax + b \geq 0 \}$$ be a non-empty convex polyhedron, and $$\phi: P \to \mathbb R$$ an affine function. Then 

$$\forall x \in P, \phi(x) \geq 0 \iff \exists \lambda_0 \in \mathbb R, \lambda \in \mathbb R^m, \lambda_0, \lambda \geq 0, \phi(x) = \lambda^T(Ax + b) + \lambda_0$$

Then applying farkas' lemma
1. becomes: $$\theta(x, n) = \mu_0 + \mu^T(A(x, n)^T + b)$$
2. becomes: $$\theta(y, n) - \theta(x, n) - 1 = \lambda_0 + \lambda^T(C(x, y, n)^T + d)$$

Collecting everything, we have the following constraints:
- $$\mu_0, \mu, \lambda_0, \lambda \geq 0$$
- $$\mu_0 + \mu^T(A(y, n)^T + b) - \mu_0 + \mu^T(A(x, n)^T + b) - 1 = \lambda_0 + \lambda^T(C(x, y, n)^T + d)$$

Since everything is affine, we can summarize everything in another polyhedron $$G = \{ (\mu_0, \mu, \lambda_0, \lambda) : H(\mu_0, \mu, \lambda_0, \lambda) \geq h \}$$, which is the set of valid farkas multipliers satisfying 1 and 2.


Then we can formulate a linear program, for example, to solve to min latency:

$$\min \mu_0 + \mu^T(A(x, n)^T + b)$$ such that
$$G(\mu_0, \mu, \lambda_0, \lambda) \geq d$$.

I have heard of other more sophisticated cost functions capturing things like parallelism, spaital locality, etc.

#### Multi-dimensional schedules
As proven in Feautrier's 1997 [paper](https://www.researchgate.net/publication/2810999_Some_efficient_solutions_to_the_affine_scheduling_problem_Part_II_Multidimensional_time): (Some efficient solutions to the affine scheduling problem Part II Multidimensional time) every static control program admits a multi-dimensional schedule. This is due to single dimensional affine schedules being not expressive enough to capture the types of program transformations we want, such as loop tiling, etc.

For example, if we desire this permutation of the iteration space: $$\theta(i, j) = (j\%2) + (j//2)*8 + (i\%2)*2 + (i//2)*4$$
```
 -> i
| 0 2  4  6
v 1 3  5  7
j 8 10 12 14
  9 11 13 15
```
we can write it as a multidimensional schedule: $$\theta(i, j) = [j, i, i/2, j/2]$$ which gets what we want due to lexicographic ordering.

The multi-dimensional schedules $$\theta: \mathbb Z^n \to \mathbb Z^m$$ the constraints are
1. $$\forall (x, n) \in \mathcal D, \theta(x, n) \geq 0$$
2. $$\forall (x, y, n) \in \mathcal P, \theta(x, n) \prec \theta(y, n)$$
   - Write $$\Delta(x, y, n) = \theta(y, n) - \theta(x, n)$$, the above constaint is the same as $$\Delta(x, y, n) \succ 0$$.
where $$\geq 0$$ denotes component-wise ordering and $$\prec$$ denotes lexicographic ordering.

Let $$\theta(x, n) = [\theta_0(x, n), \dots, \theta_n(x, n)]$$ and similar denote $$\Delta_i$$ for each component of $$\Delta$$.

Let 
$$H = \{\theta : \forall (x, n) \in \mathcal D, \theta(x, n) \geq 0 \land \forall (x, y, n) \in \mathcal P, \Delta(x, y, n) \succ 0\}$$ 
be the set of valid schedules. 

Let 

$$U_i = \{ \theta : \forall (x, n) \in \mathcal D, \forall j \leq i, \theta_j(x, n) \geq 0 \\ \land \forall (x, y, n) \in \mathcal P, \forall j \in \{1, \dots, i-1\}, \Delta_j(x, y, n) \geq 0 \land \Delta_i(x, y, n) \}$$

And note that $$H = \Cup_i U_i$$, hence every valid schedule is in one of $$U_i$$, where $$d$$--dimensional schedules are in $$U_d$$. Also note that every $$U_i$$ can be farkas converted since it follows the standard form. Therefore, every $$U_i$$ can be written as a polyhedron where each schedule is captured by its farkas multipliers.


### 3. Scanning the polyhedra to codegen the loop-nest

One of the procedures for the codegen algorithm in Bastoul's 2004 paper is projection of the polyhedron onto a dimension, one can use Fourier Motzkin elimination to achieve this.

Given a polyhedron $$P = \{ [x, y] : A[x, y]^T \leq b\}$$, the polyhedron with the variable(s) x eliminated is $$\{ y : \exists x (A[x, y]^T \leq b) \}$$.
We can assume that $$x \in \mathbb R$ and $$y \in \mathbb R^n$$ and eliminate one variable first, the rest go by induction.

Write $$P = \{ [x, y] : xa + A'y \leq b \}$$. Let $$P_+ = \{ i \in \{1..n\} : a_i > 0 \}$$, $$P_-=\{ i : a_i < 0 \}$$, $$P_= \{ i : a_i = 0 \}$$.
There are 3 disjoint groups of constraints that union to $$P$$.
1. $$xa_i + A'^T_iy \leq b_i, i \in P_+$$
2. $$xa_i + A'^T_iy \leq b_i, i \in P_-$$
3. $$xa_i + A'^T_iy \leq b_i, i \in P_=$$

Divide 1, 2 by $$|a_i|$$, and reorder to get
1. $$x \leq (b_i - A'^T_iy) / a_i, i \in P_+$$
2. $$x \geq (b_i - A'^T_iy) / a_i, i \in P_-$$

Then the polyhedron with $$x$$ eliminated is 

$$P' = \{ y : (b_i - A'^T_iy) / a_i \leq (b_j - A'^T_jy) / a_j, \forall i \in P_-, \forall j \in P_+ \} \cap \{ y : A'^T_iy \leq b_i, i \in P_= \}$$
