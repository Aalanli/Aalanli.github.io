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

Each statement is abstracted as a named integer tuple, and each specific instantiation of an iteration is represented by a d-dimension integer vector/tuple (where d= the number of loop nests around the statement).
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

In polyhedral optimization, the only loops that can be operated on are loops where the lower, upper bounds are affine expressions of constants, upper loop indices, symbolic constants, and all array accesses are also affine expressions of (symbolic) constants and loop indices. 

```
affineExpr e = e + e | e - e | c * e | floor(e / c)
```
Note that `c` is a constant and floor division by a constant is allowed.

This is because many operations we want to perform in the optimization stage become undecidable if general polynomials are allowed. Whereas operations on polyhedra and affine expressions belong to a formalism known as presburger arithmetic that is decidable (with exponential algorithms lol) (TODO: find ref).

```
for i = 0..N
    C[i] = 0    # S1(i)
    for j = 0..M
        C[i] += A[i, j] * b[j]    # S2(i, j)
```

More generally, sequences of program statements that are amendable to polyhedral techniques, static control programs, are represented by Generalized Dataflow Graphs (GDGs). $$\mathcal G = (V, E, \mathbf D, \mathbf P)$$, where $$V=\{i : S_i\}$$ is the set of vertices and each vertex is associated with a statement, $$E$$ is the set of edges and $$(i, j) \in E$$ means a dependence between $$S_i$$ and $$S_j$$. $$D_i \in \mathbf D, i \in V$$ is the domain associated with each statement, and $$P_{ij} \in \mathbf P, (i, j) \in E$$ is the dependence polyhedron associated with each dependence.


### 2. Finding a better schedule
For this stage all we need is the iteration domain(s) and dependence relation(s). The goal is to find a better schedule that maximizes some objective while satisfying the data-dependencies or other legality constraints. 

#### Interpretation of the schedule
The schedule $$\theta: D \to \mathbb Z$$ in polyhedral compilation is an affine form (due to linearity restrictions from pesky decidability concerns). We can write $$\theta(x) = t^Tx + a$$. The schedule assigns a logical date to each iteration instance, and there are several interpretations for it.
Consider $$H = \{x \in D: \theta(x) = t\}$$, $$H$$ is a half-space or a hyperplane spanning the iteration domain. 
1. $$t$$ can be interpreted as time, in which case each hyperplane for a $$t$$ is the set of iterations that can be scheduled at the same time, and the loop carrying this dimension sequentially rolls forward in time.
2. $$t$$ can be interpreted as space, in which each hyperplane represents the set of iterations that is scheduled on the same processor, and the iterations for this dimension is the process id.

$$\theta$$ needs to respect data-dependencies, eg. for the example above, $$\theta(i, j) \leq \theta(i, j+1)$$, since we read $$C[i]$$ written by the previous iteration of $$j$$. 
Since $$\theta(x)$$ is an affine function of its input, it cannot represent quadratic or higher complexity loop nests, for example `S2` above. Hence we need the multi-dimensional generalization of time, where $$\theta: D \to \mathbb Z^m$$, $$\theta(x) = Tx + a$$. In this case, the ordering is given lexicographically.

In the multi-dimensional version, $$\theta$$ can be interpreted as a 'scattering' function [[3]](#3), which sends iteration instances from one coordinate loop-nest system into another. For example:

```
for i in 0..N
  for j in 0..M
    S(i, j)
```

transformed by $$\theta(i, j) = (i+j, j) = (t, q)$$

```
orginal order
i
|678
|345
|012
.------ j
transformed order
t
|  368
| 147
|025
.------ q
```

new loop nest:
```
for q in 0..N+M-1
  for t in max(0, q-N+1)..min(q+1, M)
    S(t-q, q)
```


To optimize the over schedules, we must first find a parameterization of it. From what I can find the most general schedule parameterization is in the form $$\theta(x) = Tx + t$$, since we have to obey the affine restriction. (In Bastoul's 2004 paper [[3]](#3) there's a more general parameterization $$\theta(x) = (Tx + t) / d$$, where $$d$$ are symbolic constants, introducing this adds more complexity in the constraints, so I skip for now)

If we include symbolic constants, then $$\theta(x, n) = T(x, n)^T + t$$.

Let's first consider the 1 dimensional case $$\theta: \mathcal D \to \mathbb Z, \theta(x, n) = t^T[x, n]+a$$ and later generalize to multidimensional schedules with lexicographic ordering.

To be a valid schedule, it has to obey 2 constaints:
1. $$\forall (x, n) \in \mathcal D, \theta(x, n) \geq 0$$
- we disallow negative time
1. $$\forall (x, y, n) \in \mathcal P, \theta(x, n) \leq \theta(y, n)$$
- schedules must obey data dependencies

If we have a valid schedule satisfying the dependences, we can choose to optimize the schedule, for example, maximizing its parallelism by minimizing the maximum time it takes: $$\min_{T, t} \theta(x, n) \text{ s.t. 1 and 2 hold}$$, the schedule can assign the two iteration vectors to the same time stamp, which implies that the two iterations can be scheduled in parallel.

#### Farkas' lemma
To make the optimization over the constaints more tractable, we can apply farkas lemma (as shown in [[7]](#7)):

Let $$P = \{ x \in \mathbb R^n : Ax + b \geq 0 \}$$ be a non-empty convex polyhedron, and $$\phi: P \to \mathbb R$$ an affine function. Then 

$$\forall x \in P, \phi(x) \geq 0 \iff \exists \lambda_0 \in \mathbb R, \lambda \in \mathbb R^m, \lambda_0, \lambda \geq 0, \phi(x) = \lambda^T(Ax + b) + \lambda_0$$

Then applying farkas' lemma
1. becomes: $$\theta(x, n) = \mu_0 + \mu^T(A(x, n)^T + b)$$
2. becomes: $$\theta(y, n) - \theta(x, n) = \lambda_0 + \lambda^T(C(x, y, n)^T + d)$$

Collecting everything, we have the following constraints:
- $$\mu_0, \mu, \lambda_0, \lambda \geq 0$$
- $$\mu_0 + \mu^T(A(y, n)^T + b) - \mu_0 + \mu^T(A(x, n)^T + b) = \lambda_0 + \lambda^T(C(x, y, n)^T + d)$$

Since everything is affine, we can summarize everything in another polyhedron $$G = \{ (\mu_0, \mu, \lambda_0, \lambda) : H(\mu_0, \mu, \lambda_0, \lambda) \geq h \}$$, which is the set of valid farkas multipliers satisfying 1 and 2.

Since $$\theta(x, n) = t^T[x, n] + a = \mu_0 + \mu^T(A(x, n)^T + b)$$, we can equate variables and solve for $$t, a$$ in terms of the farkas multipliers, so we can obtain a polyhedron of the set of possible valid parameters satisfying constraints 1 and 2.

One way of optimizing the schedule as suggested by [[1]](#1) is to minimize the asymptotic latency by considering $$L(n) = l^Tn + w$$, we consider the affine form $$L(n) - \theta(x, n)$$ and want to bound the maximum of $$\theta$$ by considering $$\forall (x, n) \in D, L(n) - \theta(x, n) \geq 0$$. Since this is a affine form, we find $$\alpha, \alpha_0 \geq 0 : L(n) - \theta(x, n) = \alpha^T(Ax+b) + \alpha_0$$. Using the methods above, we can obtain a polyhedron for the parameters $$l, w, t, a$$, and minimize $$l$$ in this polyhedron.

#### Generalization to GDGs

The constraints generalized to GDGs:
1. $$\forall i \in V, \forall (x, n) \in D_i, \theta(x, n) \geq 0$$
2. $$\forall (i, j) in E, \forall (x, y, n) \in P_{ij}, \theta_i(x, n) \leq \theta_j(y, n)$$

We have a schedule $$\theta_i$$ associated with each statement $$S_i$$, $$i \in V$$, this means we must optimize all statement schedules jointly, but adding constraints should be fairly similar to what we did before.

#### Multi-dimensional schedules
As proven in Feautrier's 1997 paper [[2]](#2): every static control program admits a multi-dimensional schedule. This is due to single dimensional affine schedules being not expressive enough to capture the types of program transformations we want, such as loop tiling, etc.

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
2. $$\forall (x, y, n) \in \mathcal P, \theta(x, n) \preceq \theta(y, n)$$
   - Write $$\Delta(x, y, n) = \theta(y, n) - \theta(x, n)$$, the above constaint is the same as $$\Delta(x, y, n) \succeq 0$$.
where $$\geq 0$$ denotes component-wise ordering and $$\prec, \preceq$$ denotes lexicographic ordering.

To convert to constraints for the ILP there are some ways of doing it:
1. proceed sequentially outer-most in

What I observed in [[4]](#4), [[5]](#5) and [[2]](#2) is sequentially deciding outer-most loop schedules first, maximizing parallelism, minimizing communication, etc, and move inwards. Eventually we want a set of schedules to fully obey the data dependences ($$\prec$$ instead of $$\preceq$$), outer parallel/spatial loops can be more relaxed and have $$\geq 0$$, while inner temporal loops have to fully respect data-dependencies eventually.

Suppose we have an outer schedule $$\theta_1 = t_1^Tx + a_1$$ decided already, satisfying $$\forall x \in D, \theta_1(x) \geq 0 \land \forall (x, y) \in P, \theta_1(y) - \theta_1(x) \geq 0$$ (where I removed the symbolic parameters for convenience). Then let $$\tilde P_1 = \{ (x, y) \in P : \theta_1(y) - \theta_1(x) = 0 \}$$, the set of instances that must eventually be scheduled on separate times, but for now scheduled to the same time on this loop. Notice $$\tilde P_1$$ is a polyhedron.

The next schedule has to obey $$\forall x \in D, \theta_2(x) \geq 0 \land \forall (x, y) \in \tilde P_1, \theta_2(y) - \theta_2(x) \geq 0$$. Apply farkas, optimize and obtain the schedule for $$\theta_2$$. Repeat until we get $$d$$ schedules, where for the last schedule, the constraints are: $$\forall x \in D, \theta_{d-1}(x) \geq 0 \land \forall (x, y) \in \tilde P_{d-1}, \theta_2(y) - \theta_2(x) > 0$$ (see [[2]](#2) for more details). In practice as in [[4]](#4), we should also impose that each new schedule is linearly independent from the previously found schedules, since those schedules that are linear combinations of already found schedules are not helpful. This also ensures that the final constraint $$>0$$ does not fail as we need $$d$$ linearly independent affine schedules to cover a d-dimensional iteration domain.


#### Quasi-affine schedules

If we desire schedules of the form $$\theta(x) = floor((t^Tx + a) / c)$$, we can eliminate the division by moving constraints into the iteration domain. 

Define $$q \in \mathbb Z$$ such that $$\theta(x) = q$$, then $$qc \leq t^Tx + a \leq (q+1)c - 1$$. 

Then let $$\phi(q, x) = q$$ and let 

$$\tilde D = \{ (q, x) : Ax + b \geq 0, qc \leq t^Tx + a \leq (q+1)c - 1 \} = \{ (q, x) : \begin{bmatrix} -c & t^T \\ c & -t^T \\ 0 & A \end{bmatrix} \begin{bmatrix} q \\ x \end{bmatrix} + \begin{bmatrix} -a + c -1 \\ a \\ b \end{bmatrix} \geq 0 \}$$

We can apply farkas' lemma on $$\phi$$ and optimize over this lifted domain $$\tilde D$$, which is equivalent to the original.

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

### References

<a id="1">[1]</a> 
Feautrier, Paul. (1996). Some efficient solutions to the affine scheduling problem Part I One-dimensional Time. International Journal of Parallel Programming. 21. 10.1007/BF01407835. 

<a id="2">[2]</a> 
Feautrier, Paul. (1997). Some efficient solutions to the affine scheduling problem Part II Multidimensional time. International Journal of Parallel Programming. 21. 10.1007/BF01379404.

<a id="3">[3]</a> 
Cedric Bastoul. 2004. Code Generation in the Polyhedral Model Is Easier Than You Think. In Proceedings of the 13th International Conference on Parallel Architectures and Compilation Techniques (PACT '04). IEEE Computer Society, USA, 7–16.

<a id="4">[4]</a> 
Bondhugula, Uday et al. “Aﬃne Transformations for Communication Minimal Parallelization and Locality Optimization of Arbitrarily Nested Loop Sequences.” (2007).

<a id="5">[5]</a>
Bondhugula, Uday & Hartono, Albert & Ramanujam, J. & Sadayappan, Ponnuswamy. (2008). A practical automatic polyhedral parallelizer and locality optimizer. ACM SIGPLAN Notices. 43. 10.1145/1375581.1375595. 

<a id="6">[6]</a>
Uday Bondhugula, Aravind Acharya, and Albert Cohen. 2016. The Pluto+ Algorithm: A Practical Approach for Parallelization and Locality Optimization of Affine Loop Nests. ACM Trans. Program. Lang. Syst. 38, 3, Article 12 (May 2016), 32 pages. https://doi.org/10.1145/2896389

<a id="7">[7]</a>
Christophe Alias. Farkas Lemma made easy. 10th International Workshop on Polyhedral Compilation Techniques (IMPACT 2020), Jan 2020, Bologna, Italy. pp.1-6. ⟨hal-02422033⟩

#### Links

<a id="8">[8]</a>
Albert Cohen Pliss (2019) Tutorial: [https://pliss2019.github.io/albert_cohen_slides.pdf](https://pliss2019.github.io/albert_cohen_slides.pdf)

<a id="9">[9]</a>
CS 526 Advanced Compiler Construction: [https://misailo.web.engr.illinois.edu/courses/526-sp17/lec15.pdf](https://misailo.web.engr.illinois.edu/courses/526-sp17/lec15.pdf)

