---
layout: post
title: TExpr Better explaination
---

I'm committed to seeing this project through. I really am. Its just that I am a minor expert on procrastination - also I have a job right now.

The purpose of this post is to reflect a bit and make more clear what I am trying to do; I think the last post is not the clearst and could do with more context about polyhedral compilation.

# Intro

The ultimate goal of TExpr is find a easier/better intermediate representation and novel optimizations in a compiler for machine learning workloads - and personally it's interesting to think about this stuff. Originally TExpr targeted a much higher level representation on the level of tensor graphs. It had something like this IR:

```
TExpr e = Compute N (λ i . e)
        | Reduce N (λ i . e)
        | Let v = e in e
        | Index v I
        | Elemwise Opcode [e]
```

In this pseudo-syntax, `N` represents a concrete natural, `I` an index expression (an expression in terms only of index induction variables bound by λ, constants, symbolic parameters and the arithmetic operations (+,-,*,floordiv)), `v` represents a named tensor variable and `e` represents an inductive instance of an TExpr. Here's the canonical matrix multiplication example expressed in a few lines of code:

```
inputs: A[M, K], B[K, N]
compute M |i| {
    compute N |j| {
        reduce K |k| {
            A[i, k] * B[k, j]
        }
    }
}
```
which generates the abstract syntax:
```
Compute M (λ i . 
    (Compute N (λ j . 
        (Reduce K (λ k .
            Elemwise "mul" [Index A (i, k), Index B (k, j)]
        ))
    ))
)
```

A `Compute` statement is semantically similar to a map combinator or a for loop, and `Reduce` is similar to an associative reduction. Since everything is pure, the sub-expressions under `Compute/Reduce` can be executed in parallel. I learned this representation while working on the Hidet compiler at CentML, which takes very liberally from tvm's tensor expressions. What's nice is that this representation naturally allows for expressing buffering and tiling - via the let expression. For example:

```
compute M/T |i| {
    let a: A[T] = compute T |it| { A[i*T + it] } in
        ...
}
```

where `a` can be taken to be residing on a higher level of memory - perhaps registers or shared memory.
Another benefit of this representation is that it very easily covers the vast majority of ML operations, hence with a compiler for this representation, one can rather easily create a new ML framework from scratch - by quickly prototyping ops and tuning only critical ones like matrix multiplication. Here's layernorm by the way:
```
input: A[B, N]
compute B |i| {
    let (var, sum) = reduce N |j| { (A[i, j] ^ 2, A[i, j]) } in
    compute N |j| {
        (A[i, j] - sum/N) / (sqrt(var/N) + eps)
    }
}
```
However, despite all these niceities, it is too high-level and I don't know how to lower it in a generic and powerful way. What's already available like tvm requires manual user schedules that still do not give users the full expressiveness of the GPU, and so I've opted for a lower-level representation - one with memory effects where the gap between it and CUDA is not too large.

## TExpr v2

In the second version of the IR, both because it seems easier to optimize and targets a niche that I don't think is addressed, memory effects, memory buffers and explicit organization of the parallel hierarchy is required to be written by the programmer.

The niche is that there's not many tools that aims to make the life of CUDA programmers easier, and tries to replace them (us) instead - to varying degrees of success. The core problem of CUDA in C++ I think is scheduling: how to deal with multiple asychronous things happening at once, and exploring the space of potential optimizations in a systematic way. For example, there's often questions like "should I use vectorized accesses here and tile by this much in registers, or should I use asynchronous copies to stage in shared memory?", "should I swizzle my accesses, or pad by one to prevent bank conflicts?", "how much should I tile by?", etc. 
These are not only problems relating to picking right constants but involves the exploration and tradeoffs in balancing parallel and sequential control structures, memory layouts, resource utilization, asychronous pipelining, cache hits, and a whole host of other concerns.

It certainly does not help that CUDA C++ has bad composition properties - procedures are fundamentally leaky as they must assume the state of the caller in terms of: what type of memory is being called with? (registers, shared, global), are the threads synchronized, what subset of threads are being called with, to name a few. The only benefits to this approach is that performance is more predictable (although to be honest most of the time I can only guess, hope and pray) as the language is very close in semantics to the underlying ir (ptx) and assembly (sass). People have tried to "solve" this problem of scheduling by picking and combining various program fragments through C++ templates in the form of the CUTLASS library, or code generators in python. 

Can we alleviate this problem somewhat on the language level? Using principled programming language abstractions instead of ad-hoc metaprogramming? I have no idea what "principled programming language abstractions" are, but I think the main problem in the compositionality of procedures is the assumption of control state - that is, precisely how many threads is this function being called with, are the threads coalesced, and what state should be left in the caller? To this end, I think we should abstract the notion of CUDA threads even more than they are, into virtual parallel control flow structures that can be arbitrarily nested and composed. To explain what I mean I should first write down the syntax:

```
texpr e = View $v I   -- $v is an identifier for a memory buffer, I is an index expression. Equivalent to $v[I]
        | Const NUM
        | Index $i    -- $i is an identifier for a loop induction variable
        | Param $p    -- $p is a symbolic parameter
        | Op OPCODE [e]

tstmt s = Par N $i [s]  -- virtual parallel control flow structure
        | Seq N $i [s]  -- sequential control flow structure (just a for loop)
        | Scope LVL [s] -- what is the "scope" of the underlying code, mapped across blocks/warps/threads?
        | Bind $v I E   -- I is an index expression and E is an expr, equivalent to $v[I] = E
        | Decl $v N     -- Declares a memory buffer of size N
```

To make it more clear, here's an example program:

```
fn saxpy(xs: f32[n], ys: f32[n], a: f32): f32[n] {
    res: f32[n];
    par n |i| {
        res[i] = a * xs[i] + ys[i];
    }
}
```

which very rougly translates to the abstract syntax tree
```
Decl $res n
Par n $i [
    Bind $res ($i) (Op "add" 
        (Op "mul" 
            (Param $a) 
            (View $xs $i))
        (View $ys $i)
    )
]
```

`par N |i| {...}` only means that the stuff in `{...}` can be executed in parallel, but does not guarantee that it is executed fully in parallel. Often times it is better to mix instruction-level parallelism in by taking advantage of the hardware pipelines. The above program can be refactored into:

```
fn saxpy(xs: f32[n], ys: f32[n], a: f32): f32[n] {
    res: f32[n];
    par (n/4) |i| {
        seq (4) |j| {
            res[i * 4 + j] = a * xs[i * 4 + j] + ys[i * 4 + j];
        }
    }
}
```

which can be unrolled into (something along the lines of):
```
fn saxpy(xs: f32[n], ys: f32[n], a: f32): f32[n] {
    res: f32[n];
    par (n/4) |i| {
        xs': f32[4] = vecload[4](xs, i * 4);
        ys': f32[4] = vecload[4](ys, i * 4);
        rs': f32[4] = a * xs' + ys';
        vecstore[4](res, i * 4)
    }
}
```
Of course assuming that `n` is divisible by 4, and pointer offsets `xs`, and `ys` has the right alightment of 16 bytes.
As far as I'm aware of, the widest load instruction prior to sm9.0 is 16 bytes for a single thread, if we want to increase the instruction-level parallelism further without sacrificing coalesced loads we can repeat the process.

```
fn saxpy(xs: f32[n], ys: f32[n], a: f32): f32[n] {
    res: f32[n];
    par (n/16) |i| {
        seq (8) |k| {
            seq (4) |j| {
                res[i * 16 + k * 4 + j] = a * xs[i * 16 + k * 4 + j] + ys[i * 16 + k * 4 + j];
            }
        }
    }
}
```

This is an example but I hope you'll appreciate that things gets complicated once shared memory, register layouts and arbitrarily nested `par` instances gets into play. In CUDA, there are fixed levels of parallelism, the grid, the warps and the threads in a warp, memory must be managed explicitly, the memory layout must be fully specified. In the interests of making composition possible, `par` instances must be arbitrarily nestable and memory fragments can be left unspecified, with value semantics. Furthermore, the language spec must guarantee that all threads are logically fully converged after leaving a `par` block or a procedure, and similarly when entering a function or `par` block. The compiler is responsible for inserting synchronization operations and is free to be more aggressive with analysis to remove redundant syncs and schedule memory. The tradeoff I'm making here is compilation time for bettering programming ergonomics and hopefully runtime performance - the expensive analyses I mentioned will probably require mixed integer programming and presburger set manipulations.

