---
layout: post
title: An interesting idea about PL memory semantics
---

## The Status-Quo

*I know Rust, Haskell, Python, Julia, C/C++, Java, and have read about Flix, OCaml, so what I know will be constrained by these languages, similarly I am not familar with any formal memory model that unifies these considerations, so this entire discussion will be informal.*
It seems to me that there are at least two orthogonal directions when it comes to memory semantics that a programming language should enforce. One direction is concerned with efficiency: languages should prevent classes of invalid programs to ensuring safety under a certain efficiency constraint. The other is concerned with program organization: languages should enforce modularity and other desirable properties that people want programs to have. 

The status quo right now comes down to the divide between GC and non-GCed languages, or more generally, the amount of computation that the runtime has to spend to properly manage finite memory resources. This concern with efficiency bleeds over to user-facing memory semantics as can be seen in the differences in older languages.
C++ for example features limited language constructs to ensure memory safety for the purposes of decreasing the amount of computation it takes to track allocations, while Java and Python for the sake of user convenience muddies the waters by mixing reference and value semantics. I see newer languages like Rust trying to tackle both directions at once, using the concept of object lifetimes to maintain the efficiency of C++ while preventing invalid programs. While Flix goes the other direction and uses algebraic effects in a GCed environment to clearly delineate mutation, thereby enforcing correctness. 

In a language concerned with efficiency, such as Rust, there must exist 2 or 3 types of memory semantics that data could have. 1. Owned data, such as integers, strings, represent the notion of stack allocation and clear lifetime regions thereby allowing static analysis to deduce when the destructor should be called. 2. References models the concept of pointers to data and have reference semantics when mutation is allowed, Rust then further separates references into immutable and mutable for the sake of correctness. 

Languages concerned with convenience frees the user of considering the location of the destructor and thus the lifetimes of objects. Consequently, there must be a heavier runtime to track all allocations, necessarily on the heap, incurring linearly time complexity to manage memory. Languages that are not so much concerned with correctness, Python for instance, grants arbitrary reference semantics that is a product of the implementation, whereas Haskell and Flix, on top of their GCs also restrict their reference semantics on the type level for the sake of correctness. While more pragmatic languages like OCaml features some backdoors that allows for arbitrary mutation references like Python or Java.

Ideally, I want a language that enforces correctness, while also allowing for a certain degree of convenience and efficiency. I think Rust is almost there, it excels in terms of correctness and efficiency but is lacking in convenience.

## Lifetimes as witness to mutation

Besides the syntaxical burden, lifetimes both restricts the ability to express mutually coreferring data structures like a doubly linked list, and also infects abstractions. 

- arena pattern in rust


