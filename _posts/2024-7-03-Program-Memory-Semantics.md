---
layout: post
title: Lifetimes as witnesses to mutability
---

I think that there are two orthogonal concerns when it comes to the design of a type system for memory management, that of efficiency and correctness. 
Efficiency is concerned about modeling lower-level details on the type level as to prevent unsafe uses, while correctness is concerned about making mutability a contract as to ensure better software design, in terms of "metrics" such as composability or local reasoning. I think the two concerns are different, although they are addressed in the same manner by Rust's ownership model, for example. It attempts to both model the concept of a stack (or more generally inline allocations), which limits the expressiveness of its memory model, which necessitates escape hatches like interior mutability that subverts the goal of local mutable reasoning. 

As described in [Stacked Borrows](https://dl.acm.org/doi/10.1145/3371109), a formalization of Rust's ownership semantics, lifetime annotations acts as a stack which enforces unique writers and views into an owned data structure. This rigidity prevents the classic example of pushing to a vector while later using a reference to one of its inner elements. Or returning a reference to a stack allocated variable. I think this is what I mean when I say "modeling the stack", for the language makes the use-case of inline allocations a priority, where inline allocations are cases such as `Vec<i32>` where data is stored on the underlying buffer, or any owned object whose lifetime is implicitly tied to the current stack-frame, because it is allocated inline on that stack.

For performance reasons this use-case is essential, but otherwise allowing one layer of indirection greatly increases expressivity. This is typical of garbage collected reference semantics languages where each object is allocated on the heap, hence their lifetimes are not constrained by their containers, be it a vector or the stack. Most of these languages, however, do not address the second concern of correctness, where in the cases of Python, Java, OCaml allows unrestricted mutability and aliasing. 

I still think the idea of "ownership" is important and confers an abstraction boundary, where owned data are conceptually independent mutability wise. If we remove the restriction of "modeling the stack", and add the concept of ownership onto a language with reference semantics, I think we naturally arrive at the idea of "lifetimes as witnesses to mutability". The core idea is that a unique lifetime is associated to an owned object and all objects that it "dominates". To mutate an object, we need to present that lifetime token to be "witness" to its mutation. 

**Version 1**
```
TOKEN = '\'' ALPHA_NUMERIC
lt = TOKEN | '<' lt
type = 'i32' | 'struct' '{' ltype ':' SYM (',' ltype ':' SYM)* '}' | Vec[ltype]
ltype = lt type

typeDecl = 'type' SYM ('[' lt (',' lt)* ']')? '=' ltype
expr = new[lt] expr | ...
stmt = 'let' SYM ':' ltype '=' expr | ...
```

In version 1 we assume that all types are implicitly heap allocated, `Vec[i32]` is represented as a `Vec[i32*]`. This way, all the dominated objects have lifetimes that form tree with an owned lifetime as its root, so we can hold references to the leaves and still be able to mutate them. 
In this simple language, new objected are created with a lifetime variable from the `new` keyword. The lifetime variable can be unique, denoting a root lifetime, or a dominated lifetime.

```
let x: 'a i32 = new['a i32] 1; \\ unique
let y: < 'a = new[< 'a i32] 2; \\ dominated by a

x := 2 (with 'a); // need witness to write 
y := 3 (with 'a); // need root
```

This captures the use case of doubly linked lists
```
type DList = 'l struct {
    head: Option[LNode[< 'l]],
    tail: Option[LNode[< 'l]]
}

type LNode['l] = < 'l struct {
    data: <'l i32,
    prev: Option[LNode[< 'l]],
    next: Option[LNode[< 'l]]
}
```

I think this is semantically equivalent to the notion of mutable regions in Flix, and we can ensure local mutation by ensuring that functions that mutably alter their parameters needs to be annotated by the caller.

```
fn push_back(lst: mut['l] DList, b: i32) {
    if tail.is_none() {
        let tail: ... = new[< 'l] LNode { data: b, prev: None, next: None } (with 'l);
        tail.prev = tail (with 'l);
        tail.next = tail (with 'l);
        lst.head = tail (with 'l);
        lst.tail = tail (with 'l);
    } else {...}
}

fn get_back(lst: 'l DList): i32 {
    if tail.is_some() {
        return tail.data;
    } else {
        return 0;
    }
}

fn main() {
    let lst: 'l DList = ...;
    push_back(lst (with 'l), 1); // special annotation for mutated parameters
    let back = get_back(lst);
}
```


