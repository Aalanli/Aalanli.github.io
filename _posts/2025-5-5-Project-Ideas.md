---
layout: post
title: Project ideas on my mind for a while
---

# TExpr

**TL;DR**: a mixed-level tensor expression language/IR and optimizing compiler that generates fast gpu kernels.


Currently there's no tensor DSL (that I'm aware of) that offers the combination of these features:
1. Relatively advanced optimizations (fusion, tiling, shared+reg promotion, etc.)
2. Lower level escape hatch, and good interoperability with the optimization engine
3. completely self-contained code gen with no runtime jit

During my internship at Waabi I noticed that a problem facing them, and I'm sure many other applied machine learning companies is this: We need a reliable tool that generates good code, the input language should be high-level enough for ML researchers to move fast (or for system engineers to catchup), the output should integrate well with a complex codebase, and the kernels should be statically compilable or offer some max-latency guarantees. 

We need a tool that's not black-box, not a all-in-one solution whose final performance is unpredictable and hard to diagnose. Its intended audience are people with some GPU programming background, and should offer lower-level access to those that need it. We need a tool that can integrate well with PyTorch, TensorRT, CUDA, Rust Python. We need a tool that's more scalable than individually writing CUDA kernels for every operation that's becoming a hotspot today, only to find the codebase littered with old and unused CUDA code from last week. We need a tool that's flexible enough to support dynamicism (dynamic shape) and various dynamic, sparse and otherwise weird architectures that researches come up withh. Finally, we need a tool that has good static guarantees during runtime; JIT compilation is unacceptable for latency constrained environments like the perception model of an autonomous vehicle. 

CUDA is a fine abstraction for the level of hardware access and control it offers, but for this level of abstraction, it lacks modularity, composability. I find many patterns repeated, manually specializing code to handle various cases (vectorized loading, shared/register promotion, pipeling), and of course many trivial fusion patterns (elementwise epilogue). 
At first I've evaluated XLA for our use case, but static shape is a deal-breaker, using it in production alongside pytorch is non-trivial, especially if we want to share memory spaces with pytorch, and performance is not as good as TensorRT for common ML architectures.
Trition is a good DSL for tried and true ML operators: matrix multiplication. I've experimented with convolution in the past but it was not able to find good schedules. Implementing more exotic operations is not possible last time I checked, as there's no abstraction for shared memory.
Futhark is the closest to meeting all of the requirements I listed. It generates stand-alone portable c code that uses the cuda device API to load and compile once at the beginning of execution. Its programming model is dynamic and flexible enough to write radix sort, segmented scan and histograms, but it does not use warp-level primitives which means its performance suffers. Integrating with pytorch also requires code patching as they isolate their runtime to a separate cuda context, which requires more expensive synchronization with pytorch's context.


# Point Transformer V3

**TL;DR**: Testing the point transformer V3 architecture on various other data modalities, not just lidar point cloud. 


# ESat-Genetic

**TL;DR**: Combining equality saturation with genetic and reinforcement learning algorithms. 

