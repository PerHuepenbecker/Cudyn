# Cudyn: Dynamic Task Scheduling for Irregular Applications on GPUs

## Short Description

**Cudyn (CUDA Dynamic)** is a generic C++ CUDA library designed for the efficient execution of irregular applications on NVIDIA GPUs. At its core, the library features a dynamic task scheduling mechanism at the thread block level

## Motivation

Modern GPUs offer enormous potential for parallel computations but are primarily optimized for regular applications where control flow and memory accesses are predictable. Irregular applications, characterized by data-dependent control flow and memory access patterns (e.g., in graph algorithms, sparse matrix operations, or simulations with stochastic runtimes), pose a significant challenge for efficient GPU utilization. *Cudyn* addresses this challenge by providing a flexible and user-friendly framework for dynamically scheduling such irregular tasks.

## Core Features

* **Dynamic Task Scheduling:** Implements a block-internal dynamic scheduler that allows GPU threads to atomically fetch tasks from a shared pool as needed. This aims for better utilization and reduction of idle times, especially with highly variable task runtimes.
* **Genericity via C++ Templates:** Core components are templated to support arbitrary user-defined logic (as functors or lambdas) and data types, enabling broad applicability.
* **Policy-Based Design:** The `Launcher` and `Schedulers` are designed using the policy idiom, offering high modularity and extensibility for future scheduling strategies.
* **Abstraction and User-Friendliness:**
    * `Cudyn::Utils::Memory::CudynDevicePointer`: An RAII-based wrapper for safe and simplified GPU memory management.
    * `Cudyn::Utils::GridConfiguration`: A component for flexible definition and validation of kernel launch configurations.
* **Integrated SpMV Support:**
    * `Cudyn::CSR`: A module for handling Sparse Matrix-Vector Multiplication (SpMV) for matrices in the Compressed Sparse Row (CSR) format.
    * `Cudyn::MatrixMarketParser`: A parser for reading matrices from the common MatrixMarket format.
* **Profiling Tools:** Includes a `launchProfiled` function for easy measurement and analysis of kernel runtimes.

## Architectural Overview

*Cudyn* has a modular structure, primarily organized through C++ namespaces. A deep classical object-oriented hierarchy has been intentionally avoided in favor of flexibility and template-based composition.

The main components are:
* **`Cudyn::Scheduler`**: Contains the logic for dynamic task distribution within the GPU kernel (e.g., `genericIrregularKernel`, `genericIrregularKernelLowAtomics`).
* **`Cudyn::Launcher`**: Serves as the central API interface for invoking kernels via a chosen scheduling policy.
* **`Cudyn::Utils`**:
    * `GridConfiguration`: For managing and validating grid and block dimensions.
    * `Memory`: Provides the `CudynDevicePointer` wrapper for memory management.
* **`Cudyn::CSR`** (optional): Offers data structures and kernel functors for SpMV operations.
* **`Cudyn::MatrixMarketParser`** (optional): Enables loading of matrices.

## Prerequisites

* CUDA Toolkit (Version 12.x or newer recommended)
* A C++ compiler with support for C++17 (or newer)

## Installation

git clone [https://github.com/PerHuepenbecker/Cudyn.git](https://github.com/PerHuepenbecker/Cudyn.git)

