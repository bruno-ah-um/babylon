# Project OpenJDK Babylon

## Java to Accelerator Code via MLIR/TOSA

### Overview

**Project Babylon** is an OpenJDK initiative that extends Java beyond the traditional JVM execution model to target *foreign programming models* such as SQL engines, differentiable programming systems, machine learning runtimes, and hardware accelerators.

At the core of Babylon is **Code Reflection**, an enhanced reflective programming capability that exposes Java programs as structured code models rather than opaque bytecode. These models enable compiler-style analysis and transformation of Java code into alternative intermediate representations (IRs).

This repository is an example Project Babylon application that demonstrates how Java code can be lowered into accelerator-ready representations using MLIR, with a focus on the TOSA (Tensor Operator Set Architecture) dialect. The goal is to allow Java developers to write high-level, idiomatic Java code that can ultimately execute on GPUs, NPUs, and other specialized hardware.

---

## Motivation

Modern compute platforms increasingly rely on heterogeneous hardware, yet programming such systems often requires abandoning high-level languages in favor of C++, CUDA, or specialized DSLs.

This project explores a different approach:

- Use Java as a front-end language
- Extract semantic structure from Java methods via Code Reflection
- Lower that structure into **MLIR**, enabling:
  - Hardware-agnostic optimization
  - Reuse of existing compiler infrastructure
  - Deployment to diverse accelerator backends

---

## Relationship to Project Babylon

Project Babylon provides the foundational capability:

- `@CodeReflection`
- `CodeModel`
- Structured representations of Java methods and control flow

This repository builds *on top* of Babylon by:

- Consuming Babylon `CodeModel`s
- Translating Java operations into a **dataflow-oriented IR**
- Emitting **TOSA MLIR**, suitable for downstream compilation

---

## Why MLIR and TOSA?

MLIR provides a modular, extensible compiler infrastructure designed for heterogeneous systems.

TOSA was chosen as the initial target dialect because it:

- Provides a standardized tensor-level abstraction
- Is explicitly designed for hardware-agnostic lowering
- Serves as a stable interface between frontends and accelerator backends

By lowering Java into TOSA, this project can leverage:

- MLIR optimization passes (fusion, tiling, canonicalization)
- Existing and emerging TOSA-to-hardware backends
- LLVM for final code generation

## Project Status

> **Experimental / Exploratory**

This project is intended for research and prototyping. APIs, supported Java features, and lowering strategies are expected to evolve as Project Babylon matures.

---

## License

This project follows the licensing terms of **OpenJDK Project Babylon**.  
See the upstream repository and license files for details.

---

*Project Babylon Example: Exploring Java as a Frontend for MLIR and Accelerators.*
