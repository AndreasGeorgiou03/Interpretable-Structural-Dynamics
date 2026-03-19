# Interpretable Structural Dynamics

## Overview

This repository contains code for studying **structural identification with Physics-informed Neural Ordinary Differential Equations (PINODEs)**, with a particular focus on **interpretable modeling of structural dynamics**.

The overall project is inspired by the paper **Structural Identification with Physics-informed Neural Ordinary Differential Equations** by Lai, Mylonas, Nagarajaiah, and Chatzi. The central idea of that work is to represent structural dynamics as a combination of:

- a **physics-informed baseline term**, based on known or assumed structural mechanics,
- and a **learned discrepancy term**, represented by a neural network.

This codebase follows that general philosophy and extends it into a set of numerical and symbolic experiments on multi-degree-of-freedom structural systems. In particular, the repository studies whether neural ODE models can not only reproduce structural response accurately, but also help recover **interpretable nonlinear laws** from the learned dynamics. :contentReference[oaicite:1]{index=1}

---

## Goal

The main goal of this repository is to explore whether machine learning models, especially **physics-informed neural ODEs**, can be used for structural system identification in a way that is both:

- **accurate**, in terms of trajectory prediction and dynamic response estimation,
- and **interpretable**, in terms of understanding the nonlinear or unknown part of the learned dynamics.


---

## Experimental organization

The repository is organized around **four main experiments**.

Each experiment focuses on a different structural setting or identification stage.
To keep the main README concise, the detailed explanation of each experiment is placed in its own dedicated README inside the project source tree.

The four experiments are:

1. **4DOF free-vibration experiment**
2. **4DOF forced-vibration experiment**
3. **3DOF forced-vibration NSD experiment**
4. **3DOF symbolic recovery experiment using PySR**

---

## General modeling idea

Across the repository, the main modeling pattern is the following:

1. Define a structural dynamical system in state-space form.
2. Split its dynamics into:
   - a known or assumed physics-based term,
   - and a neural discrepancy term.
3. Train the neural model so that the predicted trajectories match the true or measured response.
4. Evaluate the model under extrapolation, or new forcing conditions.
5. Analyze what the neural component has learned.

---

## Code structure

The repository is divided into a parent-level project structure and a source-level experiment structure.

### Parent-level structure

- **`README.md`**
  General overview of the whole repository.

- **`data/`**
  Input data files such as earthquake records and other external files used in forced-vibration experiments.

- **`models/`**
  Saved trained model checkpoints and learned weights.

- **`logs/`**
  Generated figures, plots, and evaluation outputs.

---

### Source-level structure

The main code lives under:

- **`src/project/`**

Inside this folder, the code is split into the main components of the project:

- **experiment scripts**
  Scripts that run the different numerical and symbolic experiments,

- **model definitions**
  PINODE, NODE, NSD, and related neural/physics-based model implementations,

- **data utilities**
  Helpers for loading and preprocessing excitation signals,

- **symbolic recovery scripts**
  Tools for extracting interpretable symbolic laws from trained neural components.

The README files placed inside `src/project/` describe these experiments individually.

---

## Reference

This repository is based on the ideas presented in:

**Lai, Z., Mylonas, C., Nagarajaiah, S., & Chatzi, E.**
*Structural Identification with Physics-informed Neural Ordinary Differential Equations*
Journal of Sound and Vibration, 2021.

---
