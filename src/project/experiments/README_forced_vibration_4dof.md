# Forced Vibration — 4DOF Experiment

## Overview

This script studies a nonlinear **4-degree-of-freedom structural system** under **external forcing**.
Unlike the free-vibration case, the system response here depends not only on the initial condition, but also on a time-dependent excitation signal.

The purpose of the experiment is to evaluate how well a physics-informed neural ODE can reproduce the structural response under forcing and whether the learned discrepancy term can later be interpreted symbolically.

---

## Goal

The goal of this experiment is to model the dynamics of a nonlinear 4DOF structure under forced vibration and compare the learned response against the true physical system.

More specifically, the script is used to:

- simulate the true forced structural response,
- evaluate learned neural ODE models,
- compare performance under different excitation settings,
- and analyze the learned discrepancy using symbolic regression.

This experiment examines both predictive accuracy and interpretability in the presence of external loading.

---

## What is used

This script uses the following main tools and components:

- **PyTorch** for tensor computation, neural models, and checkpoint loading,
- **torchdiffeq** for integrating the dynamical system,
- **PINODE models** for physics-informed learning of structural response,
- **Matplotlib** for plotting and comparing trajectories,
- **PySINDy** for symbolic recovery of discrepancy terms.

The script may also use excitation records or analytical forcing functions depending on the setup.

---

## Physical system

The system is a nonlinear **4DOF structural model** subject to external excitation.

Its state is written as:

\[
h = [x_1, x_2, x_3, x_4, v_1, v_2, v_3, v_4]
\]

where:

- \(x_1, x_2, x_3, x_4\) are the displacements,
- \(v_1, v_2, v_3, v_4\) are the velocities.

So the state vector has dimension **8**.

The model includes:

- linear mass effects,
- linear damping,
- linear stiffness,
- nonlinear structural behavior,
- and an external forcing term.

---

## Model input

The input to the model is the **current state of the structure**:

\[
h(t) = [x_1, x_2, x_3, x_4, v_1, v_2, v_3, v_4]
\]

So the model receives:

- the current displacements,
- and the current velocities.

In the forced case, the evolution also depends on:

- the current time,
- and the forcing signal applied to the system.

Conceptually, the model uses the current structural state together with the forcing context to determine the next rate of change.

---

## Model output

The output of the model is the **time derivative of the state**:

\[
\dot{h}(t) = [\dot{x}_1, \dot{x}_2, \dot{x}_3, \dot{x}_4, \dot{v}_1, \dot{v}_2, \dot{v}_3, \dot{v}_4]
\]

This means:

- the first four outputs are the velocity components,
- the last four outputs are the acceleration components.

So the model maps:

- current state
to
- state derivative.

---

## Input to the script

The script uses:

- structural system parameters,
- initial conditions,
- time grids,
- forcing functions or excitation signals,
- saved trained model checkpoints,
- and optionally earthquake or synthetic loading data.

Because this is a forced-vibration experiment, the excitation signal is a central part of the setup.

---

## Output of the script

The script produces:

- predicted structural trajectories,
- comparisons between true and learned forced response,
- evaluation plots,
- discrepancy analysis,
- and symbolic regression expressions for the learned nonlinear or missing part of the dynamics.

---

## Summary

This script is a **4DOF forced-vibration structural dynamics experiment**.
It is used to test whether a physics-informed neural ODE can learn the behavior of a nonlinear structural system under external loading, while keeping the learned correction term interpretable and suitable for symbolic recovery.
