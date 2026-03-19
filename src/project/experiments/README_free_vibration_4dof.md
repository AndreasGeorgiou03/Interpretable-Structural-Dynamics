# Free Vibration — 4DOF Experiment

## Overview

This script studies a nonlinear **4-degree-of-freedom structural system** under **free vibration**.
In this setting, the motion is driven only by the system dynamics and the chosen initial condition, without any external forcing.

The purpose of the experiment is to evaluate how well a physics-informed neural ODE can learn the system response and whether the learned part of the dynamics can later be analyzed in an interpretable way.

---

## Goal

The goal of this experiment is to model the dynamics of a nonlinear 4DOF structure under free vibration and compare learned trajectories against the true physical response.

More specifically, the script is used to:

- simulate the true nonlinear structural response,
- evaluate learned neural ODE models,
- compare different levels of embedded physical knowledge,
- and analyze the learned discrepancy term using symbolic regression.

---

## What is used

This script uses the following main tools and components:

- **PyTorch** for tensors, neural networks, and model loading,
- **torchdiffeq** for numerical integration of the ODE system,
- **PINODE models** for learning the structural dynamics,
- **PySINDy** for sparse symbolic recovery of the learned discrepancy.

The script also loads pretrained model weights (optionally) and compares model predictions against the ground-truth trajectories.

---

## Physical system

The system is a nonlinear **4DOF structural model**.

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
- and a nonlinear restoring component.

---

## Model input

The input to the model is the **current state of the structure**:

\[
h(t) = [x_1, x_2, x_3, x_4, v_1, v_2, v_3, v_4]
\]

This means the model receives:

- the current displacement of each DOF,
- and the current velocity of each DOF.

So the input represents the instantaneous structural condition at time \(t\).

---

## Model output

The output of the model is the **time derivative of the state**:

\[
\dot{h}(t) = [\dot{x}_1, \dot{x}_2, \dot{x}_3, \dot{x}_4, \dot{v}_1, \dot{v}_2, \dot{v}_3, \dot{v}_4]
\]

In practical terms:

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
- simulation time grids,
- saved trained model checkpoints.

Since this is a free-vibration setup, no external excitation is required.

---

## Output of the script

The script produces:

- predicted structural trajectories,
- comparisons between true and learned response,
- evaluation plots,
- discrepancy analysis,
- and symbolic regression expressions for the learned nonlinear term.

---

## Summary

This script is a **4DOF free-vibration structural dynamics experiment**.
It is used to test whether a physics-informed neural ODE can learn the behavior of a nonlinear structural system from its state evolution alone, while keeping the learned dynamics open to interpretation.
