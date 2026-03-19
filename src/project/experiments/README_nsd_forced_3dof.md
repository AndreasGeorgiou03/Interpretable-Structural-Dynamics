# NSD Forced Vibration — 3DOF Experiment

## Overview

This script studies a **3-degree-of-freedom structural system** under **forced vibration**, with emphasis on learning and recovering the behavior of a **negative-stiffness-device-like (NSD) nonlinear component**.

The experiment is organized in two stages. First, a base physics-informed model is used to account for the linear part of the structural mismatch. Then, a second model is introduced to learn the additional nonlinear NSD contribution while keeping the first learned component fixed.

The overall purpose is to examine whether a structured neural ODE approach can:

- capture linear modeling errors,
- learn an additional nonlinear restoring effect,
- and reproduce structural response under earthquake excitation.

---

## Goal

The goal of this experiment is to model a **forced 3DOF nonlinear structural system** in a progressively interpretable way.

More specifically, the script is used to:

- generate synthetic structural responses under external excitation,
- identify mismatch in the nominal linear model,
- update the effective linear dynamics using a first learned correction,
- introduce a second neural component to capture NSD-type nonlinear behavior,
- and evaluate the full model under both Kobe and El Centro earthquake inputs.

This experiment is designed to separate the learning problem into two parts:

1. a **linear correction stage**, and
2. a **nonlinear NSD recovery stage**.

That makes the learned dynamics easier to interpret than a single black-box model trained on the full problem directly.

---

## What is used

This script uses the following main tools and components:

- **PyTorch** for tensors, neural networks, optimization, and checkpoint loading,
- **torchdiffeq** for numerical integration of the ODE system,
- **custom rollout solvers** based on central difference and Newmark-type updates,
- **physics-only and physics-informed 3DOF models**,
- **an NSD neural network** for the nonlinear restoring term,
- **earthquake excitation records** in `.AT2` and `.dat` format,
- **Matplotlib** for trajectory visualization.

The script imports several project-specific models, including:

- a physics-only linear model,
- a linear truth model,
- a linear PINODE mismatch model,
- an NSD truth model,
- an NSD PINODE model,
- and a dedicated `NSD_Net` used for the nonlinear component. :contentReference[oaicite:1]{index=1}

---

## Physical system

The physical system is a **3DOF structural model** written in state-space form.

Its state is:

\[
h = [x_1, x_2, x_3, v_1, v_2, v_3]
\]

where:

- \(x_1, x_2, x_3\) are the displacements,
- \(v_1, v_2, v_3\) are the velocities.

So the state vector has dimension **6**.

The model includes:

- a mass matrix \(M\),
- a damping matrix \(C\),
- a stiffness matrix \(K\),
- an external forcing term derived from earthquake input,
- and, in the nonlinear phase, an NSD-type restoring force.

The script also defines a bilinear-style NSD force law used as the synthetic truth for the nonlinear experiment. :contentReference[oaicite:2]{index=2}

---

## Model input

The input to the model is the **current state of the structure**:

\[
h(t) = [x_1, x_2, x_3, v_1, v_2, v_3]
\]

This means the model receives:

- the current displacement of each degree of freedom,
- and the current velocity of each degree of freedom.

In the forced-vibration setting, the model evolution also depends on:

- the current time \(t\),
- the excitation signal \(u(t)\),
- and the forcing amplitude.

So conceptually, the model uses the current structural state together with the external forcing to determine the next time derivative.

---

## Model output

The output of the model is the **time derivative of the state**:

\[
\dot{h}(t) = [\dot{x}_1, \dot{x}_2, \dot{x}_3, \dot{v}_1, \dot{v}_2, \dot{v}_3]
\]

In practical terms:

- the first three outputs are the velocity components,
- the last three outputs are the acceleration components.

So the model maps:

- current state
to
- state derivative.

This is why the helper function `model_accel(...)` extracts only the acceleration part of the output when needed by the custom rollout schemes. :contentReference[oaicite:3]{index=3}

---

## Structure of the experiment

This script is divided into two main modeling phases.

### 1. Linear mismatch recovery

In the first phase, the script assumes the true system is still linear, but the nominal matrices are not correct.
The true stiffness and damping matrices are deliberately perturbed relative to the nominal ones.

A linear PINODE model is then used to learn a correction on top of the nominal physics.
After training or loading this model, the script computes an **effective update** of the stiffness and damping matrices from the learned linear map.

This stage answers the question:

- can the model recover hidden linear mismatch in the system?

### 2. NSD nonlinear recovery

In the second phase, the true system includes an NSD-type nonlinear force.
A second neural component is introduced to learn this missing nonlinear effect.

The first network, which already captures the linear mismatch, is frozen.
Then a second network is trained on top of it to account for the nonlinear NSD behavior.

This stage answers the question:

- can the model isolate and learn the nonlinear NSD contribution after the linear part has already been identified?

---

## Excitation data

This experiment uses external earthquake excitation records.

The script reads:

- a **Kobe earthquake** `.AT2` file,
- and an **El Centro** `.dat` file.

The excitation signals are resampled to the internal simulation time grid before they are used in the models.

This allows the learned models to be trained and evaluated on realistic forcing signals rather than only synthetic sinusoidal inputs. :contentReference[oaicite:4]{index=4}

---

## Input to the script

The script uses the following inputs:

- nominal mass, damping, and stiffness matrices,
- perturbed “true” matrices for synthetic data generation,
- an initial state \(h_0\),
- earthquake acceleration records,
- forcing amplitudes,
- saved model checkpoints for the linear and NSD stages.

So the script combines both **structural parameters** and **forcing data** to generate and evaluate trajectories.

---

## Output of the script

The script produces:

- structural trajectories for the synthetic truth model,
- structural trajectories for the physics-only model,
- structural trajectories for the learned PINODE models,
- comparisons between nominal, corrected, and true dynamics,
- estimated updated stiffness and damping matrices,
- and saved plots for displacement and velocity response.

The output is used to evaluate how well each stage of the model reproduces the system response.

---

## Interpretation of the learned model

A key idea in this script is that the learned dynamics are **split into interpretable parts**.

- The first learned component captures the **linear mismatch** between nominal and true system matrices.
- The second learned component captures the **additional nonlinear NSD behavior**.

This separation is useful because it avoids forcing a single model to explain everything at once.
Instead, the model structure reflects the physical modeling logic:

1. start from known nominal physics,
2. correct the linear part,
3. then learn the remaining nonlinear effect.

That makes the final model easier to understand and analyze.

---

## Summary

This script is a **3DOF forced-vibration NSD experiment**.
It is used to test whether a structured physics-informed neural ODE can first recover linear model mismatch and then learn an additional NSD-type nonlinear force under earthquake excitation.

The experiment combines:

- structural dynamics,
- external forcing,
- staged neural correction,
- matrix mismatch recovery,
- nonlinear NSD learning,
- and trajectory comparison under realistic earthquake inputs.

Its main purpose is not only to fit the response accurately, but also to organize the learned dynamics in a way that remains physically meaningful.
