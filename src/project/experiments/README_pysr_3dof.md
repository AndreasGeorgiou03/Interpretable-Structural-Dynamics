# PySR Symbolic Recovery — 3DOF NSD Experiment

## Overview

This script is used for the **symbolic recovery stage** of the 3DOF NSD forced-vibration pipeline.

Its purpose is not to train the main structural dynamics model from scratch, but to take the response of an already defined and already trained physics-informed model and then recover a **symbolic expression** for the nonlinear NSD-related force using **PySR**.

At a high level, the script:

1. loads earthquake excitation data,
2. reconstructs the structural response of the true system,
3. loads the trained linear and NSD-enhanced models,
4. extracts the nonlinear force learned by the NSD network,
5. builds a dataset from structural states and corresponding learned force values,
6. transforms the raw coordinates into physically meaningful drift-based features,
7. and fits a symbolic regression model to approximate that learned nonlinear force.

So this script belongs to the **interpretability part** of the project: it translates the learned neural nonlinear behavior into a symbolic mathematical expression.

---

## Goal

The goal of this script is to recover an interpretable symbolic equation for the nonlinear NSD force in the **3DOF forced-vibration** setting.

More specifically, the script is used to:

- simulate the true structural trajectory under earthquake excitation,
- evaluate the trained PINODE + NSD model on that trajectory,
- extract the nonlinear force contribution predicted by the NSD network,
- define a compact feature representation using inter-story drifts and drift velocities,
- and use **PySR** to discover a symbolic formula that approximates this nonlinear force.

The main idea is to move from:

- a learned neural nonlinear term

to

- a sparse symbolic equation that can be inspected and interpreted physically.

---

## What is used

This script uses the following main tools and components:

- **NumPy** for array manipulation and preprocessing,
- **PyTorch** for tensors, model loading, and state handling,
- **torchdiffeq** for numerical integration of the ground-truth structural trajectory,
- **PySR** for symbolic regression,
- **juliacall** because PySR relies on the Julia backend,
- project-specific structural dynamics models,
- and earthquake excitation data loaded from external files. :contentReference[oaicite:1]{index=1}

The script imports and uses the following project components:

- `UFunFromSamples` for building excitation functions from sampled earthquake input,
- `PINODEFuncLinear3DOF` for the learned linear correction model,
- `PINODEFuncNSD_3DOF` for the model that includes the NSD neural component,
- `TruthPhaseNSD_3DOF` as the synthetic ground-truth forced nonlinear system,
- `NSD_Net` as the neural network used for the nonlinear NSD contribution,
- and `ensure_all_earthquakes()` to load or download the earthquake input files. :contentReference[oaicite:2]{index=2}

---

## Physical setting

The underlying physical system is a **3-degree-of-freedom structural model** under **forced vibration**.

Its state is written as:

\[
h = [x_1, x_2, x_3, v_1, v_2, v_3]
\]

where:

- \(x_1, x_2, x_3\) are the displacements,
- \(v_1, v_2, v_3\) are the velocities.

So the full state is **6-dimensional**.

The system includes:

- mass,
- damping,
- stiffness,
- external earthquake excitation,
- linear mismatch between nominal and true matrices,
- and an NSD-type nonlinear restoring effect.

---

## What this script does conceptually

This script assumes that the main learning stages have already happened.

That means:

- a linear correction model has already been trained,
- an NSD-enhanced model has already been trained,
- and the current task is now to **analyze what the NSD network learned**.

The script therefore works as a **post-processing and symbolic identification stage** rather than a primary training script.

Its purpose is to answer the question:

> Can the nonlinear force learned by the neural NSD model be approximated by a simple symbolic expression?

---

## Model input

There are two important notions of input in this script.

### 1. Input to the structural model

The structural model receives the current system state:

\[
h(t) = [x_1, x_2, x_3, v_1, v_2, v_3]
\]

So the model input consists of:

- 3 displacements,
- 3 velocities.

In the forced-vibration setting, the model evolution also depends on:

- time \(t\),
- the earthquake excitation signal \(u(t)\),
- and the excitation amplitude.

So the structural model uses the current state together with the forcing context to determine the next derivative.

---

### 2. Input to the symbolic regression model

The symbolic regression model does **not** use the raw state directly in the final feature representation.

Instead, the script converts the state into physically meaningful features:

- \(d_1 = x_1\)
- \(d_2 = x_2 - x_1\)
- \(d_3 = x_3 - x_2\)

which are the **drifts**, and

- \(w_1 = v_1\)
- \(w_2 = v_2 - v_1\)
- \(w_3 = v_3 - v_2\)

which are the **drift velocities**. :contentReference[oaicite:3]{index=3}

So the final input to PySR is:

\[
[d_1, d_2, d_3, w_1, w_2, w_3]
\]

This choice is important because it expresses the structural state in coordinates that are more physically meaningful for symbolic discovery than the raw absolute coordinates.

---

## Model output

Again, there are two notions of output in this script.

### 1. Output of the structural model

The structural model outputs the **time derivative of the state**:

\[
\dot{h}(t) = [\dot{x}_1, \dot{x}_2, \dot{x}_3, \dot{v}_1, \dot{v}_2, \dot{v}_3]
\]

So:

- the first three outputs are velocities,
- the last three outputs are accelerations.

This is the standard neural ODE interpretation of the structural model.

---

### 2. Output used as target for symbolic regression

The target used for symbolic regression is **not** the full state derivative.

Instead, the script extracts the nonlinear NSD contribution predicted by the NSD network and converts it into a force quantity on **DOF 1**. :contentReference[oaicite:4]{index=4}

So the symbolic regression target is:

- a scalar nonlinear force value for each time sample.

This means PySR learns a mapping of the form:

\[
[d_1, d_2, d_3, w_1, w_2, w_3]
\;\rightarrow\;
f_{\mathrm{NSD}}
\]

where \(f_{\mathrm{NSD}}\) is the learned nonlinear force associated with the NSD effect.

---

## Earthquake input data

The script uses two earthquake records:

- **El Centro**
- **Kobe**

and includes helper functions to load and resample them to the internal simulation time grid. :contentReference[oaicite:5]{index=5}

In the current execution path, the script loads the **El Centro** signal, builds a sampled excitation function, and applies it to both the truth model and the learned models. :contentReference[oaicite:6]{index=6}

This makes the symbolic dataset grounded in a realistic forced-vibration scenario rather than a purely synthetic excitation.

---

## Dataset construction

The symbolic regression dataset is built in the following way:

1. simulate the true structural trajectory under earthquake forcing,
2. pass the states through the trained NSD-enhanced model,
3. extract the nonlinear NSD acceleration contribution,
4. convert that acceleration contribution into force using the mass matrix,
5. keep the first DOF nonlinear force as the target,
6. convert the full structural state into drift and drift-velocity features,
7. fit symbolic regression on the resulting input-output pairs. :contentReference[oaicite:7]{index=7}

So each row of the symbolic dataset represents:

- the structural condition of the system at a time instant,
- and the corresponding nonlinear force learned by the network.

---

## Symbolic regression settings

The script uses **PySRRegressor** with a search space based on relatively simple operators, including:

- addition,
- subtraction,
- multiplication,
- absolute value,
- sign,
- and ReLU. :contentReference[oaicite:8]{index=8}

The objective is to find a compact symbolic expression that explains the learned nonlinear force with low loss and reasonable simplicity.

This is consistent with the overall purpose of the project: not only to model structural response accurately, but also to recover interpretable laws from learned behavior.

---

## Input to the script

The script uses the following inputs:

- nominal structural matrices \(M\), \(C\), and \(K\),
- modified “true” matrices \(M_{\text{true}}, C_{\text{true}}, K_{\text{true}}\),
- a zero initial condition,
- earthquake excitation files,
- saved model weights for the linear correction model,
- saved model weights for the NSD model,
- and the chosen symbolic feature representation. :contentReference[oaicite:9]{index=9}

So this script depends on both:

- the structural simulation setup,
- and the previously trained models.

---

## Output of the script

The script produces:

- the symbolic regression dataset,
- a discovered symbolic equation for the nonlinear NSD force,
- training fit statistics such as MSE and RMSE,
- and a human-readable equation where generic variables are replaced by the drift-based feature names. :contentReference[oaicite:10]{index=10}

So the final output is an interpretable symbolic approximation of the learned nonlinear force law.

---

## Summary

This script is a **symbolic recovery pipeline for the 3DOF NSD forced-vibration experiment**.

It takes the nonlinear force learned by a trained NSD-enhanced structural model and fits a symbolic equation to that force using PySR.

In this sense, the script acts as the bridge between:

- a neural representation of nonlinear structural behavior,
- and an interpretable symbolic representation of the same behavior.

Its main purpose is to turn a learned NSD correction into a mathematical expression that can be inspected, compared, and interpreted physically.
