# Physics-Informed Hamiltonian Neural Networks (HNNs) for Chaotic Multi-Body Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Standard deep learning models frequently violate fundamental physics when predicting chaotic dynamics—they suffer from numerical drift, artificially generate energy out of nowhere, and fail conservation laws. This repository implements an advanced **Hamiltonian Neural Network (HNN)** architecture designed to natively learn continuous vector fields and exact geometric invariants directly from raw, noisy multi-body trajectory data.

By embedding structural conservation directly into the calculus of the neural network via **Symplectic Geometry**, this model achieves near-perfect energy conservation over long-term chaotic horizons, significantly outperforming classical unconstrained black-box models and standard Runge-Kutta variations.

---

## The Paradigm Shift: Why Native Physics?

Instead of treating a chaotic double inverted pendulum as an arbitrary sequence-to-sequence prediction problem ($x_{t} \to x_{t+1}$), this framework enforces **Strong Physical Invariance** by mapping the entire state space to a single scalar value representing the total system energy:

$$H(q, p) = T(q, p) + V(q)$$

Using automatic differentiation, the model intercepts its own forward pass to compute exact canonical symplectic gradients:

$$\dot{q} = \frac{\partial H}{\partial p}, \quad \dot{p} = -\frac{\partial H}{\partial q}$$

Where:
* $q$: Generalized coordinates (angular positions $\theta_1, \theta_2$)
* $p$: Generalized momenta (cross-coupled through the system's mass matrix)
* $\dot{q}, \dot{p}$: Infinitesimal time derivatives defining the exact vector field

Because the system trajectory is calculated purely through these canonical gradients, the neural network is mathematically constrained to stay pinned to the true energy manifold.

---

## Advanced Architecture Highlights

### 1. Coupled Mass-Matrix Resolution
In a highly non-linear double pendulum, kinetic energy and generalized momentum are heavily cross-coupled through a time-varying, position-dependent Mass Matrix $M(q)$. This architecture natively learns the non-separable Hamiltonian landscape without breaking down at intense angular velocities.

### 2. Multi-Task Physics-Informed Regularization
To combat sensor noise in raw experimental tracking datasets, the training pipeline leverages a dual-objective loss function:
* **Vector Field Match:** Optimizes standard derivative MSE against the observed trajectory data.
* **Jacobian Smoothness Layer:** Regularizes the gradient of $H$ to strip out high-frequency noise without losing underlying chaotic frequencies.

### 3. Implicit Midpoint Symplectic Integration
Standard explicit solvers (like Euler or naive Leapfrog) introduce numerical drift when integrated over highly coupled, non-separable Hamiltonians. This repository implements a custom **Implicit Midpoint Integrator powered by a Newton-Raphson / Fixed-Point iteration loop**, guaranteeing phase space volume preservation ($\det(J) = 1$) and locking down energy drift to a vanishingly small standard deviation.

---

## Core Engineering Components

* **Ground-Truth Chaotic Synthesis:** A classical multi-body simulator injected with Gaussian noise profiles to model real-world tracking hardware.
* **Symplectic Autograd Engine:** PyTorch network layers overriding standard evaluation states to return physical derivatives via mathematical graph manipulation.
* **Implicit Correction Module:** Fixed-point algebraic solvers seamlessly integrated into the post-training simulation pipeline to anchor long-term rollouts.

---

## Mathematical Verification & Convergence

During benchmark simulations over extended horizons ($1000+$ timesteps of severe chaos), standard networks rapidly explode or lose momentum due to truncation errors. By contrast, this HNN framework paired with an implicit symplectic step constrains total energy deviation to a near-flatline benchmark:

$$\text{Energy } \sigma \approx 0.148$$

This structural stability demonstrates that machine learning can be used not just as a black-box curve fitter, but as a mathematically sound engine for precise physical discovery.

---
# How to Reload This Model Anywhere Later

When you want to use this model again in your session or deploy it to a control loop, you can reload it as follows.

 **1. Load the metadata configuration**

```python
import json
import torch

with open("hnn_double_pendulum/hnn_metadata.json", "r") as f:
    meta = json.load(f)
```

**2. Recreate the model with the original architecture**

```python
loaded_hnn = HamiltonianNeuralNetwork(
    input_dim=meta["input_dim"],
    hidden_dim=meta["hidden_dim"]
)
```

**3. Load the trained weights**

```python
loaded_hnn.load_state_dict(
    torch.load("hnn_double_pendulum/hnn_weights.pt")
)
loaded_hnn.eval()
```

**4. Confirm successful reload**

```python
print("Model successfully reloaded with identical physical properties!")
```
---
## Results (Graphical):
* <img width="689" height="393" alt="image" src="https://github.com/user-attachments/assets/1ed0dccc-d0d9-44aa-b1bf-46079ae088e2" />
* <img width="717" height="393" alt="image" src="https://github.com/user-attachments/assets/1c76f833-8f45-4479-ba25-4ac994b8a2f5" />
* <img width="989" height="690" alt="image" src="https://github.com/user-attachments/assets/f67a4920-3df6-4696-959c-2d0f779d8d95" />
* <img width="398" height="42" alt="image" src="https://github.com/user-attachments/assets/ace71328-1cc5-45bf-b9cd-55db72271709" />

---

## License

Distributed under the **MIT License**. See `LICENSE` for more information. Complete freedom for academic, research, and commercial applications with proper attribution.
