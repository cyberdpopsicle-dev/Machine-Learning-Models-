Validation and Experimental Results

This section documents the training convergence metrics and the learned physical solution space for the Physics-Informed Neural Network (PINN) optimizing the 1D wave equation domain.

---

## 1. Training Convergence Analysis

The network was trained for 1500 epochs using the Adam optimizer with a learning rate of 0.001. The objective function dynamically balanced two components: supervised boundary conditions and unsupervised partial differential equation (PDE) residuals.

### Loss Optimization Trajectory

| Epoch | Boundary Loss | Physics Residual Loss | Total Combined Loss |
| :--- | :--- | :--- | :--- |
| 0000 | 0.15570 | 0.00043 | 0.15613 |
| 0300 | 0.02650 | 0.00020 | 0.02670 |
| 0600 | 0.02466 | 0.00040 | 0.02506 |
| 0900 | 0.00204 | 0.00081 | 0.00286 |
| 1200 | 0.00092 | 0.00027 | 0.00119 |
| 1500 | 0.00054 | 0.00012 | 0.00065 |

### Key Observations
* **Initial State (Epoch 0):** The system starts with a dominant boundary loss (0.15570), while the physics residual loss begins exceptionally low (0.00043) due to the random initialization fields near zero satisfying the unexcited state components.
* **Transient Phase (Epoch 300 - 600):** A minor optimization bottleneck occurs as the network balances fitting the strict boundary walls against the inner differential constraints. The physics residual peaks slightly at epoch 900 (0.00081) as the system forces regularized wave propagation structures through the unmapped collocation spaces.
* **Asymptotic Convergence (Epoch 1200 - 1500):** Both objective vectors decay smoothly in tandem. By epoch 1500, the network reaches a total loss threshold of 0.00065, proving that the parameters have successfully converged to a mathematically valid physical solution profile.

---

## 2. Physical Solution Space Realization

The figure below shows the output of the trained network across the continuous spatio-temporal domain, mapping space ($x \in [-1, 1]$) against time ($t \in [0, 1]$).
