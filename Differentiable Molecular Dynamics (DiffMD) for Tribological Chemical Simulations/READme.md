# Day 8 :Differentiable Molecular Dynamics (DiffMD) for Tribological Chemical Simulations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyTorch_Geometric-Graph_Neural_Networks-34c28b)](https://pytorch-geometric.readthedocs.io/)

A pure PyTorch and PyTorch Geometric (PyG) implementation of a **Differentiable Molecular Dynamics (MD)** workflow. This repository contains an end-to-end framework designed for structural and tribological chemical simulations, where forces are explicitly derived as the negative gradient of a learned potential energy surface via reverse-mode automatic differentiation.



## Overview

Traditional molecular dynamics relies on empirical force fields or expensive quantum chemistry (DFT) calculations. This framework models atomic structures as graphs and utilizes a Graph Neural Network (GNN) variant to map chemical configurations directly to a scalar potential energy surface ($E$). 

By enforcing physical constraints natively, atomic forces ($\mathbf{F}$) are calculated directly from the energy using `torch.autograd.grad`:
$$\mathbf{F} = -\nabla_{\mathbf{R}} E$$

This ensures that the predicted forces are strictly conservative and physically consistent with the learned energy landscape.

## Key Features

- **End-to-End Differentiable Physics:** Force fields are completely conservative, trained natively via energy gradients.
- **Custom Neighborhood Mapping:** High-performance, manual molecular graph mapping engine with customizable radius cut-offs ($r = 3.5\text{ Å}$), eliminating self-loops and isolating multi-batch topologies.
- **Heterogeneous Chemical Elements:** Support for multi-element mappings (Carbon, Oxygen, and Hydrogen) represented as one-hot node features.
- **Extensible Architecture:** Fully integrated with PyTorch Geometric and ready to be swapped with advanced equivariant backbones (such as `e3nn` or MACE).

---

## Architecture & Workflow

1. **Graph Construction:** Atomic positions ($\mathbf{R}$) are mapped to adjacency lists using a pairwise distance matrix thresholded by a radial cut-off.
2. **Energy Prediction ($E$):** Node features (chemical species) and edge features (distances) are passed through an Edge-conditioned Multi-Layer Perceptron (MLP) to output a scalar potential energy.
3. **Autograd Force Derivation ($\mathbf{F}$):** Dynamic backpropagation through the coordinates layer computes the analytical gradients.
---
## Results and Visualization:
<img width="768" height="393" alt="image" src="https://github.com/user-attachments/assets/94056273-d4b8-4c94-a3d4-306379e46e55" />
<img width="618" height="393" alt="image" src="https://github.com/user-attachments/assets/c097e6b2-03b4-4980-a483-fa55892fe21c" />
<img width="568" height="96" alt="image" src="https://github.com/user-attachments/assets/8481243b-34f2-48cd-98db-1e7ddaa81673" />
<img width="526" height="150" alt="image" src="https://github.com/user-attachments/assets/ae590002-968d-4e4e-a966-26a2b193bd42" />
<img width="291" height="24" alt="image" src="https://github.com/user-attachments/assets/76badf47-3992-4092-9c68-d906799a528b" />
<img width="520" height="203" alt="image" src="https://github.com/user-attachments/assets/93df8a19-a604-40bb-9f41-8c275e5ffdf8" />
---

