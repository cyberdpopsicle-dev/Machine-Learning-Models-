# Absolute Machine Unlearning: A Minimalistic Implementation

This project demonstrates a minimal implementation of **absolute machine unlearning**, showing how an independent algorithm can load a pre-trained machine learning model and erase its learned knowledge without access to the original training data or training process.

## Workflow

```mermaid
flowchart LR
    A[Training Data] --> B[Train Base Model]
    B --> C[Save as base_model.pkl]

    C --> D[Blind Unlearning Algorithm]
    E[No Training Data Access] --> D

    D --> F[Weight Shattering]
    F --> G[Model with Absolute Amnesia]
```

## Phase 1: Knowledge Acquisition

```mermaid
flowchart TD
    A[Structured Dataset]
    B[Train Classification Model]
    C[Optimize Weights and Biases]
    D[High Prediction Accuracy]
    E[Serialize Model]
    F[base_model.pkl]

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
```

**Action:** A standard classification model is trained on a structured dataset.

**Result:** The model learns internal representations by optimizing its parameters. The trained model is serialized and saved as a `.pkl` file.

---

## Phase 2: Blind Unlearning

```mermaid
flowchart TD
    A[Load base_model.pkl]
    B[No Access to Training Data]
    C[No Hyperparameters]
    D[No Training History]
    E[Access Internal Weights]
    F[Replace Weights with Gaussian Noise]
    G[Destroy Learned Representations]

    A --> E
    B --> E
    C --> E
    D --> E
    E --> F
    F --> G
```

**Action:** An independent script loads the model without any knowledge of its original training process.

**Mechanism:** The algorithm directly overwrites the model's internal parameters (`coef_` and `intercept_`) with Gaussian random noise.

---

## Phase 3: Verification

```mermaid
flowchart LR
    A[Wiped Model]
    B[Test Dataset]
    C[Evaluate Accuracy]
    D[High Accuracy Before]
    E[Random Baseline Accuracy]
    F[Complete Memory Loss]

    A --> C
    B --> C
    C --> D
    C --> E
    E --> F
```

**Result:** Model performance collapses from a high accuracy level to near-random chance (e.g., ~10% for a 10-class classification task), indicating complete knowledge removal.

---

## Why This Matters

```mermaid
flowchart TD
    A[Absolute Unlearning]
    B[Privacy Compliance]
    C[Right to Be Forgotten]
    D[Targeted Data Removal]
    E[Reduced Retraining Costs]
    F[Lower Data Leakage Risk]

    A --> B
    B --> C
    B --> D
    B --> E
    B --> F
```

Although this project demonstrates complete memory erasure, the underlying principles align with modern AI privacy requirements. Similar approaches motivate machine unlearning systems designed to remove specific user information while avoiding expensive full-model retraining.
