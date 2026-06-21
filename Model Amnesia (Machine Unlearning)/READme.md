A minimalistic implementation of absolute machine unlearning. This project demonstrates how an independent algorithm can blind-load a pre-trained machine learning model and systematically erase its entire knowledge base without knowing how it was originally trained.

1. The Workflow
[Training Data] ──> (Base Model Trained) ──> Saved as 'base_model.pkl'
                                                    │
[No Data Access] ──> (Unlearning Model) ◄───────────┘
                            │
                    [Weight Shattering]
                            │
                            ▼
               (Model with Absolute Amnesia)
2. Core Stages
Phase 1: Knowledge Acquisition (Base Model)
Action: A standard classification model is trained on a structured dataset.

Result: The model optimizes its internal mathematical weights and coefficients to achieve high prediction accuracy. It is serialized and saved into a .pkl (pickle) file.

Phase 2: Blind Unlearning (The Amnesia Model)
Action: An independent script loads the .pkl file. It has zero access to the original training data, hyper-parameters, or training history.

Mechanism: It directly targets the internal weight tensors (coef_ and intercept_) of the structure and overwrites them with pure Gaussian random noise.

Phase 3: Verification
Action: The wiped model is exposed to test data to evaluate what remains.

Result: Accuracy drops from a high performing percentage directly down to random baseline chance (e.g., ~10% for a 10-class problem). The model is completely reset.

3. Why This Matters
While this project demonstrates Absolute Unlearning (complete memory erasure), the underlying concept forms the bedrock of modern AI compliance. It mimics corporate privacy engineering where models must "forget" specific target data (e.g., GDPR Right to Be Forgotten) without risking data leaks or requiring expensive full-model retraining.
