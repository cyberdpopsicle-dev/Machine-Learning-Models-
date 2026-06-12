# Day 4: Deep Text Generation Engine via GRU Recurrent Networks 📝

An advanced-level Natural Language Processing (NLP) framework implementing a character-level **Gated Recurrent Unit (GRU)** network. This engine processes arbitrary structural text arrays, maps latent textual transitions, and outputs generative sequence strings through optimized temperature-scaled multinomial sampling.

This project forms **Day 4** of a 15-day production-grade ML series.

---

## 📈 Recurrent Sequence Flow

The model maps input character indices into high-dimensional dense vector spaces, feeds them sequentially through gated recurrent hidden units to track long-range string structures, and maps outputs back to character vocabulary probability distributions.

```text
[ Input Token Index ]
          │
          ▼
[ Embedding Matrix ]
          │
          ▼
[ GRU Hidden State Layer ]
          │
          ▼
[ Linear Softmax Logits ]
          │
          ▼
[ Temperature Sampling ]
          │
          ▼
[ Predicted Next Index ]
```

---

## 🚀 Getting Started

### 📦 Prerequisites

Ensure you have the required deep learning dependencies configured:

```bash
pip install torch numpy
```

---

## 🗃️ Dataset Acquisition

To test the creative capacity of the generator, you will need a text corpus.

### Dataset Options

- **The Complete Works of William Shakespeare**
- Any alternative plain-text literary corpus

### Source

Text datasets can be obtained from public repositories such as:

- Project Gutenberg
- Open-source text archives

### Setup

Save the raw text file as:

```text
input.txt
```

Place it in your active project directory before training the model.

---

## 🎯 Features

- Character-level text generation
- GRU-based recurrent architecture
- Learned character embeddings
- Temperature-controlled sampling
- Custom text corpus support
- PyTorch implementation

---

## 🧠 Model Architecture

1. Character → Index Encoding
2. Embedding Layer
3. GRU Recurrent Network
4. Linear Output Projection
5. Softmax Probability Distribution
6. Temperature-Based Sampling
7. Generated Text Output

---

## 📄 License

This project is part of a 15-day machine learning engineering series and is intended for educational and research purposes.
