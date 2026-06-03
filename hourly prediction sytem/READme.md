# Day 1: Smart Grid Energy Consumption Predictor ⚡

An intermediate-level Machine Learning project that leverages **XGBoost** to forecast hourly regional electricity demand. This marks **Day 1** of a 15-day production-grade ML series, with a focus on advanced time-series feature engineering, sequential validation, and energy demand forecasting.

---

## 📈 System Architecture

The pipeline transforms raw hourly electricity consumption data into predictive features and trains a gradient-boosted regression model to capture daily, weekly, and seasonal demand patterns.

```text
[ Raw CSV Data ]
        │
        ▼
[ Temporal Feature Extraction ]
        │
        ▼
[ Lag & Rolling Window Features ]
        │
        ▼
[ Chronological Train/Test Split ]
        │
        ▼
[ XGBoost Regressor ]
        │
        ▼
[ Inference Interface ]
```

---

## 🚀 Getting Started

### 📦 Prerequisites

Install the required dependencies before running the project:

```bash
pip install pandas numpy xgboost scikit-learn
```

---

## 🗃️ Dataset

This project uses historical hourly electricity consumption data from the PJM Interconnection power grid.

**Dataset:** Hourly Energy Consumption (PJM Interconnection Data)
**Author:** Rob Mulla (Kaggle)

### Setup

1. Download the dataset from Kaggle.
2. Select **AEP_hourly.csv** (or any equivalent regional dataset).
3. Place the CSV file in your project's working directory.

---

## 💻 Pipeline Overview

The workflow includes:

* Data ingestion and preprocessing
* Temporal feature extraction
* Lag-based feature engineering
* Rolling window statistics generation
* Chronological train/test splitting
* XGBoost model training and evaluation
* Interactive demand prediction interface

---

## 🛠️ Feature Engineering

| Feature          | Type       | Description                                                                          |
| ---------------- | ---------- | ------------------------------------------------------------------------------------ |
| `hour`           | Temporal   | Captures intraday consumption behavior.                                              |
| `dayofweek`      | Temporal   | Models weekday and weekend demand variations.                                        |
| `lag_24`         | Continuous | Electricity consumption exactly 24 hours prior, providing a daily reference pattern. |
| `rolling_mean_3` | Continuous | Three-hour moving average used to capture short-term demand trends.                  |

---

## 🎯 Validation Strategy

To preserve temporal dependencies and prevent data leakage, the dataset is split **chronologically** rather than randomly.

This approach ensures that the model is evaluated on future observations, providing a realistic assessment of its forecasting performance in production environments.

---

## 📌 Key Learning Outcomes

* Time-series feature engineering
* Lag and rolling-window feature creation
* Chronological model validation
* Energy demand forecasting with XGBoost
* Production-oriented ML pipeline design
