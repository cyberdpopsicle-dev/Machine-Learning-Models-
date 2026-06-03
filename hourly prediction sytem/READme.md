# Day 1: Smart Grid Energy Consumption Predictor ⚡

An intermediate-level Machine Learning pipeline using **XGBoost** to forecast hourly regional electricity consumption. This project forms **Day 1** of a 15-day production-grade ML series, focusing on advanced time-series feature engineering and sequential validation.

---

## 📈 System Architecture

The pipeline processes raw hourly load signatures, constructs temporal matrices using historical lags and window metrics, and utilizes gradient-boosted decision trees to map complex intra-day and seasonal demand profiles.
[ Raw CSV Data ] ──> [ Temporal Extraction ] ──> [ Lag & Rolling Features ]│[ Inference Interface ] <── [ XGBoost Regressor ] <── [ Chronological Split ]
---

## 🚀 Getting Started

### 📦 Prerequisites
Ensure you have the required dependencies installed before running the pipeline:
```bash
pip install pandas numpy xgboost scikit-learn
🗃️ Dataset AcquisitionThe model relies on historical hourly electricity load infrastructure data:  Dataset: Hourly Energy Consumption (PJM Interconnection Data)  Author/Credits: Curated by Rob Mulla on Kaggle  Setup: Download AEP_hourly.csv (or any equivalent regional file) and place it in your working directory.  💻 Code ImplementationThis deployment contains the foundational dataset ingestion, sequential feature processing, validation, and an interactive prediction routine.
🛠️ Pipeline FormulationFeatureTypeOperational Metric / Descriptionhour / dayofweekCyclical DiscreteCaptures intraday load configurations & weekend load drops.  lag_24ContinuousHistorical baseline representing load signatures exactly 1 day prior[cite: 1].rolling_mean_3ContinuousMoving window tracking immediate short-term demand trends[cite: 1].🎯 Verification FrameworkTo prevent data leakage, validation splits are enforced chronologically rather than randomly[cite: 1]. This maintains sequential dependency, verifying the model's true extrapolation capacity on future timelines[cite: 1].
