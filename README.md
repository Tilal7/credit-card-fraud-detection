# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using classification algorithms on highly imbalanced data.

## Overview

Credit card fraud causes billions in losses annually. This project applies machine learning to distinguish fraudulent transactions from legitimate ones, demonstrating practical skills in handling imbalanced datasets, model evaluation, and feature analysis.

## Dataset

The dataset contains European cardholder transactions from September 2013.

| Metric | Value |
|--------|-------|
| Total Transactions | 284,807 |
| Fraud Cases | 492 (0.172%) |
| Features | 30 |
| Imbalance Ratio | 1 to 578 |

**Features** - `Time` (seconds from first transaction), `V1-V28` (PCA-transformed, anonymized), `Amount` (transaction value), `Class` (0 = legitimate, 1 = fraud)

**Source** - [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure

```
credit-card-fraud-detection/
├── data/                   # Dataset (download from Kaggle)
├── notebooks/              # Main analysis notebook
├── figures/                # Generated visualizations
├── models/                 # Trained model files
├── src/                    # Utility scripts
├── requirements.txt
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Tilal7/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows - venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place `creditcard.csv` in the `data/` folder.

## Methodology

**Data Preprocessing** - Scaled Amount and Time features using RobustScaler, applied stratified 80-20 train-test split

**Handling Imbalance** - Used SMOTE (Synthetic Minority Over-sampling) to generate synthetic fraud samples

**Models Trained** - Logistic Regression (baseline), Random Forest (ensemble), XGBoost (gradient boosting)

**Evaluation Metrics** - Precision, Recall, F1-Score, ROC-AUC, Average Precision (standard accuracy is misleading for imbalanced data)

## Results

| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Logistic Regression | 0.85 | 0.62 | 0.72 | 0.97 |
| Random Forest | 0.93 | 0.80 | 0.86 | 0.98 |
| **XGBoost** | **0.95** | **0.82** | **0.88** | **0.98** |

XGBoost achieves the best performance with 88% F1-Score and 98% ROC-AUC.

## Key Findings

**Most Predictive Features** - V17, V14, V12, and V10 showed highest importance for fraud detection

**Transaction Patterns** - Fraudulent transactions tend to have lower amounts on average, and fraud rates vary by time of day

**Model Insights** - Ensemble methods significantly outperform linear models, SMOTE improves recall without severely impacting precision

## Visualizations

The notebook generates comprehensive plots including class distribution, transaction patterns, correlation analysis, ROC curves, precision-recall curves, confusion matrices, and feature importance rankings.

## Future Improvements

Hyperparameter tuning with GridSearchCV, neural network approaches (Autoencoders), cost-sensitive learning, real-time prediction API, model explainability with SHAP values

## Technologies

Python, pandas, NumPy, scikit-learn, XGBoost, imbalanced-learn, matplotlib, seaborn

## License

MIT License

---

**Author** - Tilal7

*Educational project demonstrating machine learning skills for fraud detection in financial transactions.*
