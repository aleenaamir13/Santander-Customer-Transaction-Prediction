# Santander Customer Transaction Prediction

## Overview

This project focuses on predicting whether a customer will make a specific transaction using the Santander Customer Transaction Prediction dataset from Kaggle.

The objective is a binary classification task evaluated using the ROC-AUC metric. The project implements a complete machine learning pipeline including data exploration, feature engineering, model development, ensemble learning, and model interpretability.

---

## Dataset

**Competition:** Santander Customer Transaction Prediction

The dataset contains:

* 200 anonymized numerical features (`var_0` to `var_199`)
* Binary target variable (`target`)
* 200,000 training samples
* 200,000 test samples

### Dataset Characteristics

* High-dimensional tabular data
* Binary classification problem
* Imbalanced classes (approximately 8.95:1 ratio)
* Evaluation Metric: ROC-AUC

---

## Project Pipeline

### Phase 0: Dataset Exploration

Performed exploratory data analysis (EDA) including:

* Missing value analysis
* Class distribution analysis
* Statistical summaries
* Feature distribution visualization

#### Key Findings

* No missing values found
* Target class imbalance ratio: **8.95:1**
* Dataset contains 200 numerical features

---

### Phase 1: Model Development

#### Baseline Model

**Logistic Regression**

* StandardScaler normalization
* Train-validation split
* ROC-AUC Score: **0.8599**

#### Feature Engineering

Created 416 additional features:

##### Count Encoding Features

Frequency-based encoding for all original features.

##### Row Statistical Features

* Mean
* Standard Deviation
* Sum
* Maximum
* Minimum
* Median
* Range
* Skewness
* Positive Ratio
* Negative Ratio
* Absolute Sum

##### Count Statistics

* Count Mean
* Count Standard Deviation
* Count Maximum
* Count Minimum
* Count Sum

##### Inverse Features

Generated inverse representations of all original variables.

**Final Feature Count:** 616

---

## Models Used

### LightGBM

Parameters:

* 2500 estimators
* Learning rate = 0.03
* Max depth = 5
* 5-Fold Stratified Cross Validation

**OOF ROC-AUC:** 0.8943

---

### CatBoost

Parameters:

* 2500 iterations
* Learning rate = 0.05
* Depth = 4
* GPU acceleration
* Early stopping

**OOF ROC-AUC:** 0.8975

---

### Ensemble Model

Weighted averaging using model ROC-AUC scores.

**Final Ensemble ROC-AUC:** 0.8976

---

## Performance Comparison

| Model               | ROC-AUC |
| ------------------- | ------- |
| Logistic Regression | 0.8599  |
| LightGBM            | 0.8943  |
| CatBoost            | 0.8975  |
| Ensemble            | 0.8976  |

### Improvement Over Baseline

**+0.0378 ROC-AUC**

---

## Model Interpretability

### SHAP Analysis

SHAP was used to understand feature contributions and prediction behavior.

### Permutation Importance

Permutation importance was applied to identify the most influential features.

### Most Important Features

1. var_81
2. var_139
3. inv_var_81
4. var_53
5. var_12
6. inv_var_139
7. var_146
8. inv_var_6
9. inv_var_26
10. var_110

### Key Insight

Count-encoded rarity features were found to be highly predictive, indicating that uncommon feature values provide strong signals for identifying positive transaction cases.

---

## Kaggle Results

| Metric          | Score   |
| --------------- | ------- |
| Public ROC-AUC  | 0.89815 |
| Private ROC-AUC | 0.89582 |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* LightGBM
* CatBoost
* SHAP
* Matplotlib
* Seaborn

---

## Reproducibility

### Environment

* Python 3.x
* Random Seed = 42

### Main Libraries

* pandas
* numpy
* scikit-learn
* lightgbm
* catboost
* shap
* matplotlib
* seaborn

