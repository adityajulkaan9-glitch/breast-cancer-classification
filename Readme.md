# Breast Cancer Classification Using Machine Learning

## Project Overview

This project uses Machine Learning to classify breast cancer tumors as **Benign** or **Malignant** using numerical features extracted from breast cancer diagnostic data.

The goal is to build a classification model, evaluate its performance, and understand how changing the prediction threshold affects the detection of malignant cases.

## Problem Statement

Missing a malignant case can be serious, so this project evaluates more than accuracy. It also examines malignant recall, precision, and the confusion matrix to understand the model's predictions and errors.

## Project Goal

- Explore and understand the dataset using Exploratory Data Analysis (EDA).
- Build a simple baseline for comparison.
- Train a Logistic Regression classifier.
- Evaluate the model using classification metrics.
- Investigate threshold tuning to improve malignant case detection.

## Dataset

The dataset contains **569 rows and 32 original columns**.

- **Target:** `diagnosis`
- **Classes:** Benign and Malignant
- **Input features:** 30 numerical features
- **Missing values:** 0
- **Duplicate rows:** 0

The `diagnosis` column is used as the target variable. The `target` column, which duplicates the diagnosis labels, is excluded from the input features to prevent data leakage.

## Technologies Used

- **Python** — Programming language
- **Pandas** — Data handling and exploration
- **NumPy** — Numerical operations
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Data splitting, feature scaling, Logistic Regression, and evaluation metrics

## Machine Learning Workflow

1. Exploratory Data Analysis (EDA)
2. Feature and target separation
3. Train-test split with stratification
4. Feature scaling using `StandardScaler`
5. Majority-class baseline
6. Logistic Regression model training
7. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix
8. Threshold comparison using training data

## Results

The Logistic Regression model was evaluated on a held-out test set of 114 samples.

Using a malignant probability threshold of `0.3`, the model achieved:

- **Accuracy:** 98.25%
- **Malignant precision:** 97.62%
- **Malignant recall:** 97.62%
- **Malignant F1-score:** 97.62%

### Confusion Matrix

| Actual / Predicted | Benign | Malignant |
|---|---:|---:|
| Benign | 71 | 1 |
| Malignant | 1 | 41 |

The model correctly classified 112 out of 114 test samples, with two incorrect predictions: one false positive and one false negative.

## Limitations

- This is an educational machine learning project, not a clinically validated diagnostic system.
- The dataset and test set are relatively small; results may vary on other data.
- Threshold selection involves a trade-off between false negatives and false positives.
- Performance on this dataset does not guarantee performance on real-world patient data.

## Disclaimer

This project is intended for learning and experimentation only. It must not be used to diagnose or make medical decisions about any person.