# Facebook Ads Click Prediction

## Project Overview

This project explores whether a user is likely to click on a Facebook advertisement based on two features: **salary** and **time spent on the website**. The work is implemented in a Jupyter Notebook and covers exploratory data analysis, model training, hyperparameter tuning, model comparison, and interpretation.

The goal is to demonstrate an end-to-end binary classification workflow using Python and scikit-learn.

## Dataset

- **Observations:** 499
- **Target:** `Clicked` (`1` = clicked, `0` = did not click)
- **Predictive features:** `Salary`, `Time Spent on Site`
- **Excluded columns:** `Names`, `emails`, and `Country`

The target classes are almost balanced: approximately 50.1% clicked and 49.9% did not click. The notebook's data-quality checks examine missing values, duplicate rows, and dataset structure.

> **Note:** The notebook expects the dataset at `../Facebook_Ads_2.csv`. Place the CSV in the corresponding location or update the path in the data-loading cell.

## Workflow

1. **Data loading and quality checks** — inspect the dataset, missing values, duplicates, and feature types.
2. **Exploratory Data Analysis (EDA)** — examine distributions, relationships between features, and differences between users who clicked and those who did not.
3. **Train/test split** — reserve 30% of the data for final testing, using stratification and a fixed random seed.
4. **Preprocessing pipelines** — apply winsorization and standardization to scale-sensitive models; tree-based models use a separate pipeline.
5. **Baseline model comparison** — evaluate a dummy classifier, Logistic Regression, K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), AdaBoost, and Random Forest.
6. **Hyperparameter tuning** — use `RandomizedSearchCV` with stratified 5-fold cross-validation and ROC-AUC scoring.
7. **Repeated cross-validation and model selection** — compare tuned models with repeated stratified 5-fold cross-validation and apply the one-standard-error rule to balance performance and simplicity.
8. **Final evaluation** — assess the selected model on the held-out test set.
9. **Model interpretation** — review classification reports, confusion matrices, ROC and precision-recall curves, metric comparisons, and permutation feature importance.

## Models Compared

- Dummy Classifier (baseline)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- AdaBoost
- Random Forest
- Stacking Classifier

## Key Results

The notebook selected **Logistic Regression** using the one-standard-error rule. It performed very similarly to the best-performing SVC while remaining simpler to interpret.

| Metric | Result |
|---|---:|
| Mean cross-validation ROC-AUC | 0.9630 |
| Test ROC-AUC | 0.9637 |
| 95% bootstrap confidence interval for test ROC-AUC | 0.9300–0.9920 |
| Test accuracy | Approximately 94.7% |

Permutation importance identified `Salary` as the most influential feature, followed by `Time Spent on Site`. These values describe predictive associations in this dataset and **do not establish causation**.

Results are taken from the notebook's recorded analysis. Re-running the notebook may produce slightly different values if the dataset, software versions, or execution environment differ.

## Tools and Libraries

- Python
- Jupyter Notebook
- pandas and NumPy
- Matplotlib and Seaborn
- SciPy
- scikit-learn
- Feature-engine

## How to Run

1. Clone or download this repository.
2. Ensure Python is installed.
3. Install the required libraries:

   ```bash
   pip install pandas numpy matplotlib seaborn scipy scikit-learn feature-engine jupyter
   ```

4. Place `Facebook_Ads_2.csv` at the path expected by the notebook, or update the CSV path in the data-loading cell.
5. Open and run `Facebook_Ads(3).ipynb` in Jupyter Notebook or JupyterLab.

## Limitations

- The dataset is small (499 observations).
- The model uses only two predictive features.
- Strong results on this dataset may not generalize to new campaigns, audiences, or time periods.
- The model should be validated on a larger, more recent, representative dataset before being used for real campaign decisions.
- Business decisions should account for the relative costs of false positives and false negatives, as well as the actual click rate in production.

## Project Purpose

This project is a portfolio example demonstrating exploratory data analysis, classification, cross-validation, hyperparameter optimization, model evaluation, and feature-importance analysis.
