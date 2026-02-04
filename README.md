# ⛏️ MineToday Competition: Student Performance Prediction

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📊 Project Overview
This repository contains the solution developed for the **MineToday Data Mining Competition**. The objective was to analyze student performance data and build a predictive model to classify students into **"Lulus" (Pass)** or **"Tidak" (Fail)** categories.

What makes this project unique is the **Semi-Supervised approach**: Since the raw data lacked explicit "Pass/Fail" labels, I first used **Clustering (Gaussian Mixture Models)** to generate ground-truth labels based on score distributions, and then trained an **XGBoost Classifier** to predict these outcomes.

## 🎯 Problem Statement
* **Objective:** Predict student graduation status based on pretest scores and attendance.
* **Input Features:** `pretest_py` (Python), `pretest_ml` (Machine Learning), `pretest_st` (Statistics), and `attendance`.
* **Challenge:** The dataset was unlabeled, requiring an unsupervised approach to define success criteria before modeling.

## 💡 Solution Approach & Methodology
I implemented a robust **Unsupervised-to-Supervised** pipeline:

1.  **Data Preprocessing & Cleaning:**
    * Handled duplicates and missing values.
    * Feature Engineering using a custom `FeatureEngineer` class.
    * Standardization using `StandardScaler`.

2.  **Label Generation (The "Teacher" Model):**
    * Used **Gaussian Mixture Models (GMM)** and **K-Means** to group students into clusters.
    * Analyzed the mean scores of each cluster to identify the high-performing group.
    * Automatically assigned labels: Cluster with higher average scores $\rightarrow$ 'Lulus'.

3.  **Classification (The "Student" Model):**
    * Trained an **XGBoost Classifier** on the newly labeled data.
    * Used **Stratified K-Fold Cross Validation** (5 Folds) to ensure model stability.

## 📈 Results & Performance
The model demonstrated exceptional ability to learn the decision boundaries defined by the clustering logic.

* **Algorithm:** XGBoost Classifier
* **Validation Method:** 5-Fold Stratified Cross-Validation
* **Average Accuracy:** **~99.0%**
* **F1-Score:** **0.99**

| Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | 99% | 100% | 99% | 98% | 100% |

## 🛠️ Tools & Libraries
* **Core:** `Pandas`, `NumPy`
* **Clustering:** `Scikit-Learn` (KMeans, GaussianMixture)
* **Classification:** `XGBoost`
* **Visualization:** `Matplotlib`, `Seaborn`, `Plotly`

## 🚀 How to Run
1.  Clone this repository.
2.  Install requirements:
    ```bash
    pip install pandas scikit-learn xgboost matplotlib seaborn
    ```
3.  Run the Jupyter Notebook:
    ```bash
    jupyter notebook "MineToday_Analysis.ipynb"
    ```

---
**Author:** Yosuke Yung
*CS Student @ BINUS UNIVERSITY*
