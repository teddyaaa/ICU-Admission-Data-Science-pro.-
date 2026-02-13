# ICU-Admission
# Predicting COVID-19 ICU Admissions

A machine learning project to predict which hospitalized COVID-19 patients are likely to require ICU admission, using a real-world dataset from a Brazilian hospital.

---

## Project Overview

This repository contains code and documentation for building and evaluating machine learning models to forecast ICU admissions among confirmed COVID-19 patients in Brazil. The goal is to support proactive clinical decision-making and help hospitals manage limited ICU capacity more effectively.

---

## Dataset

- **Source**: Public COVID-19 ICU admission dataset released on Kaggle by a top-tier hospital in Brazil  
- **Scope**: Hospitalized COVID-19 patients with repeated measurements per patient  
- **Key variables**:
  - Demographics: Age (with emphasis on age above 65), gender
  - Clinical history: Comorbidities, disease groupings, hypertension (HTN), immunocompromised status
  - Laboratory metrics: Albumin, blood gas and chemistry features, blood cell counts, gas exchange metrics (e.g., P02, PCO2), lactate
  - **Target**: ICU admission (binary: 0 = no ICU, 1 = ICU)

**Example descriptive facts**:
- 385 unique patients, each represented by 5 time-window rows
- Total admissions: 1,410 non-ICU and 515 ICU stays
- Nearly half of patients are over 65 years old, with a roughly balanced gender distribution

> **Note**: Include a link to the Kaggle dataset in this section once you add it to the repo.

---

## Problem Statement

Brazil was among the countries most severely affected by the COVID-19 pandemic, with millions of confirmed cases and severe strain on ICU capacity. This project aims to build a binary classification model that predicts whether a hospitalized COVID-19 patient will require ICU admission, enabling:

- Early risk stratification at admission
- Better ICU bed planning and resource allocation
- Improved patient outcomes through timely intervention

---

## Methods

### 1. Exploratory Data Analysis (EDA)

**Key EDA steps**:

- Data inspection: `df.info()`, `df.describe()`, and missing value analysis with `df.isnull().sum()`
- Target distribution: ICU vs non-ICU admission counts to assess class balance (1,410 vs 515)
- Correlation analysis: Heatmaps to identify clusters of correlated lab features (e.g., blood gas metrics) and unique, less correlated predictors
- Feature distributions: Histograms, box plots, and bar charts for demographic and clinical features

**Important observations**:

- Strong correlation clusters among blood gas and oxygenation variables, suggesting redundancy
- Some features show low correlation with others and may carry unique predictive information
- A significantly higher proportion of patients did not require ICU admission compared to those who did

### 2. Data Preprocessing

**Preprocessing pipeline**:

- **Missing values**: Imputed using mean or median strategies, depending on feature type and distribution
- **Categorical encoding**:
  - One-hot encoding for categorical variables such as gender and time-window indicators
  - Label encoding for any ordinal features if needed
- **Feature scaling**: MinMaxScaler applied to numerical variables to normalize them into a fixed range (e.g., between -1 and 1)
- **Train-test split**: 80/20 split using `train_test_split` on patient-level samples

### 3. Modeling

**Models implemented**:

- Random Forest Classifier
- Support Vector Machine (SVC)
- Logistic Regression

**Initial and post-preprocessing performance (accuracy on held-out test set)**:

| Model               | Accuracy (initial / tuned) |
|---------------------|----------------------------|
| Random Forest       | 93% initial, ~86% after full preprocessing and tuning |
| Logistic Regression | ~85% after preprocessing |
| SVM (SVC)           | ~72% after preprocessing |

**Evaluation metrics**:
- Accuracy
- Precision
- Recall
- F1-score

In addition, feature importance from the Random Forest model was used to identify the most influential predictors associated with ICU admission.

---

## Key Findings

- Random Forest achieved the best overall performance and generalization, making it the primary model for this task
- ICU admission risk is meaningfully associated with age (especially above 65), certain disease groupings, and key laboratory and gas exchange metrics
- The target is imbalanced (more non-ICU than ICU cases), which should be considered in further work via techniques such as class weighting or resampling

---

---

## How to Run

1. **Clone the repository**:
```bash
git clone https://github.com/<your-username>/covid-icu-admissions.git
cd covid-icu-admissions
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements.txt
python src/train_models.py

## Repository Structure

