# Predicting-Credit-Risk-Using-Logistic-Regression-XGBoost-LightGBM
A machine learning project for predicting credit risk using Logistic Regression, XGBoost, and LightGBM, featuring data preprocessing, feature engineering, class imbalance handling, and comprehensive model evaluation.
📊 Credit Risk Prediction Using Machine Learning

📌 Overview

This repository presents a machine learning–based approach to credit risk prediction, aiming to classify loan applicants as good or bad credit risks. The project compares traditional and advanced models, including Logistic Regression, XGBoost, and LightGBM, to evaluate their effectiveness in handling complex and imbalanced financial data.

The work forms part of an MSc Data Science final project and demonstrates the practical application of ensemble learning techniques in financial risk assessment.

🎯 Objectives

Build predictive models for credit risk classification

Compare traditional and ensemble machine learning algorithms

Apply feature engineering to enhance predictive performance

Handle class imbalance using resampling techniques

Evaluate models using industry-relevant metrics

📂 Dataset

Source: German Credit Dataset

Type: Financial and demographic data

Target Variable: Credit Risk (Good / Bad)

Key Features

Age

Credit amount

Loan duration

Housing status

Savings and checking account information

⚙️ Methodology

Data Preprocessing
Handling missing values

Encoding categorical variables

Feature scaling

Addressing class imbalance using SMOTE

Feature Engineering
Credit-to-income ratio

Credit per month

Age-based groupings

Risk-based interaction features

Models Implemented
Logistic Regression – baseline and interpretable model

XGBoost – gradient boosting with regularization

LightGBM – efficient histogram-based gradient boosting

📈 Evaluation Metrics

Models were assessed using:

Accuracy

Precision

Recall

F1-score

ROC-AUC

Confusion Matrix

Threshold Optimization

These metrics provide a balanced evaluation, especially for imbalanced credit risk data.

🏆 Final Results

After training and evaluating multiple machine learning models, the LightGBM model was selected as the final model based on overall performance and stability.

✅ Final Saved Model

Model: LightGBM

Accuracy: 72.00%

F1-Score: 51.72%

ROC-AUC: 67.99%

The trained model, preprocessing pipeline, and evaluation metrics were saved for reproducibility and future use.

📊 Model Comparison
(Ranked by ROC-AUC) 
Model Accuracy Precision Recall F1-Score ROC-AUC 
LightGBM 0.670 0.4423 0.3833 0.4107 0.6645
Random Forest 0.690 0.4821 0.4500 0.4655 0.6605 
Gradient Boosting 0.685 0.4681 0.3667 0.4112 0.6565 
Logistic Regression 0.650 0.4375 0.5833 0.5000 0.6549 
XGBoost 0.660 0.4333 0.4333 0.4333 0.6512

🔍 Feature Importance

Key features influencing predictions include:

Credit amount (transformed)

Loan duration

Housing risk

Credit-to-income ratio

These results highlight the importance of financial behavior and loan structure in credit risk assessment.

🧠 Key Insights

Ensemble models capture nonlinear relationships more effectively

Feature engineering significantly improves model performance

Threshold optimization enhances minority-class detection

SMOTE helps mitigate class imbalance issues

🔮 Future Work

Hyperparameter optimization using GridSearchCV

Exploring CatBoost for categorical feature handling

Incorporating SHAP for improved model interpretability

Testing the models on additional financial datasets

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

XGBoost

LightGBM

Matplotlib, Seaborn
