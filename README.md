# Home Loan Data Analysis & Default Prediction

## 📌 Project Overview

This project analyzes home loan applicant data and develops a deep learning model to predict whether an applicant is likely to experience loan repayment difficulty.

The project follows an end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, class balancing, feature scaling, neural network development, training, and model evaluation.

## 🎯 Objectives

- Analyze home loan applicant characteristics.
- Explore patterns and relationships within the loan dataset.
- Handle missing values and prepare the data for modeling.
- Address class imbalance using SMOTE.
- Scale numerical features using StandardScaler.
- Build a neural network for binary classification.
- Evaluate the model using multiple classification metrics.

## 📊 Dataset

The dataset contains home loan applicant information, including financial, demographic, credit-related, and loan-related features.

The original dataset contains **5,789 records and 122 columns**. The `SK_ID_CURR` identifier was removed before modeling.

The target variable is:

- `TARGET = 0` → Payer
- `TARGET = 1` → Defaulter

## 🔍 Exploratory Data Analysis

The project includes exploratory analysis of the loan dataset to understand:

- Target-class distribution
- Applicant characteristics
- Financial variables
- Loan-related variables
- Relationships between features
- Class imbalance

Visualizations were created using **Matplotlib** and **Seaborn**.

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Checked for duplicate records.
2. Checked missing values.
3. Removed the `SK_ID_CURR` identifier.
4. Prepared features (`X`) and target (`y`).
5. Split the data into training and testing sets using stratification.
6. Applied **SMOTE** to balance the training classes.
7. Applied **StandardScaler** to scale the features.

After SMOTE, the training data contained equal numbers of both classes:

- Class 0: 4,273
- Class 1: 4,273

## 🧠 Deep Learning Model

A fully connected **Artificial Neural Network (ANN)** was developed using TensorFlow/Keras.

### Architecture

```text
Input
  ↓
Dense (128 neurons, ReLU)
  ↓
Dense (64 neurons, ReLU)
  ↓
Dense (32 neurons, ReLU)
  ↓
Dense (16 neurons, ReLU)
  ↓
Dropout (0.3)
  ↓
Dense (1 neuron, Sigmoid)
