# PRODIGY_DS_03
Build a decision tree classifier to predict whether a customer will purchase a product or service based on their demographic and behavioral data. Use a dataset such as the Bank Marketing dataset from the UCI Machine Learning Repository.

## Overview
This repository contains code for a machine learning model that classifies bank customers based on certain features. The classification is performed using a Decision Tree Classifier.

## Prerequisites
- Python 3.x
- Required Python packages: numpy, pandas, matplotlib, scikit-learn

## Installation
1. Clone this repository to your local machine.
2. Make sure you have Python 3.x installed
3. Install the required packages using pip: pip install numpy pandas matplotlib scikit-learn


## Usage
1. Make sure that you have the dataset file `bank.csv` in the same directory as the code.
2. Run the provided code in your Python environment.

## Description
- `bank.csv`: This file contains the dataset used for classification.
- `classification.py`: Python script containing the code for the classification task.
- `README.md`: This README file providing information about the repository.

The target variable `deposit` has the following unique values: ['yes', 'no'].We performed similarly for all the columns present in the dataset i.e.'age', 'job', 'marital', 'education', 'default', 'balance', 'housing','loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays','previous', 'poutcome'.

## Instructions
1. Importing necessary libraries: numpy, pandas, matplotlib.pyplot, sklearn.model_selection.train_test_split, sklearn.tree.DecisionTreeClassifier, sklearn.tree.plot_tree, sklearn.metrics.accuracy_score.
2. Load the dataset using pandas `read_csv` function.
3. Explore the dataset using `info`, `shape`, and `columns` functions to understand its structure and contents.
4. Preprocess the data if necessary (e.g., handling missing values, encoding categorical variables).
5. Preprocess the data by dropping the target column (`deposit`), and separating features (`X`) and target (`y`).
6. Perform one-hot encoding on categorical features using `pd.get_dummies`.
7. Split the dataset into training and testing sets.
8. Initialize a Decision Tree Classifier model.
9. Train the classifier using the training data.
10. Make predictions on the testing data.
11. Calculate the accuracy of the model using `accuracy_score`.
12. Print or display the accuracy.
13. Visualize the decision tree using `plot_tree` function from `sklearn.tree`.


