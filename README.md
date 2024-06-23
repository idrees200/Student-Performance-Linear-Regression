# Student Performance Linear Regression

This repository contains Python code for predicting student performance using linear regression on the UCI Student Performance dataset.

## Overview

The script utilizes a custom implementation of linear regression and evaluates the model using various metrics such as R-squared, confusion matrix, sensitivity, specificity, precision, recall, F1 score, mean squared error (MSE), and accuracy.

## Dataset

The dataset used is from the UCI Machine Learning Repository, specifically the Student Performance dataset (ID: 320). It includes features related to students' study habits, health, family relationships, failures, and absences, aiming to predict the final grades (G3).

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Code Structure

1. **Data Loading and Preprocessing**:
   - Loads the dataset using `fetch_ucirepo` from `ucimlrepo` library.
   - Selects relevant features (`studytime`, `health`, `famrel`, `failures`, `absences`) and the target variable (`G3`).
   - Splits the data into training and testing sets.

2. **Custom Linear Regression**:
   - Implements a custom `LinearRegressionCustom` class for linear regression with gradient descent.
   - Fits the model to training data and predicts on test data.
   - Computes R-squared score using manual calculations.

3. **Evaluation Metrics**:
   - Defines `ConfusionMatrix` and `EvaluationMetrics` classes for model evaluation.
   - Plots the confusion matrix and calculates various metrics such as sensitivity, specificity, precision, recall, F1 score, MSE, and accuracy.

4. **Model Training and Evaluation**:
   - Trains the linear regression model multiple times to find the highest accuracy.
   - Saves the model with the highest accuracy to a pickle file.
   - Loads the saved model and evaluates it on test data.

5. **Visualization**:
   - Visualizes the relationship between a selected feature (`failures`) and the actual vs. predicted final grades using scatter plots.

## Usage

1. **Setup Environment**:
   - Install required libraries using `pip install numpy pandas matplotlib seaborn scikit-learn`.

2. **Run the Script**:
   - Execute the script to perform data loading, preprocessing, model training, evaluation, and visualization.
   - Adjust parameters or explore additional features for enhanced analysis.

3. **Output**:
   - The script produces visualizations (scatter plots, histograms, correlation matrix), evaluation metrics, and model performance insights.

## Future Enhancements

- Incorporate cross-validation for robust model evaluation.
- Explore feature engineering techniques to improve prediction accuracy.
- Implement advanced regression techniques (e.g., Ridge, Lasso) for regularization.

![Screenshot 2024-06-23 200416](https://github.com/idrees200/Student-Performance-Linear-Regression/assets/113856749/5c74fcd2-c5fd-42e4-b759-d51034d51dc1)
![Screenshot 2024-06-23 200355](https://github.com/idrees200/Student-Performance-Linear-Regression/assets/113856749/e17766c6-5efe-460b-bce5-555b84f86e36)
![Screenshot 2024-06-23 200348](https://github.com/idrees200/Student-Performance-Linear-Regression/assets/113856749/b6c2607b-0c11-4c70-aaa1-69e8278228cb)
![Screenshot 2024-06-23 200319](https://github.com/idrees200/Student-Performance-Linear-Regression/assets/113856749/3abd6eff-01fc-426f-a6a6-a7b7c8e50719)
![Screenshot 2024-06-23 200422](https://github.com/idrees200/Student-Performance-Linear-Regression/assets/113856749/17aa997b-3fc4-474c-a2ed-1f38e8c1a69c)
