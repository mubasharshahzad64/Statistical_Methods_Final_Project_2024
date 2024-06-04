# Abalone Age Prediction

This project aims to predict the age of abalones using physical measurements through various regression models. The dataset used for this analysis is the Abalone dataset from the UCI Machine Learning Repository.

## Dataset Characteristics
- **Type**: Tabular
- **Subject Area**: Biology
- **Associated Tasks**: Classification, Regression
- **Feature Type**: Categorical, Integer, Real
- **Instances**: 4177
- **Features**: 8

## Dataset Information
The age of abalones is determined by counting the number of rings in their shells. This dataset includes physical measurements that can be used to predict age, which is a tedious and time-consuming task if done manually.

### Features
- **Sex**: Categorical (M, F, I)
- **Length**: Continuous (mm) - Longest shell measurement
- **Diameter**: Continuous (mm) - Perpendicular to length
- **Height**: Continuous (mm) - With meat in shell
- **Whole_weight**: Continuous (grams) - Whole abalone
- **Shucked_weight**: Continuous (grams) - Weight of meat
- **Viscera_weight**: Continuous (grams) - Gut weight (after bleeding)
- **Shell_weight**: Continuous (grams) - After being dried
- **Rings**: Integer - Number of rings (+1.5 gives the age in years)

### Missing Values
There are no missing values in the dataset.

## Project Overview
The goal of this project is to build regression models to predict the age of abalones using the provided physical measurements and to compare their performance. The models used include:
- Linear Regression
- Ridge Regression
- Lasso Regression

### Steps
1. **Data Loading and Preprocessing**
   - Load the data and preprocess it by encoding categorical variables and scaling continuous variables.
   - Add a new column for age by adding 1.5 to the rings.

2. **Exploratory Data Analysis (EDA)**
   - Perform graphical and statistical analyses to understand data distributions and relationships.

3. **Feature Scaling and Train-Test Split**
   - Scale the features and split the dataset into training and testing sets.

4. **Model Training and Evaluation**
   - Train Linear Regression, Ridge Regression, and Lasso Regression models.
   - Evaluate models using Mean Squared Error (MSE) and R² metrics.

5. **Results and Model Comparison**
   - Compare the models based on their performance metrics.

6. **Residuals Analysis**
   - Analyze residuals to check for patterns and model accuracy.

7. **Conclusion**
   - Summarize the findings and suggest improvements.

## Results
### Model Performance
- **Linear Regression**
  - MSE: 4.8912
  - R²: 0.5482

- **Ridge Regression**
  - MSE: 4.8911
  - R²: 0.5482

- **Lasso Regression**
  - MSE: 7.6826
  - R²: 0.2903

### Residuals Analysis
#### Comparison of Residuals
| Model               | First 10 Residuals                      | Mean    | Std   | Min   | 25%   | 50%   | 75%   | Max   |
|---------------------|-----------------------------------------|---------|-------|-------|-------|-------|-------|-------|
| Linear Regression   | -2.76, -2.24, 1.99, -2.99, 2.83, 0.76, -2.41, -3.14, -0.19, -0.80 | -0.009 | 2.21  | -6.01 | -1.38 | -0.35 | 0.82  | 9.78  |
| Lasso Regression    | -1.56, -1.98, 5.36, -1.96, 4.23, 1.45, -2.58, -3.44, -1.64, -0.26 | -0.023 | 2.77  | -5.24 | -1.74 | -0.65 | 0.84  | 12.18 |
| Ridge Regression    | -2.76, -2.25, 2.02, -2.99, 2.84, 0.77, -2.42, -3.13, -0.18, -0.80 | -0.008 | 2.21  | -6.02 | -1.38 | -0.35 | 0.83  | 9.77  |

#### Residuals Visualization
- Boxplot of residuals for ridge regression
![Residuals for Ridge Regression](residual_for_ridge.png)

- Residual distribution for lasso regression
![Residual Distribution for Lasso Regression](residual_destribution_for_lasso_regression.png)

- Residual vs predicted values for ridge regression
![Residual vs Predicted Values for Ridge](Residual_vs_predicted_values_for_ridge.png)

- Residual vs predicted values for lasso regression
![Residual vs Predicted Values for Lasso](residual_vs_predicted_value_for_lasso.png)

- Q-Q plot of residuals for ridge regression
![Q-Q Plot of Residuals for Ridge](q-q_plot_of_residual.png)

- Q-Q plot of residuals for lasso regression
![Q-Q Plot of Residuals for Lasso](q-q_lasso.png)

## Python Scripts

### Data Loading and Preprocessing
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('abalone.csv')
df.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

# Preprocess the data
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
df['Age'] = df['Rings'] + 1.5
df.drop(columns=['Rings'], inplace=True)
