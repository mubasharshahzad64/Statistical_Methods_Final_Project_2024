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
```
### Feature Scaling and Train-Test Split
```
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Scale features and split dataset
scaler = StandardScaler()
X = df.drop(columns=['Age'])
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
### Linear Regression
```
from sklearn.linear_model import Ridge

# Train and evaluate Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
residuals_ridge = y_test - y_pred_ridge

print(f"Ridge Regression\nMSE: {mse_ridge}\nR²: {r2_ridge}")
```
### Ridge Regression 
```
from sklearn.linear_model import Ridge

# Train and evaluate Ridge Regression model
ridge_model = Ridge()
ridge_model.fit(X_train_scaled, y_train)
y_pred_ridge = ridge_model.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)
residuals_ridge = y_test - y_pred_ridge

print(f"Ridge Regression\nMSE: {mse_ridge}\nR²: {r2_ridge}")
```
### Lasso Regression 
```
from sklearn.linear_model import Lasso

# Train and evaluate Lasso Regression model
lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_model.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
residuals_lasso = y_test - y_pred_lasso

print(f"Lasso Regression\nMSE: {mse_lasso}\nR²: {r2_lasso}")
```
### Conclussion 
Based on the Mean Squared Error (MSE) and R² values, Linear Regression and Ridge Regression performed similarly and significantly better than Lasso Regression. Ridge Regression is slightly preferred due to its ability to handle multicollinearity without compromising performance. It achieved the lowest MSE and a high R² value, indicating a good fit and predictive capability.

Residuals analysis revealed that both Linear and Ridge models had similar patterns of residuals, indicating that the models captured the underlying patterns of the data well. However, there were some outliers with high residual values, suggesting that the models might be underestimating or overestimating the age of some abalones. Lasso Regression showed higher variability in residuals, which might indicate that it did not generalize as well to the test data.

In future work, exploring advanced regression techniques and feature engineering may yield further improvements in model performance.

### Acknowledgements
Thanks to the UCI Machine Learning Repository for providing the Abalone dataset and to the readers of this report.

### Author
Muhammad Mubashar Shahzad, University of Trieste

© 2024 Muhammad Mubashar Shahzad, University of








