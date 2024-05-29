# Abalone Age Prediction

This project aims to predict the age of abalones based on their physical measurements using a linear regression model. The dataset used contains various physical measurements of abalones, and the goal is to predict the number of rings, which can be used to estimate their age.

## Table of Contents
- [Dataset Information](#dataset-information)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Script](#running-the-script)
- [Understanding the Script](#understanding-the-script)
- [Troubleshooting](#troubleshooting)

## Dataset Information

The dataset contains 4,177 instances of abalones and 8 features. The features include:
- Sex: Categorical (M, F, I)
- Length: Continuous (mm)
- Diameter: Continuous (mm)
- Height: Continuous (mm)
- Whole_weight: Continuous (grams)
- Shucked_weight: Continuous (grams)
- Viscera_weight: Continuous (grams)
- Shell_weight: Continuous (grams)
- Rings: Integer (target variable)

The age of an abalone can be estimated as `Rings + 1.5`.

## Prerequisites

Before running the script, the following installed:
- Python 3.7+
- pandas
- scikit-learn
- joblib

install the necessary packages using pip:
```sh
pip install pandas scikit-learn joblib
```
## Installation
# Download the Dataset:

Download the abalone.csv file from the UCI Machine Learning Repository.
Place the CSV File:

Place the abalone.csv file in the same directory as Python script.
Save the Python Script:import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load the dataset
df = pd.read_csv('abalone.csv')

# Verify the columns and data
print(df.columns)
print(df.head())

# Step 2: Preprocessing
# Convert categorical variable 'Sex' into numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Adjust target variable 'Rings' to get the actual age
df['Age'] = df['Rings'] + 1.5
df.drop(columns=['Rings'], inplace=True)

# Step 3: Split the data into training and testing sets
X = df.drop(columns=['Age'])
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Optionally, save the trained model for future use
joblib.dump(model, 'abalone_age_predictor.pkl')
Running the Script
python abalone_age_prediction.py
Understanding the Script
Loading the Dataset:

The dataset is loaded into a pandas DataFrame.
The columns of the DataFrame are printed to verify they match the expected structure.
Preprocessing:

The 'Sex' column is converted to numerical values using one-hot encoding.
The target variable 'Rings' is adjusted to represent the actual age.
Splitting the Data:

The data is split into training and testing sets.
Training the Model:

## A linear regression model is trained using the training data.
Evaluating the Model:

The model's performance is evaluated using mean squared error and R^2 score.
Saving the Model:

The trained model is saved for future use using joblib.
