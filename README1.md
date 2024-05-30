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
## Step 1: Understanding the Data
1. First, let's summarize the dataset and its features:
2. Categorical Feature: Sex (M, F, I)
3. Continuous Features: Length, Diameter, Height, Whole_weight, Shucked_weight, Viscera_weight, Shell_weight
4. Target Variable: Rings (which, when adjusted, gives the age)

## Step 2: Preparing the Environment

Python installed.
Libraries: pandas, numpy, scikit-learn, matplotlib, and seaborn.
```
pip install pandas numpy scikit-learn matplotlib seaborn

```

# Step 3: Load the dataset
Place the CSV file (abalone.csv) in the same directory as my Python script
Import pandas:
pandas is a powerful library for data manipulation and analysis.
```
Read the CSV file:

# Load the dataset into a DataFrame

df = pd.read_csv('abalone.csv')

Inspect the Data

# Display the first few rows
print(df.head())
```
 Verify the Column Names
Ensure that the column names match the ones described in the dataset information. Sometimes, the first row might be treated as header incorrectly if the file format is not as expected.
Load the CSV file into a pandas DataFrame:
```
import pandas as pd
# Load the dataset
df = pd.read_csv('abalone.csv')
# Display the first few rows to ensure it loaded correctly
print(df.head())
# Optionally, inspect the column names and data types
print(df.columns)
print(df.dtypes)
```

# Step 3: Python script
Python script that covers loading the dataset, preprocessing, and training a linear regression model to predict the age of abalones based on their physical measurements.
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
df = pd.read_csv('abalone.csv')

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
import joblib
joblib.dump(model, 'abalone_age_predictor.pkl')
```
Run the script using the Python interpreter
```
python abalone_age_prediction.py
```
##  Results
```
   Sex  Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  Shell_weight  Rings
0    M   0.455     0.365   0.095        0.5140          0.2245          0.1010         0.150     15
1    M   0.350     0.265   0.090        0.2255          0.0995          0.0485         0.070      7
2    F   0.530     0.420   0.135        0.6770          0.2565          0.1415         0.210      9
3    M   0.440     0.365   0.125        0.5160          0.2155          0.1140         0.155     10
4    I   0.330     0.255   0.080        0.2050          0.0895          0.0395         0.055      7
```
