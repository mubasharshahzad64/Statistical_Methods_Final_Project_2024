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
import pandas as pd
# Load the dataset
df = pd.read_csv('abalone.csv')
# Display the first few rows to ensure it loaded correctly
print(df.head())
# Optionally, inspect the column names and data types
print(df.columns)
print(df.dtypes)
```

# Step 4: Python script
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
## Step 5: Data Preprocessing
Convert the Sex Column to Numerical Values:
Use one-hot encoding to convert the categorical `Sex` column into numerical columns.

Adjust the Target Variable:
The `Rings` column needs to be adjusted to get the actual age.
# Complete Script with Adjustments

```
import pandas as pd

# Define column names
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

# Load the dataset with the correct column names
df = pd.read_csv('abalone.csv', names=column_names)

# Display the first few rows to ensure it loaded correctly
print("Initial DataFrame with Correct Column Names:")
print(df.head())

# Perform one-hot encoding on the 'Sex' column
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert boolean columns to integers
df['Sex_I'] = df['Sex_I'].astype(int)
df['Sex_M'] = df['Sex_M'].astype(int)

# Display the first few rows to ensure the conversion
print("\nDataFrame with Sex Columns as Integers:")
print(df.head())

# Adjust the target variable to get the actual age
df['Age'] = df['Rings'] + 1.5

# Drop the original 'Rings' column
df.drop(columns=['Rings'], inplace=True)

# Display the first few rows after adding the Age column
print("\nDataFrame with Age Column:")
print(df.head())
```
## Output
Load the dataset with proper headers, perform one-hot encoding on the `Sex` column, convert the boolean columns to integers, adjust the target variable to `Age`, and display the transformed DataFrame.
```
Initial DataFrame with Correct Column Names:
  Sex  Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  Shell_weight  Rings
0   M   0.455     0.365   0.095        0.5140          0.2245          0.1010         0.150     15
1   M   0.350     0.265   0.090        0.2255          0.0995          0.0485         0.070      7
2   F   0.530     0.420   0.135        0.6770          0.2565          0.1415         0.210      9
3   M   0.440     0.365   0.125        0.5160          0.2155          0.1140         0.155     10
4   I   0.330     0.255   0.080        0.2050          0.0895          0.0395         0.055      7

DataFrame after One-Hot Encoding:
   Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  Shell_weight  Rings  Sex_I  Sex_M
0   0.455     0.365   0.095        0.5140          0.2245          0.1010         0.150     15  False   True
1   0.350     0.265   0.090        0.2255          0.0995          0.0485         0.070      7  False   True
2   0.530     0.420   0.135        0.6770          0.2565          0.1415         0.210      9  False  False
3   0.440     0.365   0.125        0.5160          0.2155          0.1140         0.155     10  False   True
4   0.330     0.255   0.080        0.2050          0.0895          0.0395         0.055      7   True  False

DataFrame with Sex Columns as Integers:
   Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  Shell_weight  Rings  Sex_I  Sex_M
0   0.455     0.365   0.095        0.5140          0.2245          0.1010         0.150     15      0      1
1   0.350     0.265   0.090        0.2255          0.0995          0.0485         0.070      7      0      1
2   0.530     0.420   0.135        0.6770          0.2565          0.1415         0.210      9      0      0
3   0.440     0.365   0.125        0.5160          0.2155          0.1140         0.155     10      0      1
4   0.330     0.255   0.080        0.2050          0.0895          0.0395         0.055      7      1      0

DataFrame with Age Column:
   Length  Diameter  Height  Whole_weight  Shucked_weight  Viscera_weight  Shell_weight  Sex_I  Sex_M   Age
0   0.455     0.365   0.095        0.5140          0.2245          0.1010         0.150      0      1  16.5
1   0.350     0.265   0.090        0.2255          0.0995          0.0485         0.070      0      1   8.5
2   0.530     0.420   0.135        0.6770          0.2565          0.1415         0.210      0      0  10.5
3   0.440     0.365   0.125        0.5160          0.2155          0.1140         0.155      0      1  11.5
4   0.330     0.255   0.080        0.2050          0.0895          0.0395         0.055      1      0   8.5
```
# Step 6: Exploratory Data Analysis (EDA)
Xploratory Data Analysis (EDA) is a critical step in the data analysis process, helping to understand the data's structure, patterns, and relationships.
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv('abalone.csv', names=column_names)

# Convert 'Sex' column to numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert boolean columns to integers
df['Sex_I'] = df['Sex_I'].astype(int)
df['Sex_M'] = df['Sex_M'].astype(int)

# Adjust the target variable to get the actual age
df['Age'] = df['Rings'] + 1.5
df.drop(columns=['Rings'], inplace=True)

# EDA: Visualize Data Distribution
# Distribution of Age
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Boxplot for numerical features
plt.figure(figsize=(14, 8))
df.boxplot()
plt.xticks(rotation=45)
plt.title('Boxplot of Numerical Features')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize relationships between features
sns.pairplot(df)
plt.show()

# Analysis of individual features
# Distribution of Length
plt.figure(figsize=(10, 6))
sns.histplot(df['Length'], bins=20, kde=True)
plt.title('Distribution of Length')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.show()

# Distribution of Diameter
plt.figure(figsize=(10, 6))
sns.histplot(df['Diameter'], bins=20, kde=True)
plt.title('Distribution of Diameter')
plt.xlabel('Diameter')
plt.ylabel('Frequency')
plt.show()

# Relationship between Length and Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Length'], y=df['Age'])
plt.title('Length vs Age')
plt.xlabel('Length')
plt.ylabel('Age')
plt.show()

# Relationship between Whole_weight and Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Whole_weight'], y=df['Age'])
plt.title('Whole Weight vs Age')
plt.xlabel('Whole Weight')
plt.ylabel('Age')
plt.show()
```
## Step 6: Choose Linear Regression Model
proceed with model training using a linear regression model.
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv('abalone.csv', names=column_names)

# Convert 'Sex' column to numerical values using one-hot encoding
df = pd.get_dummies(df, columns=['Sex'], drop_first=True)

# Convert boolean columns to integers
df['Sex_I'] = df['Sex_I'].astype(int)
df['Sex_M'] = df['Sex_M'].astype(int)

# Adjust the target variable to get the actual age
df['Age'] = df['Rings'] + 1.5
df.drop(columns=['Rings'], inplace=True)

# Separate features and target variable
X = df.drop(columns=['Age'])  # Features
y = df['Age']  # Target variable

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
```
# Result
```
Mean Squared Error (MSE): 4.8912
R-squared (R2) Score: 0.5482
```
A lower MSE indicates better predictive accuracy, while a higher R-squared score indicates a better fit of the model to the data.



