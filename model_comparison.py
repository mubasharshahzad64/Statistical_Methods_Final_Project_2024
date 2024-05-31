import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
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

# Initialize the models
models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=1.0),
    "Ridge Regression": Ridge(alpha=1.0)
}

# Dictionary to store evaluation results
results = {}

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R2': r2}

# Print the results
for name, metrics in results.items():
    print(f"{name}:")
    print(f"Mean Squared Error (MSE): {metrics['MSE']}")
    print(f"R-squared (R2) Score: {metrics['R2']}\n")
