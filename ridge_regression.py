import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']
df = pd.read_csv('D:/Statistical_Methods_Final_Project_2024/Statistical_Methods_Final_Project_2024/abalone.csv', names=column_names)

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

# Initialize the Ridge regression model
ridge_model = Ridge(alpha=1.0)

# Train the model on the training data
ridge_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = ridge_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression")
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)
