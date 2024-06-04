import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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

# Initialize the Lasso Regression model
lasso_model = Lasso(alpha=1.0)

# Train the model on the training data
lasso_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_lasso = lasso_model.predict(X_test)

# Evaluate the model
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("Lasso Regression")
print("Mean Squared Error (MSE):", mse_lasso)
print("R-squared (R2) Score:", r2_lasso)

# Calculate residuals
residuals_lasso = y_test - y_pred_lasso

# Print first few residuals
print("\nFirst 10 Residuals:")
print(residuals_lasso.head(10))

# Print summary statistics of residuals
print("\nResiduals Summary Statistics:")
print(residuals_lasso.describe())

# Plot residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals_lasso, bins=20, kde=True)
plt.title('Distribution of Residuals (Lasso)')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Scatter plot of residuals
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_lasso, residuals_lasso)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Values (Lasso)')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.show()

# Q-Q plot to check normality of residuals
plt.figure(figsize=(10, 6))
stats.probplot(residuals_lasso, dist="norm", plot=plt)
plt.title('Q-Q Plot of Residuals (Lasso)')
plt.show()
