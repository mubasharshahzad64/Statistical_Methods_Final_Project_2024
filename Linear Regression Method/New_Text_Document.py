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
