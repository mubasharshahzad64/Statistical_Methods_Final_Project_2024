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
