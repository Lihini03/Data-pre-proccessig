# data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


# Step 1: Load Dataset

df = pd.read_csv("../data/titanic_dataset.csv")

print("Original Dataset:")
print(df.head(6))

print("\nDataset Info:")
print(df.info())


# Step 2: Data Cleaning

# Drop irrelevant columns
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)

# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

print("\n After Cleaning (missing values fixed, irrelevant columns dropped):")
print(df.isnull().sum())


# Step 3: Encode Categorical Data

label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
df['Embarked'] = label.fit_transform(df['Embarked'])

print("\n  After Encoding:")
print(df.head())


# Step 4: Feature Scaling

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

print("\n After Scaling (Age & Fare normalized):")
print(df[['Age', 'Fare']].head())


# Step 5: Data Reduction (Optional - PCA)

features = df.drop(columns=['Survived'])  # keep target separate
pca = PCA(n_components=2)
reduced = pca.fit_transform(features)

df_pca = pd.DataFrame(reduced, columns=['PC1','PC2'])
df_pca['Survived'] = df['Survived']

print("\n  PCA Reduced Dataset (2 components):")
print(df_pca.head())


# Step 6: Final Output

print("\n🎯 Final Preprocessed Dataset:")
print(df.head())

# Save processed dataset
df.to_csv("titanic_preprocessed.csv", index=False)
print("\n Preprocessed dataset saved as titanic_preprocessed.csv")
