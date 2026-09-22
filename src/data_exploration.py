import pandas as pd


# Load the BodyM measurements dataset
df = pd.read_csv("data/raw/measurements.csv")


# Display basic information about the dataset
print("Dataset loaded successfully!")
print()

print("Number of rows and columns:")
print(df.shape)

print()

print("Column names:")
print(df.columns.tolist())

print()

print("First 5 rows of the dataset:")
print(df.head())

print()

print("Dataset information:")
df.info()

print()

print("Missing values in each column:")
print(df.isnull().sum())
