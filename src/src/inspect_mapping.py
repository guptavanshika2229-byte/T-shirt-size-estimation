# ==========================================
# BODYM DATASET MAPPING INSPECTION
# ==========================================

import pandas as pd


# ==========================================
# 1. LOAD PHOTO MAPPING DATA
# ==========================================

print("Loading BodyM photo mapping...")

df = pd.read_csv("data/raw/subject_to_photo_map.csv")


print("\nDataset loaded successfully!")


# ==========================================
# 2. DISPLAY BASIC INFORMATION
# ==========================================

print("\nNumber of rows and columns:")
print(df.shape)


print("\nColumn names:")
print(df.columns.tolist())


# ==========================================
# 3. DISPLAY FIRST 10 ROWS
# ==========================================

print("\nFirst 10 rows:")
print(df.head(10))


# ==========================================
# 4. DATASET INFORMATION
# ==========================================

print("\nDataset information:")
df.info()


# ==========================================
# 5. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())