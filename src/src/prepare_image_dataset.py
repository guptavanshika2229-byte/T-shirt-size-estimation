# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# IMAGE DATASET PREPARATION
# ==========================================

import pandas as pd
import os


# ==========================================
# 1. LOAD DATA
# ==========================================

print("Loading BodyM datasets...")

measurements = pd.read_csv(
    "data/raw/measurements.csv"
)

photo_mapping = pd.read_csv(
    "data/raw/subject_to_photo_map.csv"
)

print("Datasets loaded successfully!")


# ==========================================
# 2. DISPLAY DATASET SIZES
# ==========================================

print("\nMeasurements dataset:")
print(measurements.shape)

print("\nPhoto mapping dataset:")
print(photo_mapping.shape)


# ==========================================
# 3. CHECK SUBJECT IDs
# ==========================================

print("\nUnique subjects in measurements:")
print(measurements["subject_id"].nunique())

print("\nUnique subjects in photo mapping:")
print(photo_mapping["subject_id"].nunique())


# ==========================================
# 4. MERGE DATASETS
# ==========================================

print("\nCombining photo information with body measurements...")

merged_data = photo_mapping.merge(
    measurements,
    on="subject_id",
    how="inner"
)


# ==========================================
# 5. DISPLAY RESULT
# ==========================================

print("\nCombined dataset created!")

print("Combined dataset shape:")
print(merged_data.shape)


print("\nCombined dataset columns:")
print(merged_data.columns.tolist())


# ==========================================
# 6. DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows of combined dataset:")

print(
    merged_data.head()
)


# ==========================================
# 7. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")

print(
    merged_data.isnull().sum()
)


# ==========================================
# 8. SAVE COMBINED DATASET
# ==========================================

os.makedirs(
    "data/processed",
    exist_ok=True
)

output_path = (
    "data/processed/image_measurement_dataset.csv"
)

merged_data.to_csv(
    output_path,
    index=False
)


print("\nCombined dataset saved successfully!")

print(
    "Saved at:",
    output_path
)


# ==========================================
# 9. SUMMARY
# ==========================================

print("\n==========================================")
print("IMAGE DATASET PREPARATION COMPLETED")
print("==========================================")