# ==========================================
# BODYM TRAINING MASK DOWNLOAD
# ==========================================

import subprocess
import os


# ==========================================
# 1. CREATE OUTPUT FOLDER
# ==========================================

output_folder = "data/raw/train_masks"

os.makedirs(
    output_folder,
    exist_ok=True
)


# ==========================================
# 2. DOWNLOAD BODYM TRAINING MASKS
# ==========================================

print("Starting BodyM training mask download...")

print("\nSource:")
print("s3://amazon-bodym/train/mask/")

print("\nDestination:")
print(output_folder)

print("\nThis may take some time...")


result = subprocess.run(
    [
        "aws",
        "s3",
        "sync",
        "--no-sign-request",
        "s3://amazon-bodym/train/mask/",
        output_folder
    ]
)


# ==========================================
# 3. CHECK RESULT
# ==========================================

if result.returncode == 0:

    print("\n==========================================")
    print("DOWNLOAD COMPLETED SUCCESSFULLY")
    print("==========================================")

else:

    print("\n==========================================")
    print("DOWNLOAD FAILED")
    print("==========================================")

    print("AWS CLI returned an error.")


# ==========================================
# 4. COUNT DOWNLOADED FILES
# ==========================================

image_count = 0

for file in os.listdir(output_folder):

    if file.endswith(".png"):

        image_count += 1


print("\nPNG images downloaded:", image_count)

print("\n==========================================")
print("MASK DOWNLOAD PROCESS FINISHED")
print("==========================================")