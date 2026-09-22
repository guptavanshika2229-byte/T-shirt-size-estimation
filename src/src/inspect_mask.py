# ==========================================
# BODYM SILHOUETTE IMAGE INSPECTION
# ==========================================

import cv2


# Load the silhouette image
image = cv2.imread("data/raw/test_mask.png")


# Check whether image was loaded
if image is None:
    print("❌ Could not load image")
    exit()


print("✅ Image loaded successfully")

print("Image dimensions:")
print("Width :", image.shape[1])
print("Height:", image.shape[0])

print("Number of channels:", image.shape[2])


# Display the image
cv2.imshow("BodyM Silhouette", image)


print("\nPress Q to close the image window.")


while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()