# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# Size Prediction
# ==========================================

import pandas as pd
import joblib


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

print("Loading trained model...")

model = joblib.load("models/tshirt_size_model.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")
features = joblib.load("models/features.pkl")

print("Model loaded successfully!")


# ==========================================
# 2. GET BODY MEASUREMENTS FROM USER
# ==========================================

print("\nEnter the following body measurements in centimeters:")

ankle = float(input("Ankle girth: "))
arm_length = float(input("Arm length: "))
bicep = float(input("Bicep girth: "))
calf = float(input("Calf girth: "))
forearm = float(input("Forearm girth: "))
height = float(input("Height: "))
hip = float(input("Hip girth: "))
leg_length = float(input("Leg length: "))
shoulder_breadth = float(input("Shoulder breadth: "))
shoulder_to_crotch = float(input("Shoulder-to-crotch length: "))
thigh = float(input("Thigh girth: "))
waist = float(input("Waist girth: "))
wrist = float(input("Wrist girth: "))


# ==========================================
# 3. CREATE INPUT DATAFRAME
# ==========================================

input_data = pd.DataFrame(
    [[
        ankle,
        arm_length,
        bicep,
        calf,
        forearm,
        height,
        hip,
        leg_length,
        shoulder_breadth,
        shoulder_to_crotch,
        thigh,
        waist,
        wrist
    ]],
    columns=features
)


# ==========================================
# 4. MAKE PREDICTION
# ==========================================

prediction = model.predict(input_data)


# ==========================================
# 5. CONVERT NUMERIC OUTPUT TO SIZE
# ==========================================

predicted_size = label_encoder.inverse_transform(prediction)


# ==========================================
# 6. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("T-SHIRT SIZE PREDICTION")
print("==========================================")

print("Predicted T-shirt size:", predicted_size[0])

print("==========================================")