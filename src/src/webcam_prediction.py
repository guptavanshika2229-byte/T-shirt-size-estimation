# ==========================================
# T-SHIRT SIZE ESTIMATION PROJECT
# WEBCAM PREDICTION WITH PERSON SEGMENTATION
# ==========================================

import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp


# ==========================================
# 1. LOAD MODELS
# ==========================================

print("Loading models...")

measurement_model = joblib.load(
    "models/body_measurement_model.pkl"
)

measurement_features = joblib.load(
    "models/body_measurement_features.pkl"
)

size_model = joblib.load(
    "models/tshirt_size_model.pkl"
)

size_features = joblib.load(
    "models/features.pkl"
)

label_encoder = joblib.load(
    "models/label_encoder.pkl"
)

print("Models loaded successfully!")


# ==========================================
# 2. MEDIAPIPE PERSON SEGMENTATION
# ==========================================

mp_selfie_segmentation = (
    mp.solutions.selfie_segmentation
)

segmenter = (
    mp_selfie_segmentation.SelfieSegmentation(
        model_selection=1
    )
)


# ==========================================
# 3. EXTRACT SILHOUETTE FEATURES
# ==========================================

def extract_features(mask):

    body_pixels = np.column_stack(
        np.where(mask > 0)
    )

    if len(body_pixels) == 0:
        return None

    y = body_pixels[:, 0]
    x = body_pixels[:, 1]

    min_y = y.min()
    max_y = y.max()

    min_x = x.min()
    max_x = x.max()

    body_height = (
        max_y - min_y + 1
    )

    body_width = (
        max_x - min_x + 1
    )

    body_area = len(body_pixels)

    total_area = (
        mask.shape[0] *
        mask.shape[1]
    )

    body_area_ratio = (
        body_area / total_area
    )

    height_width_ratio = (
        body_height / body_width
    )


    # --------------------------------------
    # Width at different body positions
    # --------------------------------------

    def width_at_height(position):

        row_y = int(
            min_y +
            position * body_height
        )

        row_y = max(
            0,
            min(
                row_y,
                mask.shape[0] - 1
            )
        )

        row = np.where(
            mask[row_y] > 0
        )[0]

        if len(row) == 0:
            return 0

        return (
            row.max() -
            row.min() +
            1
        )


    shoulder_width = (
        width_at_height(0.25)
    )

    chest_width = (
        width_at_height(0.35)
    )

    waist_width = (
        width_at_height(0.50)
    )

    hip_width = (
        width_at_height(0.65)
    )

    thigh_width = (
        width_at_height(0.75)
    )


    # --------------------------------------
    # Create feature dataframe
    # --------------------------------------

    features = pd.DataFrame([{

        "body_height_pixels":
            body_height,

        "body_width_pixels":
            body_width,

        "body_area":
            body_area,

        "body_area_ratio":
            body_area_ratio,

        "height_width_ratio":
            height_width_ratio,

        "shoulder_width":
            shoulder_width,

        "chest_width":
            chest_width,

        "waist_width":
            waist_width,

        "hip_width":
            hip_width,

        "thigh_width":
            thigh_width,

        "shoulder_width_ratio":
            shoulder_width /
            body_height,

        "chest_width_ratio":
            chest_width /
            body_height,

        "waist_width_ratio":
            waist_width /
            body_height,

        "hip_width_ratio":
            hip_width /
            body_height,

        "thigh_width_ratio":
            thigh_width /
            body_height

    }])

    return features


# ==========================================
# 4. OPEN CAMERA
# ==========================================

print("\nOpening camera...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("❌ Could not open camera")

    segmenter.close()

    exit()


print("✅ Camera opened successfully")
print()
print("Stand approximately 2-3 meters")
print("away from the camera.")
print()
print("Press Q to quit.")


# ==========================================
# 5. CAMERA LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read frame")
        break


    # --------------------------------------
    # Flip camera for natural view
    # --------------------------------------

    frame = cv2.flip(
        frame,
        1
    )


    # --------------------------------------
    # Convert BGR → RGB
    # --------------------------------------

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------
    # PERSON SEGMENTATION
    # --------------------------------------

    results = segmenter.process(
        rgb_frame
    )


    # --------------------------------------
    # Create clean binary mask
    # --------------------------------------

    if results.segmentation_mask is not None:

        mask = (
            results.segmentation_mask
            > 0.60
        ).astype(
            np.uint8
        ) * 255


        # ----------------------------------
        # Remove small noise
        # ----------------------------------

        kernel = np.ones(
            (5, 5),
            np.uint8
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )


        # ----------------------------------
        # Keep largest connected component
        # ----------------------------------

        num_labels, labels, stats, _ = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )


        if num_labels > 1:

            largest_label = 1 + np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )

            mask = np.where(
                labels == largest_label,
                255,
                0
            ).astype(
                np.uint8
            )


        # ==================================
        # EXTRACT IMAGE FEATURES
        # ==================================

        features = extract_features(
            mask
        )


        if features is not None:

            try:

                # ==========================
                # BODY MEASUREMENT PREDICTION
                # ==========================

                X_measurement = features[
                    measurement_features
                ]


                predicted_measurements = (
                    measurement_model.predict(
                        X_measurement
                    )
                )


                measurement_names = [

                    "ankle",
                    "arm-length",
                    "bicep",
                    "calf",
                    "chest",
                    "forearm",
                    "height",
                    "hip",
                    "leg-length",
                    "shoulder-breadth",
                    "shoulder-to-crotch",
                    "thigh",
                    "waist",
                    "wrist"
                ]


                measurement_df = (
                    pd.DataFrame(
                        predicted_measurements,
                        columns=
                        measurement_names
                    )
                )


                # ==========================
                # T-SHIRT SIZE PREDICTION
                # ==========================

                X_size = (
                    measurement_df[
                        size_features
                    ]
                )


                size_prediction = (
                    size_model.predict(
                        X_size
                    )
                )


                predicted_size = (
                    label_encoder
                    .inverse_transform(
                        size_prediction
                    )[0]
                )


                # ==================================
                # DISPLAY MEASUREMENTS
                # ==================================

                display = frame.copy()


                # Dark overlay for text
                overlay = display.copy()

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (display.shape[1], 260),
                    (0, 0, 0),
                    -1
                )

                display = cv2.addWeighted(
                    overlay,
                    0.60,
                    display,
                    0.40,
                    0
                )


                # ==================================
                # TITLE
                # ==================================

                cv2.putText(
                    display,
                    "T-SHIRT SIZE ESTIMATION",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2
                )


                # ==================================
                # FINAL SIZE
                # ==================================

                cv2.putText(
                    display,
                    f"SIZE: {predicted_size}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 0),
                    3
                )


                # ==================================
                # LEFT COLUMN
                # ==================================

                left_measurements = [

                    "ankle",
                    "arm-length",
                    "bicep",
                    "calf",
                    "chest",
                    "forearm",
                    "height"
                ]


                y_position = 115


                for name in left_measurements:

                    value = (
                        measurement_df[
                            name
                        ].iloc[0]
                    )

                    text = (
                        f"{name}: "
                        f"{value:.1f} cm"
                    )

                    cv2.putText(
                        display,
                        text,
                        (20, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1
                    )

                    y_position += 20


                # ==================================
                # RIGHT COLUMN
                # ==================================

                right_measurements = [

                    "hip",
                    "leg-length",
                    "shoulder-breadth",
                    "shoulder-to-crotch",
                    "thigh",
                    "waist",
                    "wrist"
                ]


                y_position = 115


                for name in right_measurements:

                    value = (
                        measurement_df[
                            name
                        ].iloc[0]
                    )

                    text = (
                        f"{name}: "
                        f"{value:.1f} cm"
                    )

                    cv2.putText(
                        display,
                        text,
                        (330, y_position),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (255, 255, 255),
                        1
                    )

                    y_position += 20


                # ==================================
                # DISPLAY
                # ==================================

                cv2.imshow(
                    "T-Shirt Size Estimation",
                    display
                )


            except Exception as error:

                cv2.putText(
                    frame,
                    "Prediction Error",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                cv2.imshow(
                    "T-Shirt Size Estimation",
                    frame
                )


        else:

            cv2.imshow(
                "T-Shirt Size Estimation",
                frame
            )


    else:

        cv2.imshow(
            "T-Shirt Size Estimation",
            frame
        )


    # ==========================================
    # QUIT
    # ==========================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

segmenter.close()

print("\nCamera prediction stopped.")