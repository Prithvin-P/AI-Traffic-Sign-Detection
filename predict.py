import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/traffic_sign_cnn.keras"
TEST_FOLDER = "test_image"

IMAGE_SIZE = 32


# ============================================================
# TRAFFIC SIGN NAMES
# GTSRB - 43 CLASSES
# ============================================================

CLASS_NAMES = {
    0: "Speed limit 20 km/h",
    1: "Speed limit 30 km/h",
    2: "Speed limit 50 km/h",
    3: "Speed limit 60 km/h",
    4: "Speed limit 70 km/h",
    5: "Speed limit 80 km/h",
    6: "End of speed limit 80 km/h",
    7: "Speed limit 100 km/h",
    8: "Speed limit 120 km/h",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}


# ============================================================
# LOAD MODEL
# ============================================================

print("======================================")
print(" TRAFFIC SIGN AI - PREDICTION")
print("======================================")
print()

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully!")
print()


# ============================================================
# FIND IMAGE
# ============================================================

if not os.path.exists(TEST_FOLDER):

    print("ERROR:")
    print(f"Folder '{TEST_FOLDER}' does not exist.")
    print()
    print("Create the folder:")
    print("D:\\traffic sign classification\\test_image")

    exit()


image_files = [

    file for file in os.listdir(TEST_FOLDER)

    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )

]


if len(image_files) == 0:

    print("ERROR: No image found inside test_image folder.")

    exit()


# ============================================================
# PROCESS FIRST IMAGE
# ============================================================

image_name = image_files[0]

image_path = os.path.join(
    TEST_FOLDER,
    image_name
)


print("Testing image:")
print(image_name)
print()


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(image_path)


if image is None:

    print("ERROR: Could not read image.")

    exit()


# ============================================================
# PREPROCESS IMAGE
# ============================================================

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


image = cv2.resize(
    image,
    (IMAGE_SIZE, IMAGE_SIZE)
)


image = image.astype("float32") / 255.0


image = np.expand_dims(
    image,
    axis=0
)


# ============================================================
# PREDICTION
# ============================================================

predictions = model.predict(
    image,
    verbose=0
)[0]


# ============================================================
# TOP 3 PREDICTIONS
# ============================================================

top_3_indices = np.argsort(
    predictions
)[-3:][::-1]


top_3 = []


for class_id in top_3_indices:

    confidence = float(
        predictions[class_id] * 100
    )

    sign_name = CLASS_NAMES.get(
        int(class_id),
        "Unknown"
    )

    top_3.append(
        (
            int(class_id),
            sign_name,
            confidence
        )
    )


# ============================================================
# MAIN RESULT
# ============================================================

predicted_class = top_3[0][0]

predicted_sign = top_3[0][1]

confidence = top_3[0][2]


# ============================================================
# DISPLAY RESULT
# ============================================================

print("======================================")
print(" PREDICTION RESULT")
print("======================================")

print(
    f"Image          : {image_name}"
)

print(
    f"Predicted Class: {predicted_class}"
)

print(
    f"Traffic Sign   : {predicted_sign}"
)

print(
    f"Confidence     : {confidence:.2f}%"
)

print()


# ============================================================
# UNKNOWN WARNING
# ============================================================

print("======================================")
print(" IMAGE VALIDATION")
print("======================================")


# This is a preliminary warning.
# It is NOT a complete unknown-image detector yet.

if confidence < 50:

    print(
        "⚠ WARNING: Low confidence prediction."
    )

    print(
        "The image may not be a supported traffic sign."
    )

else:

    print(
        "✓ Image appears compatible with"
    )

    print(
        "  one of the trained traffic-sign classes."
    )


print()


# ============================================================
# TOP 3 RESULTS
# ============================================================

print("======================================")
print(" TOP 3 PREDICTIONS")
print("======================================")


for position, item in enumerate(
    top_3,
    start=1
):

    class_id = item[0]

    sign_name = item[1]

    confidence = item[2]


    print(
        f"{position}. "
        f"{sign_name}"
    )

    print(
        f"   Class      : {class_id}"
    )

    print(
        f"   Confidence : {confidence:.2f}%"
    )

    print()


print("Prediction completed successfully!")