from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os


app = Flask(__name__)

# ==============================
# MODEL
# ==============================

MODEL_PATH = "models/traffic_sign_cnn.keras"

model = load_model(MODEL_PATH)


# ==============================
# TRAFFIC SIGN NAMES
# ==============================

class_names = {
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
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
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


# ==============================
# UPLOAD FOLDER
# ==============================

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==============================
# HOME PAGE
# ==============================

@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please select an image."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        filename = secure_filename(file.filename)

        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        file.save(filepath)

        # ==============================
        # READ IMAGE
        # ==============================

        image = cv2.imread(filepath)

        if image is None:
            return render_template(
                "index.html",
                error="Unable to read the image."
            )

        # Convert BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        image = cv2.resize(
            image,
            (32, 32)
        )

        # Normalize
        image = image.astype(
            "float32"
        ) / 255.0

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # ==============================
        # PREDICTION
        # ==============================

        prediction = model.predict(
            image,
            verbose=0
        )

        predicted_class = int(
            np.argmax(prediction[0])
        )

        confidence = float(
            prediction[0][predicted_class] * 100
        )

        sign_name = class_names[
            predicted_class
        ]

        result = {
            "class_id": predicted_class,
            "sign_name": sign_name,
            "confidence": f"{confidence:.2f}",
            "image": filename
        }

    return render_template(
        "index.html",
        result=result
    )


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(
        debug=True
    )
