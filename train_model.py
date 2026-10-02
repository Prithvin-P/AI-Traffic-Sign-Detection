import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers


# ==========================================
# 1. SETTINGS
# ==========================================

DATASET_PATH = "datasets/Train"
IMG_SIZE = 32
NUM_CLASSES = 43

print("======================================")
print(" TRAFFIC SIGN CLASSIFICATION - CNN")
print("======================================")
print()


# ==========================================
# 2. CHECK DATASET
# ==========================================

if not os.path.exists(DATASET_PATH):
    print("ERROR: Dataset folder not found!")
    print("Expected:", DATASET_PATH)
    exit()

print("Dataset found!")
print("Location:", DATASET_PATH)
print()


# ==========================================
# 3. LOAD IMAGES
# ==========================================

images = []
labels = []

print("Loading images...")
print("This may take a few minutes.")
print()

for class_id in range(NUM_CLASSES):

    class_path = os.path.join(DATASET_PATH, str(class_id))

    if not os.path.exists(class_path):
        print("Missing class:", class_id)
        continue

    files = os.listdir(class_path)

    print(
        f"Loading class {class_id:2d} "
        f"({len(files)} images)"
    )

    for image_name in files:

        image_path = os.path.join(
            class_path,
            image_name
        )

        image = cv2.imread(image_path)

        if image is None:
            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(
            image,
            (IMG_SIZE, IMG_SIZE)
        )

        images.append(image)
        labels.append(class_id)


# ==========================================
# 4. CONVERT TO NUMPY
# ==========================================

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print()
print("Dataset loaded successfully!")
print("Total images:", len(X))
print("Image shape:", X.shape)
print("Labels shape:", y.shape)
print()


# ==========================================
# 5. NORMALIZATION
# ==========================================

X = X / 255.0

print("Images normalized successfully!")
print("Minimum pixel value:", X.min())
print("Maximum pixel value:", X.max())
print()


# ==========================================
# 6. TRAIN / VALIDATION SPLIT
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training and validation split completed!")
print("Training images:", len(X_train))
print("Validation images:", len(X_val))
print()


# ==========================================
# 7. CREATE CNN MODEL
# ==========================================

print("Creating CNN model...")

model = models.Sequential([

    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        input_shape=(32, 32, 3)
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("CNN model compiled successfully!")
print()


# ==========================================
# 9. TRAIN MODEL
# ==========================================

print("Starting CNN training...")
print()

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

print()
print("CNN training completed successfully!")
print()


# ==========================================
# 10. CREATE MODELS FOLDER
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)


# ==========================================
# 11. SAVE MODEL
# ==========================================

MODEL_PATH = "models/traffic_sign_cnn.keras"

model.save(MODEL_PATH)

print("======================================")
print(" MODEL SAVED SUCCESSFULLY!")
print("======================================")
print("Model location:")
print(MODEL_PATH)
print()