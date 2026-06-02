"""
train_vision.py
----------------
Lightweight transfer-learning trainer for the autonomous microgreens IoT vision system.

Backbone : MobileNetV3-Small (ImageNet weights, frozen) -> fast CPU training
Input    : (224, 224, 3)
Output   : 5-way softmax classification head
Data      : loaded from ./data/images/ via image_dataset_from_directory
Artifact : models/vision_model.keras  (HDF5, loadable with tf.keras.models.load_model)

No external file dependencies beyond the ./data/images/ folder and TensorFlow.
"""

import os
import tensorflow as tf

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DATA_DIR = "./data/images/"
MODEL_PATH = "models/vision_model.keras"

IMG_SIZE = (224, 224)          # MobileNetV3-Small expects 224x224x3
BATCH_SIZE = 32
SEED = 123
EPOCHS = 65                     # max epochs, as specified
VALIDATION_SPLIT = 0.2

# The dataset is expected to contain exactly these 5 classes. The actual class
# *names* and their integer ordering are inferred from the subfolder names of
# DATA_DIR (alphabetical), which is what determines the softmax output order.
# Known/visible classes: "healthy", "wilting", "overwatered", "mold_..." (+1 more).
EXPECTED_NUM_CLASSES = 3


def build_datasets():
    """Load train/validation datasets from the local folder structure.

    Expected layout:
        ./data/images/<class_name>/<image>.jpg
    Labels are integer-encoded (label_mode="int") to pair with
    SparseCategoricalCrossentropy.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)

    print(f"Discovered {num_classes} classes: {class_names}")
    assert num_classes == EXPECTED_NUM_CLASSES, (
        f"Expected {EXPECTED_NUM_CLASSES} classes under {DATA_DIR}, "
        f"but found {num_classes}: {class_names}. "
        "Check your folder structure (one subfolder per class)."
    )

    # Performance: cache decoded images and overlap data loading with training.
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names


def build_model(num_classes):
    """Assemble the transfer-learning model as a single Sequential stack.

    Note on preprocessing: MobileNetV3Small is instantiated with
    include_preprocessing=True (the default), so the network expects raw pixel
    values in the [0, 255] range and performs its own rescaling/normalization
    internally. image_dataset_from_directory yields [0, 255] floats, so NO
    manual Rescaling layer is needed (and adding one would double-normalize).
    """
    # Augmentation layers live inside the model sequence, as requested. They are
    # active only during training and become no-ops at inference time.
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
        ],
        name="data_augmentation",
    )

    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )
    # Freeze the backbone: only the classification head trains -> fast on CPU.
    # With trainable=False, the internal BatchNorm layers also run in inference
    # mode and will not update their running statistics.
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            data_augmentation,
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="microgreens_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def main():
    train_ds, val_ds, class_names = build_datasets()

    model = build_model(num_classes=len(class_names))
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    # Ensure the target directory exists, then save in HDF5 format.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Sanity check: confirm the production load path works as expected.
    #reloaded = tf.keras.models.load_model(MODEL_PATH)
    #print("Reload check passed -> tf.keras.models.load_model() succeeded.")
    #print(f"Output classes (in order): {class_names}")


if __name__ == "__main__":
    main()
