
import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from PIL import Image, ImageFilter


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

IMAGE_SIZE = (224, 224)
PATCH_SIZE = 16
NUM_PATCHES = 196

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ==========================================
# 2. CUSTOM VISION TRANSFORMER LAYERS
# ==========================================

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):

    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):

        patches = tf.image.extract_patches(
            images=images,
            sizes=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            strides=[
                1,
                self.patch_size,
                self.patch_size,
                1
            ],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        return tf.reshape(
            patches,
            [-1, NUM_PATCHES, patches.shape[-1]]
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "patch_size": self.patch_size
        })

        return config


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):

    def __init__(
        self,
        num_patches,
        projection_dim,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = layers.Dense(
            projection_dim
        )

        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )

    def call(self, patches):

        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        return (
            self.projection(patches)
            + self.position_embedding(positions)
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })

        return config


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

def load_trained_model(model_path):

    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "Patches": Patches,
            "PatchEncoder": PatchEncoder
        },
        compile=False
    )


# ==========================================
# 4. IMAGE PREPROCESSING
# ==========================================

def preprocess_image(image):

    image = image.convert("RGB")

    image = np.array(image)

    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    image = image / 255.0

    image = tf.image.resize_with_pad(
        image,
        224,
        224
    )

    return tf.expand_dims(
        image,
        axis=0
    )


# ==========================================
# 5. INTEGRATED GRADIENTS
# ==========================================

def integrated_gradients(
    model,
    image,
    target_class=None,
    steps=64
):

    image = tf.cast(image, tf.float32)

    baseline = tf.zeros_like(image)

    # Get prediction
    prediction = model(
        image,
        training=False
    )

    if target_class is None:

        target_class = int(
            tf.argmax(
                prediction[0]
            ).numpy()
        )

    # Interpolation steps
    alphas = tf.linspace(
        0.0,
        1.0,
        steps + 1
    )

    interpolated = (
        baseline
        + alphas[:, None, None, None]
        * (image - baseline)
    )

    # Calculate gradients of log probability
    with tf.GradientTape() as tape:

        tape.watch(interpolated)

        outputs = model(
            interpolated,
            training=False
        )

        target_scores = tf.math.log(
            outputs[:, target_class] + 1e-8
        )

    gradients = tape.gradient(
        target_scores,
        interpolated
    )

    # Average gradients
    average_gradients = tf.reduce_mean(
        (
            gradients[:-1]
            + gradients[1:]
        ) / 2.0,
        axis=0
    )

    # Integrated Gradients
    integrated = (
        image - baseline
    ) * average_gradients

    # Combine RGB channels
    attribution = tf.reduce_sum(
        integrated,
        axis=-1
    )[0].numpy()

    # Positive contributions
    positive_map = np.maximum(
        attribution,
        0
    )

    # Percentile normalization
    scale = np.percentile(
        positive_map,
        99.5
    )

    positive_map = np.clip(
        positive_map / (scale + 1e-8),
        0,
        1
    )

    return positive_map, target_class


# ==========================================
# 6. OCCLUSION SENSITIVITY
# ==========================================

def occlusion_sensitivity(
    model,
    image,
    target_class=None,
    patch_size=16,
    batch_size=32,
    mode="black"
):

    # Convert input to NumPy
    original_image = image[0].numpy()

    height, width, channels = original_image.shape

    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            "Image dimensions must be divisible by patch_size."
        )

    grid_height = height // patch_size
    grid_width = width // patch_size

    # Original prediction
    original_prediction = model(
        image,
        training=False
    ).numpy()[0]

    if target_class is None:

        target_class = int(
            np.argmax(original_prediction)
        )

    original_score = float(
        original_prediction[target_class]
    )

    # Create blurred image if required
    if mode == "blur":

        pil_image = Image.fromarray(
            np.uint8(
                np.clip(original_image * 255, 0, 255)
            )
        )

        blurred_image = np.asarray(
            pil_image.filter(
                ImageFilter.GaussianBlur(radius=15)
            ),
            dtype=np.float32
        ) / 255.0

    elif mode == "black":

        blurred_image = None

    else:

        raise ValueError(
            "mode must be 'black' or 'blur'"
        )

    occluded_images = []

    # Replace one patch at a time
    for row in range(grid_height):

        for col in range(grid_width):

            modified_image = original_image.copy()

            y1 = row * patch_size
            y2 = y1 + patch_size

            x1 = col * patch_size
            x2 = x1 + patch_size

            if mode == "black":

                modified_image[
                    y1:y2,
                    x1:x2,
                    :
                ] = 0

            else:

                modified_image[
                    y1:y2,
                    x1:x2,
                    :
                ] = blurred_image[
                    y1:y2,
                    x1:x2,
                    :
                ]

            occluded_images.append(
                modified_image
            )

    occluded_images = np.asarray(
        occluded_images,
        dtype=np.float32
    )

    # Predict in batches
    prediction_batches = []

    for start in range(
        0,
        len(occluded_images),
        batch_size
    ):

        batch = occluded_images[
            start:start + batch_size
        ]

        batch_predictions = model(
            batch,
            training=False
        ).numpy()

        prediction_batches.append(
            batch_predictions
        )

    occluded_predictions = np.concatenate(
        prediction_batches,
        axis=0
    )

    # Confidence reduction
    confidence_changes = (
        original_score
        - occluded_predictions[:, target_class]
    )

    occlusion_map = confidence_changes.reshape(
        grid_height,
        grid_width
    )

    # Keep positive confidence reductions
    occlusion_map = np.maximum(
        occlusion_map,
        0
    )

    # Resize to original image dimensions
    heatmap = tf.image.resize(
        occlusion_map[..., np.newaxis],
        (height, width)
    ).numpy().squeeze()

    # Normalize
    heatmap = heatmap / (
        np.max(heatmap) + 1e-8
    )

    return (
        heatmap,
        target_class,
        original_score
    )
