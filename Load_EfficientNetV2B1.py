import tensorflow as tf
def load_base_model():
    base_model=tf.keras.applications.efficientnet_v2.EfficientNetV2B1(
        include_top=False,
        weights="imagenet",
        input_shape=(512,512,3),
    )
    return base_model

model = load_base_model()