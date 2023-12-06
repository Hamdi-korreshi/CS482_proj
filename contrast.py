import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the vision encoder (image model)
vision_model = keras.Sequential(
    [
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.GlobalMaxPooling2D(),
    ]
)

# Define the text encoder (language model)
language_model = keras.Sequential(
    [
        layers.Input(shape=(None,), dtype="int32"),
        layers.Embedding(input_dim=5000, output_dim=128),
        layers.Bidirectional(layers.LSTM(256, return_sequences=True)),
        layers.GlobalMaxPooling1D(),
    ]
)

# Define the contrastive loss function
class ContrastiveLoss(keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0], 1))
        y_pred = tf.math.l2_normalize(y_pred, axis=1)
        y_true = tf.math.l2_normalize(y_true, axis=1)
        return 2 - 2 * tf.reduce_sum(y_true * y_pred, axis=1)

# Define the dual encoder model
class DualEncoder(keras.Model):
    def __init__(self, text_model, img_model, temperature=1, **kwargs):
        super(DualEncoder, self).__init__(**kwargs)
        self.text_model = text_model
        self.img_model = img_model
        self.temperature = temperature
        self.loss_tracker = keras.metrics.Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        # Define the forward pass.
        # Both the image and text models are used to compute embeddings.
        img_emb = self.img_model(features["image"])
        text_emb = self.text_model(features["caption"])
        return img_emb, text_emb

    def compute_loss(self, x):
        # Compute the loss.
        img_emb, text_emb = x
        return ContrastiveLoss(self.temperature)(img_emb, text_emb)

    def train_step(self, data):
        # Define the training step.
        with tf.GradientTape() as tape:
            loss = self.compute_loss(self(data, training=True))
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        # Define the test step.
        loss = self.compute_loss(self(data, training=False))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
