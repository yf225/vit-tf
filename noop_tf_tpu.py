# !pip install -U tensorflow-addons

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
print("Tensorflow version " + tf.__version__)

mixed_precision_policy = "mixed_bfloat16"
tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)


import re
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from official.common import distribute_utils
import tensorflow_addons as tfa
from absl import app

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

micro_batch_size = 24  # batch size per TPU core
# global_batch_size = micro_batch_size * tpu_strategy.num_replicas_in_sync  # defined later

num_epochs = 10
learning_rate = 0.001

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

from typing import Tuple

def build_model(
    image_size_tuple: Tuple[int, int],
    classes: int,
):
    """Build a no-op model.
    """
    x = tf.keras.layers.Input(shape=(image_size_tuple[0], image_size_tuple[1], 3))
    x_flattened = tf.keras.layers.Flatten()(x)
    x_sum = tf.reduce_sum(x_flattened, axis=1, keepdims=True)
    y = tf.keras.layers.Dense(classes)(x_sum)
    return tf.keras.models.Model(inputs=x, outputs=y)

"""## Training
Actually train the model. The first epoch will be quite a bit slower as we must XLA-compile the execution graph and load the data.
"""
def run():
    # # Start TF profiler server.
    # profiler_port = 9012
    # tf.profiler.experimental.server.start(profiler_port)

    strategy = distribute_utils.get_distribution_strategy(
        distribution_strategy="tpu",
        num_gpus=1,  # How many GPUs to use at each worker with the DistributionStrategies API. The default is 1.
        tpu_address=os.environ["TPU_NAME"])

    strategy_scope = distribute_utils.get_strategy_scope(strategy)

    print("num_replicas_in_sync: ", strategy.num_replicas_in_sync)

    global_batch_size = micro_batch_size * strategy.num_replicas_in_sync

    # Input data
    num_examples = global_batch_size * 3  # NOTE: this is a compromise, because CPU RAM on Cloud TPU machine is very low
    num_classes = 1000  # Default in Megatron ViT
    input_shape = (image_size, image_size, 3)

    train_examples = np.random.randn(num_examples, *input_shape)
    train_labels = np.random.randint(0, num_classes, size=(num_examples, 1))
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    # train_dataset = train_dataset.batch(global_batch_size).repeat(10).prefetch(2)

    with strategy_scope: # creating the model in the TPUStrategy scope means we will train the model on the TPU
        model = build_model(
            image_size_tuple=(image_size, image_size),
            classes=num_classes,
        )
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        metrics = []
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Do we use the same loss in Megatron for ViT?
            metrics=metrics,
        )

    history = model.fit(
        train_examples,
        train_labels,
        # train_dataset,
        batch_size=global_batch_size,
        epochs=num_epochs,
        callbacks=[],
    )

    return history


def main(_):
  run()


if __name__ == '__main__':
  app.run(main)
