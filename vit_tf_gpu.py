# On AWS GPU node
"""
docker run -it tensorflow/tensorflow:2.7.0-gpu bash

### In docker:
apt install -y git

pip install einops tensorflow_datasets matplotlib
pip install -U tensorflow-addons

export PATH=/usr/local/cuda-11.2/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH}
export TF_XLA_FLAGS=--tf_xla_auto_jit=2

cd ~
rm -rf ./vit-tf || true
git clone https://github.com/yf225/vit-tf.git
cd ./vit-tf

# Copy this file content to vit.py on GPU node.
# Run the file:
python3 vit_tf_gpu.py --bits=16 --micro_batch_size=4
"""

# -*- coding: utf-8 -*-
"""Vision Transformer with TF2.0 on GPU (using ViT from Keras tutorial)
"""

import tensorflow as tf

# mixed_precision_policy = "mixed_float16"
# tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)

import re
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from absl import app
import time
import sys

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int)
parser.add_argument("--micro_batch_size", type=int)
parser.add_argument("--mode", type=str)
args = parser.parse_args()
micro_batch_size = args.micro_batch_size  # batch size per GPU
bits = args.bits
assert bits in [16, 32]
if bits == 16:
    global_dtype = tf.float16
    dtype_str = "float16"
elif bits == 32:
    global_dtype = tf.float32
    dtype_str = "float32"
assert args.mode in ["eager", "graph"]
sys.argv = sys.argv[:-3]

num_epochs = 10
learning_rate = 1e-8

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2

from typing import Tuple

class Patches(layers.Layer):
    def __init__(self, patch_size, dtype):
        super(Patches, self).__init__()
        self.patch_size = patch_size
        self.global_dtype = dtype

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, hidden_size, dtype):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.global_dtype = dtype
        self.projection = layers.Dense(units=hidden_size, dtype=self.global_dtype)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=hidden_size, dtype=self.global_dtype
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.global_dtype = dtype

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {hidden_size} should be divisible by number of heads = {num_heads}"
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query", dtype=self.global_dtype)
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key", dtype=self.global_dtype)
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value", dtype=self.global_dtype)
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out", dtype=self.global_dtype)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights


class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.global_dtype = dtype

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
            dtype=self.global_dtype,
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    name=f"{self.name}/Dense_0",
                    dtype=self.global_dtype,
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout, dtype=self.global_dtype),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1", dtype=self.global_dtype),
                tf.keras.layers.Dropout(self.dropout, dtype=self.global_dtype),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0", dtype=self.global_dtype
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2", dtype=self.global_dtype
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout, dtype=self.global_dtype)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + tf.cast(inputs, self.global_dtype)
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

def build_model(
    image_size_tuple: Tuple[int, int],
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    classes: int,
    dropout: float,
    dtype,
    activation=None,
    include_top=True,
    representation_size=None,
):
    """Build a ViT model.
    Args:
        image_size: The size of input images.
        patch_size: The size of each patch (must fit evenly in image_size)
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        num_layers: The number of transformer layers to use.
        hidden_size: The number of filters to use
        num_heads: The number of transformer heads
        mlp_dim: The number of dimensions for the MLP output in the transformers.
        dropout_rate: fraction of the units to drop for dense layers.
        activation: The activation to use for the final layer.
        include_top: Whether to include the final classification layer. If not,
            the output will have dimensions (batch_size, hidden_size).
        representation_size: The size of the representation prior to the
            classification layer. If None, no Dense layer is inserted.
    """
    assert (image_size_tuple[0] % patch_size == 0) and (
        image_size_tuple[1] % patch_size == 0
    ), "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size_tuple[0], image_size_tuple[1], 3), dtype=global_dtype)
    # Sanity check for dtype
    assert str(x.dtype.name) == dtype_str
    # Create patches.
    patches = Patches(patch_size, dtype=dtype)(x)
    # Encode patches.
    y = PatchEncoder(num_patches, hidden_size, dtype=dtype)(patches)

    for n in range(num_layers):
        y, _ = TransformerBlock(
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"Transformer/encoderblock_{n}",
            dtype=dtype,
        )(y)
    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm", dtype=dtype
    )(y)
    y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="ExtractToken")(y)
    if representation_size is not None:
        y = tf.keras.layers.Dense(
            representation_size, name="pre_logits", activation="tanh", dtype=dtype
        )(y)
    if include_top:
        top_dense = tf.keras.layers.Dense(classes, name="head", activation=activation, dtype=dtype)
        y = top_dense(y)
        # Sanity check for dtype
        assert str(y.dtype.name) == dtype_str
        assert str(top_dense.dtype) == dtype_str
    return tf.keras.models.Model(inputs=x, outputs=y)

"""## Training
Actually train the model. The first epoch will be quite a bit slower as we must XLA-compile the execution graph and load the data.
"""
def run():
    print("Working on: bits: {}, micro_batch_size: {}".format(bits, micro_batch_size))

    # # Start TF profiler server.
    # profiler_port = 9012
    # tf.profiler.experimental.server.start(profiler_port)

    strategy = tf.distribute.MirroredStrategy()

    strategy_scope = strategy.scope()

    global_batch_size = micro_batch_size * strategy.num_replicas_in_sync

    # Input data
    num_examples = global_batch_size * 2
    num_steps = num_examples / global_batch_size
    num_classes = 1000  # Default in Megatron ViT
    input_shape = (image_size, image_size, 3)

    # rng = np.random.default_rng()
    train_examples = np.zeros(shape=(num_examples, *input_shape), dtype=np.float32).astype(global_dtype.as_numpy_dtype)
    train_labels = np.random.randint(0, num_classes, size=(num_examples, 1))
    # train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    # train_dataset = train_dataset.batch(global_batch_size).repeat(10).prefetch(2)

    with strategy_scope:
        model = build_model(
            image_size_tuple=(image_size, image_size),
            patch_size=patch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mlp_dim=4*hidden_size,
            classes=num_classes,
            dropout=0.,
            dtype=global_dtype,
        )
        # print(model.summary())
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        metrics = []
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Do we use the same loss in Megatron for ViT?
            metrics=metrics,
            run_eagerly=(args.mode == "eager"),
        )
        model.run_eagerly = (args.mode == "eager")

    # Warm up
    history = model.fit(
        train_examples,
        train_labels,
        # train_dataset,
        batch_size=global_batch_size,
        epochs=5,
        callbacks=[],
    )

    delta = 5
    start_time = time.time()
    history = model.fit(
        train_examples,
        train_labels,
        # train_dataset,
        batch_size=global_batch_size,
        epochs=3,
        callbacks=[],
    )
    first_epoch_group_time = time.time() - start_time

    start_time = time.time()
    history = model.fit(
        train_examples,
        train_labels,
        # train_dataset,
        batch_size=global_batch_size,
        epochs=3 + delta,
        callbacks=[],
    )
    second_epoch_group_time = time.time() - start_time
    print("bits: {}, micro_batch_size: {}, time per step: {:.3f}s".format(bits, micro_batch_size, (second_epoch_group_time - first_epoch_group_time) / delta / num_steps))

    return history


def main(_):
  run()


if __name__ == '__main__':
  app.run(main)
