import tensorflow as tf
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
        self.projection = layers.Dense(units=hidden_size, dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros')
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
        self.query_dense = tf.keras.layers.Dense(hidden_size, name="query", dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros')
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key", dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros')
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value", dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros')
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out", dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros')

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
                    kernel_initializer='zeros',
                    bias_initializer='zeros',
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout, dtype=self.global_dtype),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1", dtype=self.global_dtype, kernel_initializer='zeros', bias_initializer='zeros'),
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
    image_size: int,
    patch_size: int,
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    mlp_dim: int,
    classes: int,
    dropout: float,
    dtype,
    dtype_str,
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
    assert image_size % patch_size == 0, "image_size must be a multiple of patch_size"
    x = tf.keras.layers.Input(shape=(image_size, image_size, 3), dtype=dtype)
    # Sanity check for dtype
    assert str(x.dtype.name) == dtype_str
    # Create patches.
    patches = Patches(patch_size, dtype=dtype)(x)
    # Encode patches.
    num_patches = (image_size // patch_size) ** 2
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
            representation_size, name="pre_logits", activation="tanh", dtype=dtype, kernel_initializer='zeros', bias_initializer='zeros'
        )(y)
    if include_top:
        top_dense = tf.keras.layers.Dense(classes, name="head", activation=activation, dtype=dtype, kernel_initializer='zeros', bias_initializer='zeros')
        y = top_dense(y)
        # Sanity check for dtype
        assert str(y.dtype.name) == dtype_str
        assert str(top_dense.dtype) == dtype_str
    return tf.keras.models.Model(inputs=x, outputs=y)
