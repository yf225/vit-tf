# On AWS GPU node
"""
docker run -it --gpus all tensorflow/tensorflow:2.7.0-gpu bash

### In docker:
apt install -y git

# Then

pip install einops tensorflow_datasets matplotlib
pip install -U tensorflow-addons

export PATH=/usr/local/cuda-11.2/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64:${LD_LIBRARY_PATH}

cd ~
rm -rf ./vit-tf || true
git config --global credential.helper store
git clone https://github.com/yf225/vit-tf.git
# enter username and password

# Then

cd ./vit-tf

# Max fusion
CUDA_VISIBLE_DEVICES=0,1,2,3 TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=2" python3 vit_tf_gpu.py --bits=16 --micro_batch_size=4

# No fusion
CUDA_VISIBLE_DEVICES=0,1,2,3 TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=-1 --tf_xla_max_cluster_size=1" python3 vit_tf_gpu.py --bits=16 --micro_batch_size=4

# No fusion
CUDA_VISIBLE_DEVICES=0 TF_XLA_FLAGS="--tf_xla_enable_xla_devices --tf_xla_auto_jit=-1 --tf_xla_max_cluster_size=1" python3 vit_tf_gpu.py --bits=16 --micro_batch_size=4
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
from custom_vit_model import build_model

num_attention_heads = 16
hidden_size = 1280
num_layers = 32

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--bits", type=int)
parser.add_argument("--micro_batch_size", type=int)
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
new_argv = []
for argv in sys.argv:
    if not argv.startswith("--"):
        new_argv.append(argv)
sys.argv = new_argv

num_epochs = 10
learning_rate = 1e-8

image_size = 224
patch_size = 16  # Size of the patches to be extract from the input images

"""## Training
Actually train the model. The first epoch will be quite a bit slower as we must XLA-compile the execution graph and load the data.
"""
def run():
    print("Working on: bits: {}, micro_batch_size: {}".format(bits, micro_batch_size))

    # # Start TF profiler server.
    # profiler_port = 9012
    # tf.profiler.experimental.server.start(profiler_port)

    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync

    strategy_scope = strategy.scope()

    global_batch_size = micro_batch_size * num_devices

    # Input data
    num_examples = global_batch_size * 2
    num_steps = num_examples / global_batch_size
    num_classes = 1000  # Default in Megatron ViT
    input_shape = (image_size, image_size, 3)

    train_examples = np.zeros(shape=(num_examples, *input_shape), dtype=np.float32).astype(global_dtype.as_numpy_dtype)
    train_labels = np.random.randint(0, num_classes, size=(num_examples, 1))
    # See https://keras.io/guides/distributed_training/ for how to set batch size
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels)).batch(global_batch_size).repeat(10).prefetch(2)

    with strategy_scope:
        model = build_model(
            image_size=image_size,
            patch_size=patch_size,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            mlp_dim=4*hidden_size,
            classes=num_classes,
            dropout=0.,
            dtype=global_dtype,
            dtype_str=dtype_str,
        )
        # print(model.summary())
        optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        metrics = []
        model.compile(
            optimizer=optimizer,
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # Do we use the same loss in Megatron for ViT?
            metrics=metrics,
        )

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
    print("flag: {}, bits: {}, micro_batch_size: {}, time per step (s): {:.3f}".format(os.environ["TF_XLA_FLAGS"], bits, micro_batch_size, (second_epoch_group_time - first_epoch_group_time) / delta / num_steps))

    return history


def main(_):
  run()


if __name__ == '__main__':
  app.run(main)
