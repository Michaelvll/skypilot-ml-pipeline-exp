# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""ResNet-50 implemented with Keras running on Cloud TPUs.

This file shows how you can run ResNet-50 on a Cloud TPU using the TensorFlow
Keras support. This is configured for ImageNet (e.g. 1000 classes), but you can
easily adapt to your own datasets by changing the code appropriately.

On tpu-v3-8, the batch size is 1024
# Train, GPU AMP XLA float16.
export XLA_FLAGS='--xla_gpu_cuda_data_dir=/usr/local/cuda/' && \
python3 resnet50_tpu/resnet50.py \
  --tpu=gpu \
  --data=$DATA_DIR \
  --precision=float16 \
  --model_dir=gs://resnet-test/resnet-realImagenet-gpu \
  --num_cores=1 \
  --per_core_batch_size=256 \
  --amp --xla --loss_scale=128 \
  2>&1 | tee run-realData-gpu-float16.log

# Train, TPU bfloat16.
python3 resnet50_tpu/resnet50.py \
  --tpu=$TPU_NAME \
  --data=$DATA_DIR \
  --precision=bfloat16 \
  --model_dir=gs://resnet-test/resnet-realImagenet-gpu \
  2>&1 | tee run-realData-gpu-float16.log

# Inference on GPU.
python3 resnet50_tpu/resnet50.py \
  --tpu=gpu \
  --data=$DATA_DIR \
  --precision=float16 \
  --model_dir=gs://resnet-test/resnet-realImagenet-gpu \
  --num_cores=1 \
  --mode=infer \
  --per_core_batch_size=16 \
  --amp --xla --loss_scale=128
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import time

from absl import app
from absl import flags
from absl import logging

import tensorflow.compat.v2 as tf

import imagenet_input
import model_saving_utils
import resnet_model


# Common flags for TPU models.
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored. If not specified, save to /tmp/resnet50.'))
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('infer_images', 1000000, 'Batch size per TPU core/GPU.')
flags.DEFINE_enum('mode', 'train', ['train', 'infer'], help='Mode to run: train or infer.')
FLAGS = flags.FLAGS

# Imagenet training and test data sets.
APPROX_IMAGENET_TRAINING_IMAGES = 1281167  # Number of images in ImageNet-1k train dataset.
IMAGENET_VALIDATION_IMAGES = 50000  # Number of images in eval dataset.
NUM_CLASSES = 1000

# Training hyperparameters.
_EPOCHS = 90
_USE_BFLOAT16 = 'bfloat16'
_BASE_LEARNING_RATE = 0.1
DEFAULT_MODEL_DIR = '/tmp/resnet50'

# Allow overriding epochs, steps_per_epoch for testing
flags.DEFINE_integer('num_epochs', _EPOCHS, '')
flags.DEFINE_integer(
    'steps_per_epoch', None,
    'Steps for epoch during training. If unspecified, use default value.')
flags.DEFINE_string('precision', _USE_BFLOAT16, 'float32, float16, bfloat16.')

flags.DEFINE_bool(
  'amp', False,
  'Whether to use automated mixed precision.')
flags.DEFINE_bool(
  'xla', False,
  'Whether to use accelerated linear algebra.')
flags.DEFINE_integer('loss_scale', -1, 'Loss Scale for AMP.')

# Learning rate schedule
_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


class ResnetLearningRateSchedule(
    tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate):
    super(ResnetLearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = _LR_SCHEDULE[0]
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in _LR_SCHEDULE:
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate
    }


def safe_mean(losses):
  total = tf.reduce_sum(losses)
  num_elements = tf.dtypes.cast(tf.size(losses), dtype=losses.dtype)
  return tf.math.divide_no_nan(total, num_elements)


def main(unused_argv):
  use_gpu = (FLAGS.tpu is not None and FLAGS.tpu.lower() == 'gpu')
  if use_gpu:
    tf.keras.backend.set_image_data_format('channels_first')
  assert use_gpu or (not FLAGS.amp and not FLAGS.xla), 'AMP and XLA only supported on GPU.'
  if use_gpu:
    # From Nvidia Repo, explained here: https://github.com/NVIDIA/DeepLearningExamples/issues/57
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_GPU_THREAD_COUNT'] = '2'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
  if FLAGS.amp:
    os.environ["TF_ENABLE_AUTO_MIXED_PRECISION_GRAPH_REWRITE"] = "1"
  if FLAGS.xla:
    # https://github.com/tensorflow/tensorflow/blob/8d72537c6abf5a44103b57b9c2e22c14f5f49698/tensorflow/compiler/jit/flags.cc#L78-L87
    # 1: on for things very likely to be improved
    # 2: on for everything
    # fusible: only for Tensorflow operations that XLA knows how to fuse
    #
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=1'
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
    # Best Performing XLA Option
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=fusible'
    os.environ["TF_XLA_FLAGS"] = (os.environ.get("TF_XLA_FLAGS", "") + " --tf_xla_enable_lazy_compilation=false")
  tf.enable_v2_behavior()
  model_dir = FLAGS.model_dir if FLAGS.model_dir else DEFAULT_MODEL_DIR
  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = FLAGS.steps_per_epoch or (int(
      APPROX_IMAGENET_TRAINING_IMAGES // batch_size))
  steps_per_eval = int(1.0 * math.ceil(IMAGENET_VALIDATION_IMAGES / batch_size))
  logging.info('Saving checkpoints at %s', model_dir)
  logging.info('Use TPU at %s', FLAGS.tpu if FLAGS.tpu is not None else 'local')

  if use_gpu:
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(communication=tf.distribute.experimental.CollectiveCommunication.NCCL)
  else:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  if FLAGS.mode != 'infer':
    imagenet_train = imagenet_input.ImageNetInput(
        is_training=True,
        data_dir=FLAGS.data,
        batch_size=FLAGS.per_core_batch_size,
        precision=FLAGS.precision)
    imagenet_eval = imagenet_input.ImageNetInput(
        is_training=False,
        data_dir=FLAGS.data,
        batch_size=FLAGS.per_core_batch_size,
        precision=FLAGS.precision)
    train_dataset = strategy.experimental_distribute_datasets_from_function(
        imagenet_train.input_fn)
    test_dataset = strategy.experimental_distribute_datasets_from_function(
        imagenet_eval.input_fn)

  if FLAGS.precision == 'bfloat16':
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)
  # if FLAGS.mode == 'infer' and FLAGS.precision == 'float16':
    # tf.keras.backend.set_floatx('float16')

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = resnet_model.ResNet50(num_classes=NUM_CLASSES)
    if FLAGS.mode == 'infer':
      saved_weights = os.path.join(FLAGS.model_dir, 'saved_weights.h5')
      model.load_weights(saved_weights)
    base_lr = _BASE_LEARNING_RATE * batch_size / 256
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=ResnetLearningRateSchedule(steps_per_epoch, base_lr),
        momentum=0.9,
        nesterov=True)
    if FLAGS.amp:
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer, loss_scale='dynamic' if FLAGS.loss_scale==-1 else FLAGS.loss_scale)
    training_loss = tf.keras.metrics.Mean('training_loss', dtype=tf.float32)
    training_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'training_accuracy', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)
    logging.info('Finished building Keras ResNet-50 model')

    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    initial_epoch = 0
    # latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    # if latest_checkpoint:
    #   # checkpoint.restore must be within a strategy.scope() so that optimizer
    #   # slot variables are mirrored.
    #   checkpoint.restore(latest_checkpoint)
    #   logging.info('Loaded checkpoint %s', latest_checkpoint)
    #   initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  # Create summary writers
  # train_summary_writer = tf.summary.create_file_writer(
  #     os.path.join(model_dir, 'summaries/train'))
  # test_summary_writer = tf.summary.create_file_writer(
  #     os.path.join(model_dir, 'summaries/test'))

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        if FLAGS.precision != 'float32':
          predictions = tf.cast(predictions, tf.float32)

        # Loss calculations.
        #
        # Part 1: Prediction loss.
        prediction_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions)
        loss1 = tf.reduce_mean(prediction_loss)
        # Part 2: Model weights regularization
        loss2 = tf.reduce_sum(model.losses)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = loss1 + loss2
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      training_loss.update_state(loss)
      training_accuracy.update_state(labels, predictions)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      images, labels = inputs
      predictions = model(images, training=False)
      if FLAGS.precision != 'float32':
        predictions = tf.cast(predictions, tf.float32)
      loss = tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            predictions)
      loss = safe_mean(loss)
      test_loss.update_state(loss)
      test_accuracy.update_state(labels, predictions)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def infer_step(images):
    """Inference StepFn."""
    def step_fn(inputs):
      predictions = model(inputs, training=False)
      if FLAGS.precision != 'float32':
          predictions = tf.cast(predictions, tf.float32)
      return predictions

    return strategy.run(step_fn, args=(images,))

  step_interval = 200

  if FLAGS.mode == 'infer':
    logging.info('Starting inference...')
    total_steps = FLAGS.infer_images // FLAGS.per_core_batch_size
    warmup_inf_steps = 100
    counter = 0
    inf_times = []
    import numpy as np
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import preprocess_input
    img_sgl = image.load_img('kitten_small.jpg', target_size=(224, 224))
    img_arr = image.img_to_array(img_sgl)
    img_arr2 = np.expand_dims(img_arr, axis=0)
    img_arr3 = preprocess_input(np.repeat(img_arr2, FLAGS.per_core_batch_size, axis=0))
    model_feed_dict = tf.convert_to_tensor(img_arr3, dtype=tf.float16)
    shapes = []
    while counter < total_steps + warmup_inf_steps:
        start_time = time.time()
        batch = infer_step(model_feed_dict)
        shapes.append(batch.numpy().shape)
        end_time = time.time()
        test_accuracy.reset_states()
        if counter > warmup_inf_steps:
            inf_times.append(end_time - start_time)
        counter += 1
        if counter % 1000 == 0:
            logging.info('Evaluation Iter ' + str(counter) + f'\nMean Latency: {np.mean(inf_times) * 1000:.2f} ms')
        if counter >= total_steps + warmup_inf_steps:
            break
    inf_times = np.array(inf_times)
    import pandas as pd
    throughput = FLAGS.infer_images / np.sum(inf_times)
    mean_latency = 1000.0 * np.mean(inf_times)
    P90_latency = 1000.0 * np.percentile(inf_times, 90)
    P99_latency = 1000.0 * np.percentile(inf_times, 99)
    
    df = pd.DataFrame({
        'batch_size': [FLAGS.per_core_batch_size],
        'throughput': [throughput],
        'p90_ms': [P90_latency],
        'p99_ms': [P99_latency],
        'mean_ms': [mean_latency],
        'num_images': [(counter - warmup_inf_steps) * FLAGS.per_core_batch_size],
    })
    print(df)
    df.to_csv(f'results-{FLAGS.per_core_batch_size}.csv', index=False, header=True)
    return
  train_iterator = iter(train_dataset)
  for epoch in range(initial_epoch, FLAGS.num_epochs):
    epoch_start_time = time.time()
    logging.info('Starting to run epoch: %s', epoch)
    # with train_summary_writer.as_default():
    start_time = time.time()
    for step in range(steps_per_epoch):
      if step % step_interval == 0:
        time_per_step = (time.time() - start_time) / step_interval
        logging.info(f'Running step {step} in epoch {epoch} [sec/step: {time_per_step}]')
        start_time = time.time()
      train_step(train_iterator)  
    # tf.summary.scalar(
    #     'loss', training_loss.result().numpy(), step=optimizer.iterations)
    # tf.summary.scalar(
    #     'accuracy',
    #     training_accuracy.result().numpy(),
    #     step=optimizer.iterations)
    logging.info('Training loss: %s, accuracy: %s%%',
                  round(training_loss.result().numpy(), 4),
                  round(training_accuracy.result().numpy() * 100, 2))
    epoch_time = time.time() - epoch_start_time
    logging.info(f'Epoch time: {epoch_time}; Seconds per step: {epoch_time / steps_per_epoch}')

    training_loss.reset_states()
    training_accuracy.reset_states()

    test_iterator = iter(test_dataset)
    logging.info('got test iterator')
    for step in range(steps_per_eval):
      if step % 20 == 0:
        logging.info('Starting to run eval step %s of epoch: %s', step,
                      epoch)
      test_step(test_iterator)
    # tf.summary.scalar(
    #     'loss', test_loss.result().numpy(), step=optimizer.iterations)
    # tf.summary.scalar(
    #     'accuracy', test_accuracy.result().numpy(), step=optimizer.iterations)
    logging.info('Test loss: %s, accuracy: %s%%',
                  round(test_loss.result().numpy(), 4),
                  round(test_accuracy.result().numpy() * 100, 2))
    test_loss.reset_states()
    test_accuracy.reset_states()

    # checkpoint_name = checkpoint.save(os.path.join(model_dir, 'checkpoint'))
    model_saving_utils.save_model(model, model_dir)
    # logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  logging.set_verbosity(logging.INFO)
  app.run(main)
