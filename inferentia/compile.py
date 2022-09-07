import numpy as np
import shutil
import tensorflow as tf
import tensorflow.neuron as tfn

MODEL_DIR = 'keras-resnet50'
COMPILED_MODEL_DIR = 'compiled-keras-resnet50'
model=tf.keras.models.load_model(MODEL_DIR)

batch_sizes = [16]
for batch_size in batch_sizes:
    example_input = np.zeros([batch_size, 224, 224, 3], dtype=np.float32)

    # Prepare export directory (old one removed)
    compiled_model_dir = f'{COMPILED_MODEL_DIR}_batch' + str(batch_size)
    shutil.rmtree(compiled_model_dir, ignore_errors=True)


    model_neuron = tfn.trace(model, example_input)
    model_neuron.save(compiled_model_dir)
