import argparse
import numpy as np
import shutil
import tensorflow.compat.v1 as tf
import tensorflow.neuron as tfn
from resnet_model import ResNet50

import os

os.environ['NEURON_CC_FLAGS'] = '--fp32-cast=all'
tf.keras.backend.set_learning_phase(0)
tf.keras.backend.set_floatx('float16')


COMPILED_MODEL_DIR = 'compiled-keras-resnet50'
model = ResNet50(1000)
model.load_weights('saved_weights.h5')
print(model.inputs[0].dtype)
shutil.rmtree('tmp-model', ignore_errors=True)
tf.saved_model.simple_save(session=tf.keras.backend.get_session(), export_dir='tmp-model', inputs={'input': model.inputs[0]}, outputs={'output': model.outputs[0]})


batch_sizes = [16]
for batch_size in batch_sizes:
    example_input = np.zeros([batch_size, 224, 224, 3], dtype=np.float16)

    # Prepare export directory (old one removed)
    compiled_model_dir = f'{COMPILED_MODEL_DIR}_batch' + str(batch_size)
    shutil.rmtree(compiled_model_dir, ignore_errors=True)


    tfn.saved_model.compile('tmp-model', compiled_model_dir, batch_size=batch_size, model_feed_dict={'input': example_input})
