if False:
    # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
    # Launch applicaiton on remote computer: 
    # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait train_imdb.py
    import ptvsd
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=('10.150.41.30', 3000), redirect_output=True)
    # Pause the program until a remote debugger is attached
    print("Wait for debugger attach")
    ptvsd.wait_for_attach()

import numpy as np
import argparse
import os
import sys
import shutil
import glob
import cv2
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import convert_to_constants

import resnet_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

#parser.add_argument('--debug', action='store_true', help='Wait for debugge attach.')
parser.add_argument('--debug', type=bool, default=True, help='Wait for debugge attach')

parser.add_argument('--data_dir', type=str, 
                    default='/store/Datasets/imdb/imdb_crop/00/',
                    #default='C:\\data\\datasets\\imdb',
                    help='Path to the directory containing the imdb data tf record.')
parser.add_argument('--savedmodel', type=str, 
                    default='saved_model/1587647071',
                    #default='C:\\data\\training\\imdb\\savedmodel\\1582542379',
                    help='Path to the pre-trained model checkpoint.')


_NUM_CLASSES = 21
_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

def serving_input_fn():
    shape = [_HEIGHT, _WIDTH, _DEPTH]
    features = {
        "features" : tf.FixedLenFeature(shape=shape, dtype=tf.string),
    }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(features)


def main(unused_argv):
  # model = tf.saved_model.load(FLAGS.savedmodel) # tf2
    # https://www.tensorflow.org/guide/saved_model
    saved_model_loaded = tf.saved_model.load( FLAGS.savedmodel, tags=[tf.saved_model.SERVING])
    infer = saved_model_loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    
    imageFiles = glob.glob(FLAGS.data_dir+'/*.jpg')

    for i, imgFile in enumerate(imageFiles):
        img = cv2.imread(imgFile,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (_WIDTH,_HEIGHT))

        output = infer(img)

        print('{}={}'.format(imgFile, output[0].numpy()))

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.debug:
        # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
        # Launch applicaiton on remote computer: 
        # > python3 -m ptvsd --host 0.0.0.0 --port 3000 --wait predict_imdb.py
        import ptvsd
        # Allow other computers to attach to ptvsd at this IP address and port.
        ptvsd.enable_attach(address=('0.0.0.0', 3000), redirect_output=True)
        # Pause the program until a remote debugger is attached
        print("Wait for debugger attach")
        ptvsd.wait_for_attach()
        print("Debugger Attached")

    main(unparsed)
    print('{} exit'.format(sys.argv[0]))
