"""Train a Resnet model for age classification and gender regression from the imdb dataset."""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import argparse
import os
import sys
import shutil
import glob
import cv2
import datetime
#import tensorrt as trt
#import uff
#import pycuda.driver as cuda
#import pycuda.autoinit
import tensorflow as tf

print('Python Version {}'.format(sys.version))

parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, default=False, help='Wait for debugge attach')

parser.add_argument('--model', type=str, default='./saved_model/1588440354',
                    help='Base directory for the model.')

parser.add_argument('--data_dir', type=str, 
                    default='/store/Datasets/imdb/imdb_crop/00/',
                    #default='C:\\data\\datasets\\imdb',
                    help='Path to the directory containing the imdb data tf record.')

parser.add_argument('--match', type=str, default='*',
                    help='File wildcard')

# Pre-trained models: https://github.com/tensorflow/models/blob/master/research/slim/README.md
parser.add_argument('--pre_trained_model', type=str, 
                    default='/store/training/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
                    #default='C:\\data\\training\\resnet_v2_101_2017_04_14\\resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')


_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3


def get_filenames(data_dir, ext):
    """Return a list of filenames.

    Args:
        is_training: A boolean denoting whether the input is for training.
        data_dir: path to the the directory containing the input data.

    Returns:
        A list of file names.
    """
    return glob.glob(os.path.join(data_dir, ext))

def build_engine(FLAGS):
    uff_model = uff.from_tensorflow_frozen_model(FLAGS.model, debug_mode=True, return_graph_info=True)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
        builder.max_workspace_size = 1 << 30
        builder.fp16_mode = True
        builder.max_batch_size = 1
        parser.register_input("Input", (3, _WIDTH, _HEIGHT))
        parser.register_output("MarkOutput_0")
        parser.parse(uff_model_path, network)
        
        print("Building TensorRT engine, this may take a few minutes...")
        trt_engine = builder.build_cuda_engine(network)
    

def main(FLAGS):
    #TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

    #engine = build_engine(FLAGS)

    loaded = tf.saved_model.load(FLAGS.model)
    print(list(loaded.signatures.keys()))
    infer = loaded.signatures["serving_default"]
    print(infer.structured_outputs)
    print (infer.inputs[0])
    imgs = get_filenames(FLAGS.data_dir, FLAGS.match)

    img = cv2.imread(imgs[0])
    tfimg = tf.image.resize_with_crop_or_pad(tf.constant(img), _HEIGHT, _WIDTH)
    outputs = infer(tfimg)

    start_time = datetime.datetime.now()
    for i, imfile in enumerate(imgs):
        img = cv2.imread(imfile)
        tfimg = tf.image.resize_with_crop_or_pad(tf.constant(img), _HEIGHT, _WIDTH)
        outputs = infer(tfimg)

        print('{}: pred_age {}, pred_gender {}, '.format(i, outputs['pred_age'].numpy()[0,0],outputs['pred_gender'].numpy()[0]))
    analysis_done = datetime.datetime.now()
    total_time = (analysis_done-start_time).total_seconds()

    print('average image time {}'.format(total_time/len(imgs)))

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

    main(FLAGS)
    print('complete')
