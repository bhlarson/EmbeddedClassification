

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
import numpy as np
import datetime
import tflite_runtime.interpreter as tflite

print('Python Version {}'.format(sys.version))

parser = argparse.ArgumentParser()

parser.add_argument('--debug', type=bool, default=True, help='Wait for debugger attach')
parser.add_argument('--model', type=str, default='./tflite/1588860197.tflite', help='Model path')

parser.add_argument('--data_dir', type=str, 
                    default='/home/mendel/data/imdb',
                    #default='C:\\data\\datasets\\imdb',
                    help='Path to the directory containing the imdb data tf record.')

parser.add_argument('--match', type=str, default='*',
                    help='File wildcard')


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

    interpreter = tflite.Interpreter(model_path=FLAGS.model)
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()   

    #loaded = tf.saved_model.load(FLAGS.model)
    #print(list(loaded.signatures.keys()))
    #infer = loaded.signatures["serving_default"]
    #print(infer.structured_outputs)
    #print (infer.inputs[0])
    imgs = get_filenames(FLAGS.data_dir, FLAGS.match)

    img = cv2.imread(imgs[0])
    imgShape = img.shape
    center = np.array([imgShape[1]/2, imgShape[0]/2])
    d =  np.array([_HEIGHT/2,_WIDTH/2])
    p1 = tuple((center-d).astype(int))
    p1 = (max(p1[0],0),max(p1[1],0))
    p2 = tuple((center+d).astype(int))
    p2 = (min(p2[0],imgShape[0]-1),min(p2[1],imgShape[1]-1))
    crop = cv2.resize(img[p1[1]:p2[1], p1[0]:p2[0]],(_WIDTH,_HEIGHT))
    interpreter.set_tensor(input_details[0]['index'], crop.astype(np.float32))
    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    age = interpreter.get_tensor(output_details[0]['index'])[0][0]

    gender = 'male'
    if(interpreter.get_tensor(output_details[1]['index'])[0] < 1):
        gender = 'female'

    print('{}:{}, {}:{}'.format(output_details[0]['name'], age, output_details[1]['name'],gender))

    start_time = datetime.datetime.now()
    for i, imfile in enumerate(imgs):
        img = cv2.imread(imfile)
        imgShape = img.shape
        center = np.array([imgShape[1]/2, imgShape[0]/2])
        d =  np.array([_HEIGHT/2,_WIDTH/2])
        p1 = tuple((center-d).astype(int))
        p1 = (max(p1[0],0),max(p1[1],0))
        p2 = tuple((center+d).astype(int))
        p2 = (min(p2[0],imgShape[0]-1),min(p2[1],imgShape[1]-1))
        crop = cv2.resize(img[p1[1]:p2[1], p1[0]:p2[0]],(_WIDTH,_HEIGHT))
        interpreter.set_tensor(input_details[0]['index'], crop.astype(np.float32))
        interpreter.invoke()


        age = interpreter.get_tensor(output_details[0]['index'])[0][0]

        gender = 'male'
        if(interpreter.get_tensor(output_details[1]['index'])[0] < 1):
            gender = 'female'

        print('{}:{}, {}:{}'.format(output_details[0]['name'], age, output_details[1]['name'],gender))

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
