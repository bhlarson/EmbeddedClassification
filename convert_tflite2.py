import argparse
import os
import sys
import shutil
import glob
import cv2
import numpy as np
import datetime
import tensorflow as tf


parser = argparse.ArgumentParser()

parser.add_argument('--debug', action='store_true', help='Wait for debugger attach')

parser.add_argument('--inmodel', type=str, default='./saved_model/1590151212', help='Model path')
parser.add_argument('--outmodel', type=str, default='./tflite/1590151212_int8.tflite', help='Model path')

parser.add_argument('--data_dir', type=str, default='/store/Datasets/imdb/imdb_crop/18', help='Path to data directory ')
parser.add_argument('--match', type=str, default='*', help='File wildcard')
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

#def representative_dataset_gen():
def representative_dataset_gen(files, steps = 25):
  
  for i in range(steps):

    img = cv2.imread(files[i])
    img = cv2.resize(img,(_WIDTH,_HEIGHT))
    yield [img.astype(np.float32)]

def main(FLAGS):
    files = get_filenames(FLAGS.data_dir, FLAGS.match)

    converter = tf.lite.TFLiteConverter.from_saved_model('./saved_model/1590151212')
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]

    #converter.representative_dataset = representative_dataset_gen
    converter.representative_dataset = representative_dataset_gen
    converter.representative_dataset = lambda:representative_dataset_gen(files)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_model = converter.convert()
    open("./tflite/1590151212_int8.tflite", "wb").write(tflite_model)
    stream = os.popen('edgetpu_compiler ./tflite/1590151212_int8.tflite -o ./etpu')
    compileout = stream.read()
    print(compileout)
    

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