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


import argparse
import os
import sys
import shutil
import glob
import cv2
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as tf_trt
from tensorflow.python.saved_model import signature_constants
#import tensorrt as trt
#import uff
import tensorflow as tf

print('Python Version {}'.format(sys.version))

parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=20,
                    help='Number of training epochs: '
                         'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                         'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                         'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                         'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                         'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                         'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

parser.add_argument('--epochs_per_eval', type=int, default=1,
                    help='The number of training epochs to run between evaluations.')

parser.add_argument('--tensorboard_images_max_outputs', type=int, default=6,
                    help='Max number of batch elements to generate for Tensorboard.')

parser.add_argument('--batch_size', type=int, default=24,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, 
                    default='/store/Datasets/imdb/imdb_crop/26',
                    #default='C:\\data\\datasets\\imdb',
                    help='Path to the directory containing the imdb data tf record.')

parser.add_argument('--base_architecture', type=str, default='resnet_v2_101',
                    choices=['resnet_v2_50', 'resnet_v2_101'],
                    help='The architecture of base Resnet building block.')

# Pre-trained models: https://github.com/tensorflow/models/blob/master/research/slim/README.md
parser.add_argument('--pre_trained_model', type=str, 
                    default='/store/training/resnet_v2_101_2017_04_14/resnet_v2_101.ckpt',
                    #default='C:\\data\\training\\resnet_v2_101_2017_04_14\\resnet_v2_101.ckpt',
                    help='Path to the pre-trained model checkpoint.')

#parser.add_argument('--debug', action='store_true', help='Wait for debugge attach.')
parser.add_argument('--debug', type=bool, default=True, help='Wait for debugge attach')

parser.add_argument('--savedmodel', type=str, 
                    default='saved_model/1587836450',
                    #default='C:\\data\\training\\imdb\\savedmodel\\1582542379',
                    help='Path to the pre-trained model checkpoint.')


_NUM_CLASSES = 21
_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

def getTrtGraphDef(saved_model_path):
    saved_model_loaded = tf.saved_model.load(saved_model_path)
    graph_func = saved_model_loaded.signatures[
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    return graph_func.graph.as_graph_def()

def main(FLAGS):

    #TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    trt_graph = getTrtGraphDef(FLAGS.savedmodel)

    imageFiles = glob.glob(FLAGS.data_dir+'/*.jpg')

    for i, imgFile in enumerate(imageFiles):
        img = cv2.imread(imgFile,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (_WIDTH,_HEIGHT))

        pred = trt_graph(img)[0]

    '''with tf.Session() as sess:
        # First load the SavedModel into the session    
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],FLAGS.savedmodel)
        graph = tf.get_default_graph()
        for i, op in enumerate(graph.get_operations()):
        if op.name:
            print(op.name)
        else:
            print(op.node_def.name)
        features = graph.get_tensor_by_name("features:0")
        pred_gender = graph.get_tensor_by_name("pred_gender:0")
        pred_age = graph.get_tensor_by_name("pred_age:0")


        image = tf.placeholder(tf.float32, shape=[None, _HEIGHT, _WIDTH, _DEPTH], name="input_image")

        for i, imgFile in enumerate(imageFiles):
        img = cv2.imread(imgFile,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (_WIDTH,_HEIGHT))
        pred = sess.run(pred_gender, feed_dict={image: [img]})

        fig = plt.figure(figsize=(7,11))
        ax1 = fig.add_subplot(3,1,1)
        ax1.imshow(record[0][0])
        ax2 = fig.add_subplot(3,1,2)
        ax2.imshow(tf.squeeze(record[1][0], axis=2))
        ax3 = fig.add_subplot(3,1,3)
        ax3.imshow(pred)
        plt.show()

        output = sess.run([output_tensor], feed_dict={input_tensor: input_data})'''

    print('complete')

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
    print('{} exit'.format(sys.argv[0]))