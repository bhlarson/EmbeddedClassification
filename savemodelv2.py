"""Train a Resnet model for age classification and gender regression from the imdb dataset."""
# Run with tensorflow container: FROM tensorflow/tensorflow:2.1.0-gpu-py3

import argparse
import os
import sys
import shutil
import glob
import numpy as np
import tensorflow as tf

import resnet_modelv2 as resnet_model
#from tensorflow.python.tools import freeze_graph

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))

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

parser.add_argument('--batch_size', type=int, default=3,
                    help='Number of examples per batch.')

parser.add_argument('--learning_rate_policy', type=str, default='poly',
                    choices=['poly', 'piecewise'],
                    help='Learning rate policy to optimize loss.')

parser.add_argument('--max_iter', type=int, default=30,
                    help='Number of maximum iteration used for "poly" learning rate policy.')

parser.add_argument('--data_dir', type=str, 
                    default='/store/Datasets/imdb',
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

parser.add_argument('--output_stride', type=int, default=16,
                    choices=[8, 16],
                    help='Output stride for DeepLab v3. Currently 8 or 16 is supported.')

parser.add_argument('--freeze_batch_norm', action='store_true',
                    help='Freeze batch normalization parameters during the training.')

parser.add_argument('--initial_learning_rate', type=float, default=7e-3,
                    help='Initial learning rate for the optimizer.')

parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                    help='End learning rate for the optimizer.')

parser.add_argument('--initial_global_step', type=int, default=0,
                    help='Initial global step for controlling learning rate when fine-tuning model.')

parser.add_argument('--weight_decay', type=float, default=2e-4,
                    help='The weight decay to use for regularizing the model.')

#parser.add_argument('--debug', action='store_true', help='Wait for debugge attach.')
parser.add_argument('--debug', type=bool, default=True, help='Wait for debugge attach')

parser.add_argument('--resnet_size', type=int, default=101,
                    help='Resnet size (18, 34, 50, 101, 152, 200)')

parser.add_argument('--savedmodel', type=str, default='./saved_model', help='Path to savedmodel.')
parser.add_argument('--tflitemodel', type=str, default='./tflite', help='Path to tflite model.')



_NUM_CLASSES = 21
_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

_POWER = 0.9
_MOMENTUM = 0.9

_BATCH_NORM_DECAY = 0.9997

'''def freeze_model(saved_model_dir, output_filename, output_node_names):
    output_graph_filename = os.path.join(saved_model_dir, output_filename)
    initializer_nodes = ''
    output_node_names = output_node_names
    freeze_graph.freeze_graph(
        input_saved_model_dir=saved_model_dir,
        output_graph=output_graph_filename,
        saved_model_tags = tag_constants.SERVING,
        output_node_names=output_node_names,
        initializer_nodes=initializer_nodes,
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        restore_op_name=None,
        filename_tensor_name=None,
        clear_devices=False,
        input_meta_graph=False,
    )
    print('graph frozen!')'''

def serving_input_receiver_fn():
    shape = [_HEIGHT, _WIDTH, _DEPTH]
    image = tf.compat.v1.placeholder(dtype=tf.uint8, shape=shape, name='image')
    images = tf.expand_dims(image, 0)
    return tf.estimator.export.TensorServingInputReceiver(images, image)

def main(unused_argv):

  if FLAGS.clean_model_dir:
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)

  params = {
          'output_stride': FLAGS.output_stride,
          'batch_size': FLAGS.batch_size,
          'base_architecture': FLAGS.base_architecture,
          'pre_trained_model': FLAGS.pre_trained_model,
          'batch_norm_decay': _BATCH_NORM_DECAY,
          'num_classes': _NUM_CLASSES,
          'tensorboard_images_max_outputs': FLAGS.tensorboard_images_max_outputs,
          'weight_decay': FLAGS.weight_decay,
          'learning_rate_policy': FLAGS.learning_rate_policy,
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step,
          'resnet_size': FLAGS.resnet_size,
          'kGender':50.0,
          'kAge':1.0,
          'learning_rate':1e-3,
          'data_format':None,
      }

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  model = tf.estimator.Estimator(
      model_fn=resnet_model.resnetv2_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params=params)


  savedmodel = model.export_saved_model(FLAGS.savedmodel, serving_input_receiver_fn, experimental_mode=tf.estimator.ModeKeys.PREDICT, as_text=True)

  print('savedmodel {}'.format(savedmodel.decode('utf-8')))

  # Converting a SavedModel to a TensorFlow Lite model.
  if False:
    converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel)
    tflite_model = converter.convert()
    open(FLAGS.tflitemodel+"model.tflite", "wb").write(tflite_model)

  if True:
    # https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html
    from tensorflow.python.compiler.tensorrt import trt_convert
    output_saved_model_dir = FLAGS.savedmodel+'TRT'

    conversion_params = trt_convert.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<32))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(maximum_cached_engines=100)
    conversion_params = conversion_params._replace(minimum_segment_size=1)

    converter = trt_convert.TrtGraphConverterV2(input_saved_model_dir=savedmodel.decode('utf-8'),conversion_params=conversion_params)
    converter.convert()
    # converter.build fails.  Continue with converter.convert to see if network will run.
    #num_runs = 1 # Not clearly defined.  May be number of runs in calibration: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/compiler/tensorrt/trt_convert.py
    #def my_input_fn():
        #return np.random.normal(size=(_HEIGHT, _WIDTH, _DEPTH)).astype(np.int8)
    #    for _ in range(num_runs):
    #        inp1 = np.random.normal(size=(_HEIGHT, _WIDTH, _DEPTH)).astype(np.float32)
    #        yield inp1
    #converter.build(input_fn=my_input_fn)
    converter.save(output_saved_model_dir)

    #with tf.Session() as sess:
    #    # First load the SavedModel into the session    
    #    tf.saved_model.loader.load(
    #        sess, [tf.saved_model.tag_constants.SERVING],
    #      output_saved_model_dir)
    #    output = sess.run([output_tensor], feed_dict={input_tensor: input_data})


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
