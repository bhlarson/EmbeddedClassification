"""Train a Resnet model for age classification and gender regression from the imdb dataset."""

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

import tensorflow as tf
from matplotlib import pyplot as plt

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

parser.add_argument('--clean_model_dir', action='store_true',
                    help='Whether to clean up the model directory if present.')

parser.add_argument('--train_epochs', type=int, default=1,
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

parser.add_argument('--batch_size', type=int, default=1,
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

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--resnet_size', type=int, default=101,
                    help='Resnet size (18, 34, 50, 101, 152, 200)')


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

_NUM_IMAGES = {
    'train': 10582,
    'validation': 1449,
}


def get_filenames(is_training, data_dir):
  """Return a list of filenames.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: path to the the directory containing the input data.

  Returns:
    A list of file names.
  """
  if is_training:
    return glob.glob(os.path.join(data_dir, 'training-?????-of-?????.tfrecord'))
  else:
    return glob.glob(os.path.join(data_dir, 'validation-?????-of-?????.tfrecord'))


def parse_record(raw_record):
  feature = {
        'subject':  tf.FixedLenFeature((), tf.string, default_value=''),
        'height':  tf.FixedLenFeature((), tf.int64),
        'width':  tf.FixedLenFeature((), tf.int64),
        'depth':  tf.FixedLenFeature((), tf.int64),
        'gender': tf.FixedLenFeature((), tf.int64),
        'age': tf.FixedLenFeature((), tf.float32),
        'path': tf.FixedLenFeature((), tf.string, default_value=''),
        'image': tf.FixedLenFeature((), tf.string, default_value=''),
  }

  parsed = tf.parse_single_example(raw_record, feature)

  #image = tf.io.decode_raw(parsed['image'], tf.uint8)
  image = tf.io.decode_jpeg(parsed['image'], _DEPTH)
  #image_shape = tf.stack([parsed['height'], parsed['width'], _DEPTH])
  #image = tf.reshape(image, image_shape)
  #image.set_shape([_HEIGHT, _WIDTH, _DEPTH])
  image = tf.image.resize_with_crop_or_pad(image, _HEIGHT, _WIDTH)
 
  label = {
    'name':parsed['subject'],
    'gender':parsed['gender'],
    'age':parsed['age']
  }


  return image, label


def preprocess_image(image, label, is_training):
  """Preprocess a single image of layout [height, width, depth]."""
  '''if is_training:
    # Randomly scale the image and label.
    image, label = preprocessing.random_rescale_image_and_label(
        image, label, _MIN_SCALE, _MAX_SCALE)

    # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
    image, label = preprocessing.random_crop_or_pad_image_and_label(
        image, label, _HEIGHT, _WIDTH, _IGNORE_LABEL)

    # Randomly flip the image and label horizontally.
    image, label = preprocessing.random_flip_left_right_image_and_label(
        image, label)

    image.set_shape([_HEIGHT, _WIDTH, 3])
    label.set_shape([_HEIGHT, _WIDTH, 1])'''

  #tf.image.resize_with_crop_or_pad(image, _HEIGHT, _WIDTH)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(is_training, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  #if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    #dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])

  if is_training:

    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    # is a relatively small dataset, we choose to shuffle the full epoch.
    dataset = dataset.shuffle(buffer_size=500)
    # We call repeat after shuffling, rather than before, to prevent separate epochs from blending together.
    dataset = dataset.repeat(num_epochs)

  dataset = dataset.map(parse_record)
  
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(batch_size)

  return dataset

  #iterator = dataset.make_one_shot_iterator()
  #images, labels = iterator.get_next()

  #return images, labels

def serving_input_fn():
    shape = [_WIDTH, _HEIGHT, _DEPTH]
    features = {
        "image" : tf.FixedLenFeature(shape=shape, dtype=tf.uint8),
    }
    return tf.estimator.export.build_parsing_serving_input_receiver_fn(features)

def main(FLAGS):
  tf.compat.v1.enable_eager_execution

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
          'num_train': _NUM_IMAGES['train'],
          'initial_learning_rate': FLAGS.initial_learning_rate,
          'max_iter': FLAGS.max_iter,
          'end_learning_rate': FLAGS.end_learning_rate,
          'power': _POWER,
          'momentum': _MOMENTUM,
          'freeze_batch_norm': FLAGS.freeze_batch_norm,
          'initial_global_step': FLAGS.initial_global_step,
          'resnet_size': FLAGS.resnet_size,
          'kGender':1.0,
          'kAge':1.0,
          'learning_rate':1e-3,
          'data_format':None,
          #'data_format':'channels_last',
      }

  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  model = tf.estimator.Estimator(
      model_fn=resnet_model.resnetv2_model_fn,
      model_dir=FLAGS.model_dir,
      config=run_config,
      params=params)

  for step in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
    tensors_to_log = {
      #'learning_rate': 'learning_rate',
      #'cross_entropy': 'cross_entropy',
      #'train_px_accuracy': 'train_px_accuracy',
      #'train_mean_iou': 'train_mean_iou',
    }

    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=10)
    #train_hooks = [logging_hook]
    #eval_hooks = None

    #if FLAGS.debug:
    #  debug_hook = tf_debug.LocalCLIDebugHook()
    #  train_hooks.append(debug_hook)
    #  eval_hooks = [debug_hook]

    print("Start evaluation.")
    # Evaluate the model and print results
    predictions = model.predict(input_fn=lambda: input_fn(False, FLAGS.data_dir, 1))
    for i, prediction in enumerate(predictions):
      msg = '{}: pred_age {}, pred_gender {}, '.format(i, prediction['pred_age'],prediction['pred_gender'])
      print(msg)
      #plt.imshow(prediction['image'])
      #plt.title(msg)
      #plt.show()


    #eval_results = model.evaluate(
    #    input_fn=lambda: input_fn(False, FLAGS.data_dir, 1),
        #steps=10  # For debug
    #)
    #print(eval_results)
    #  plt.imshow(prediction['image'])
    #  plt.show()

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(FLAGS)
