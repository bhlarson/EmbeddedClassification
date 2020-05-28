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
import numpy as np
import resnet_model
from utils import preprocessing
from tensorflow.python import debug as tf_debug
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

print('Python Version {}'.format(sys.version))
print('Tensorflow version {}'.format(tf.__version__))
print('GPU Available: {}'.format(tf.test.is_gpu_available()))
if(tf.test.is_gpu_available()):
  print('GPU Devices: {}'.format(tf.test.gpu_device_name()))


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default='./model',
                    help='Base directory for the model.')

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

parser.add_argument('--debug', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

parser.add_argument('--savedmodel', type=str, 
                    default='saved_model/1590410988',
                    help='Path to the pre-trained model checkpoint.')

parser.add_argument('--tflmodel', type=str, 
                    default='tflite/2020-05-25-12-52-19-lit_int8.tflite',
                    help='Path tensorflow lite model.')

parser.add_argument('--usetfl', action='store_true',
                    help='Whether to use debugger to track down bad values during training.')

_NUM_CLASSES = 21
_HEIGHT = 200
_WIDTH = 200
_DEPTH = 3
_MIN_SCALE = 0.5
_MAX_SCALE = 2.0
_IGNORE_LABEL = 255

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True, saveto='confusion.svg'):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(saveto)
    

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

  parsed = tf.parse_single_example(serialized=raw_record, features=feature)
  image = tf.io.decode_jpeg(parsed['image'], _DEPTH)
  image = tf.image.resize_with_crop_or_pad(image, _HEIGHT, _WIDTH)

  label = {
    'name':parsed['subject'],
    'gender':parsed['gender'],
    'age':parsed['age']
  }


  return image, label

def input_fn(data_dir, batch_size=20, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  dataset = tf.data.Dataset.from_tensor_slices(get_filenames(False, data_dir))
  dataset = dataset.flat_map(tf.data.TFRecordDataset)
  dataset = dataset.map(parse_record)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(batch_size)

  return dataset

def main(FLAGS):

  # Prepare dataset
  dataset = input_fn(FLAGS.data_dir, FLAGS.batch_size)
  #it = iter(dataset)
  itr = tf.data.make_one_shot_iterator(dataset)
  next = itr.get_next()
  age = []
  age_p = []
  gender = []
  gender_p = []
  with tf.Session() as sess:

    if(FLAGS.usetfl):
      # Load TFLite model and allocate tensors.
      #interpreter = tflite.Interpreter(FLAGS.model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
      interpreter = tf.lite.Interpreter(FLAGS.tflmodel)
      interpreter.allocate_tensors()

      # Get input and output tensors.
      input_details = interpreter.get_input_details()
      output_details = interpreter.get_output_details()      
    else:
      # First load the SavedModel into the session    
      tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING],FLAGS.savedmodel)
      graph = tf.get_default_graph()
      #for i, op in enumerate(graph.get_operations()):
      #  if op.name:
      #    print(op.name)
      #  else:
      #    print(op.node_def.name)

      features = graph.get_tensor_by_name("features:0")
      pred_gender = graph.get_tensor_by_name("pred_gender:0")
      pred_age = graph.get_tensor_by_name("pred_age/BiasAdd:0")
    try:
      #while True:
      for i in tqdm(range(250)):


        [image,label] = sess.run(next)

        age.append(label['age'][0])
        gender.append(label['gender'][0])

        if FLAGS.usetfl:
          interpreter.set_tensor(input_details[0]['index'], image[0])
          interpreter.invoke()
          pred_age = output_details[0]['quantization'][0]*interpreter.get_tensor(output_details[0]['index'])[0][0]
          pred_gender = interpreter.get_tensor(output_details[1]['index'])[0]

          gender_p.append(pred_gender)
          age_p.append(pred_age)

        else:
          pred = sess.run([pred_gender, pred_age], feed_dict={'input_image:0': image[0]})
          gender_p.append(pred[0][0])
          age_p.append(pred[1][0][0])

    except tf.errors.OutOfRangeError:
      pass

    '''for i, imgFile in enumerate(imageFiles):
      img = cv2.imread(imgFile,cv2.IMREAD_COLOR)
      img = cv2.resize(img, (_WIDTH,_HEIGHT)).astype(np.float)
      pred = sess.run([pred_gender, pred_age], feed_dict={'input_image:0': img})
      
      if pred[0][0] > 0.5 :
        gender = 'male'
      else:
        gender = 'female'

      print('{} pred_gender {} pred_age {}'.format(imgFile, gender, int(round(pred[1][0][0]))))'''

  ageError = np.average(np.absolute(np.array(age)-np.array(age_p)))

  confusion = confusion_matrix(gender, gender_p)
  plot_confusion_matrix(confusion, normalize=False, target_names=['female','male'], title="Gender Confusion", saveto="confusion.svg")

  print('Age average error {} Gender confusion {}'.format(ageError,confusion))

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