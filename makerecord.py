import os
import sys
import argparse
import json
import build_data
import random

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.

  Args:
    dataset_split: The dataset split (e.g., train, test).

  Raises:
    RuntimeError: If loaded image and label have different shape.
  """
  dataset = os.path.basename(dataset_split)[:-4]
  sys.stdout.write('Processing ' + dataset)
  filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
  num_images = len(filenames)
  num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

  image_reader = build_data.ImageReader('jpeg', channels=3)
  label_reader = build_data.ImageReader('png', channels=1)

  for shard_id in range(_NUM_SHARDS):
    output_filename = os.path.join(
        FLAGS.output_dir,
        '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
        seg_height, seg_width = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
        # Convert to tf example.
        example = build_data.image_seg_to_tfexample(
            image_data, filenames[i], height, width, seg_data)
        tfrecord_writer.write(example.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

def shuffle(seed, *lists):

    if seed is None:
        seed = random.random()
    for ls in lists:
        random.Random(seed).shuffle(ls)

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _str_feature(value):
    return _bytes_feature(str.encode(value))
    

def Example(args, i, imbddata):

    imgpath = '{}imdb_crop/{}'.format(args.path,imbddata['full_path'][i])
    img = cv2.imread(imgpath)
    y,x,c = img.shape
    scale = args.size/max(y,x)
    xOffset = int((args.size-scale*x)/2)
    yOffset = int((args.size-scale*y)/2)

    M = np.float32([[scale,0,xOffset],[0,scale,yOffset]])
    dst = cv2.warpAffine(img,M,(args.size,args.size))

    if args.show:
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(imbddata['name'][i])

        a = fig.add_subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        a.set_title('age:{}, gender:{}'.format(imbddata['age'][i],imbddata['gender'][i]))

        plt.show()

    feature = {
        'subject':  _str_feature(imbddata['name'][i]),
        'height':  tf.train.Feature(int64_list=tf.train.Int64List(value=[args.size])),
        'width':  tf.train.Feature(int64_list=tf.train.Int64List(value=[args.size])),
        'depth':  tf.train.Feature(int64_list=tf.train.Int64List(value=[3])),
        'gender': tf.train.Feature(float_list=tf.train.FloatList(value=[float(imbddata['gender'][i])])),
        'age': tf.train.Feature(float_list=tf.train.FloatList(value=[imbddata['age'][i]])),
        'path': _str_feature(imbddata['full_path'][i]),
        'image': _bytes_feature(dst.tobytes())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

    

def WriteRecords(args, datasets, imbddata):
    '''
        datasets = [{'name':'training', 'ratio':0.7}, {'name':'validation', 'ratio':0.2}, {'name':'test', 'ratio':0.1}]
        imbddata = []
    '''

    # shuffle records between datasets and shards

    shuffle(args.seed, imbddata['age'], 
        imbddata['dob'], 
        imbddata['face_location'], 
        imbddata['face_score'], 
        imbddata['full_path'], 
        imbddata['gender'], 
        imbddata['name'], 
        imbddata['photo_taken'], 
        imbddata['second_face_score'])

    start = 0
    numEntries = len(imbddata['dob'])
    for shard_id in range(args.shards):
        for ids, dataset in enumerate(datasets):
            output_filename = os.path.join(args.out, '%s-%05d-of-%05d.tfrecord' % (dataset['name'], shard_id, args.shards))
            if(ids < len(dataset)-1):
                stop = int(numEntries*dataset['ratio'])
            else:
                stop = numEntries

            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(start,stop):
                    if(imbddata['face_score'][i]>= 1.0 and imbddata['age'][i] and imbddata['gender'][i]):
                        example = Example(args, i,imbddata)
                        tfrecord_writer.write(example.SerializeToString())
            start = stop

            sys.stdout.write('\n')
            sys.stdout.flush()



def main(args):
    imbddata = {}
    with open(args.path + "imbddata.json") as json_file:
        imbddata = json.load(json_file)
    datasets = [{'name':'training', 'ratio':0.7}, {'name':'validation', 'ratio':0.2}, {'name':'test', 'ratio':0.1}]
    WriteRecords(args, datasets, imbddata)

    datasets = [{'name':'training', 'ratio':0.7}, {'name':'validation', 'ratio':0.2}, {'name':'test', 'ratio':0.1}]
    WriteRecords(args, datasets, imbddata)
    
    #for i in range(0,len(imbddata['gender'])):
    #    img = args.path +'imdb_crop/'+ imbddata['full_path'][i]
    #    features_dataset = tf.data.Dataset.from_tensor_slices((feature(imbddata['gender'][i]), feature(imbddata['age'][i])))


    print('exit')
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path', 
        type=str,
        default='C:\\data\\datasets\\imdb\\',
        #default='/store/datasets/imdb/',
        help='Path to data directory')

    parser.add_argument('--out', 
        type=str,
        default='C:\\data\\datasets\\imdb\\',
        #default='/store/datasets/imdb/',

        help='Path to output directory')

    parser.add_argument('--image_wildcard', 
        type=str,
        default='.tif',
        help='Image file wildcard e.g.: .tif')

    parser.add_argument('--annotation_wildcard', 
        type=str,
        default='_cls.png',
        help='Image file wildcard e.g.: _cls.png')

    parser.add_argument('--seed', 
        type=float, 
        default=None, 
        help='Random float seed between 0.1 to 1.0')

    
    parser.add_argument('--shards', 
        type=int,
        default= 12,
        help='Number of tfrecord shards')

    parser.add_argument('--set_probability', 
        nargs='+', type=float,
        default= [0.7, 0.3],
        help='set probability min=0.0, max = 1.0')

    parser.add_argument('--datasets', nargs='+', 
        type=str,
        default=['training', 'validation'],
        help='set probability min=0.0, max = 1.0')

    parser.add_argument('--size', type=int,
        default= 128,
        help='Image pizel size')

    parser.add_argument('--show', 
        type=bool,
        default=False,
        help='Display incremental results')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)