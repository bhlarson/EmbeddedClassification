if False:
    # https://code.visualstudio.com/docs/python/debugging#_remote-debugging
    # Launch applicaiton on remote computer: 
    # > python3 -m ptvsd --host 10.150.41.30 --port 3000 --wait makerecord.py
    import ptvsd
    # Allow other computers to attach to ptvsd at this IP address and port.
    ptvsd.enable_attach(address=('10.150.41.30', 3000), redirect_output=True)
    # Pause the program until a remote debugger is attached
    print("Wait for debugger attach")
    ptvsd.wait_for_attach()

import os
import sys
import argparse
import json
import build_data
import random
import math
import datetime
import scipy.io as sio

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from console_progressbar import ProgressBar

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path', 
        type=str,
        #default='C:\\data\\datasets\\imdb\\',
        default='/store/Datasets/imdb/',
        help='Path to data directory')

    parser.add_argument('--out', 
        type=str,
        #default='C:\\data\\datasets\\imdb\\',
        default='/store/Datasets/imdb/',

        help='Path to output directory')

    parser.add_argument('--image_wildcard', 
        type=str,
        default='.jpg',
        help='Image file wildcard e.g.: .jpg')

    parser.add_argument('--annotation_wildcard', 
        type=str,
        default='_cls.png',
        help='Image file wildcard e.g.: _cls.png')

    parser.add_argument('--seed', 
        type=float, 
        default=None, 
        help='Random float seed')

    
    parser.add_argument('--shards', 
        type=int,
        default= 1,
        help='Number of tfrecord shards')

    parser.add_argument('--sets', type=json.loads,
        default='[{"name":"training", "ratio":0.7}, {"name":"validation", "ratio":0.3}]',
        help='Json string containin array of [{"name":"<>", "ratio":<probability>}]')

    parser.add_argument('--size', type=int,
        default= 200,
        help='Image pizel size')

    parser.add_argument('--show', 
        type=bool,
        default=False,
        help='Display incremental results')

    args = parser.parse_args()
    return args

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
    with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
      start_idx = shard_id * num_per_shard
      end_idx = min((shard_id + 1) * num_per_shard, num_images)
      for i in range(start_idx, end_idx):
        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            i + 1, len(filenames), shard_id))
        sys.stdout.flush()
        # Read the image.
        image_filename = os.path.join(
            FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
        image_data = tf.compat.v1.gfile.FastGFile(image_filename, 'rb').read()
        height, width = image_reader.read_image_dims(image_data)
        # Read the semantic segmentation annotation.
        seg_filename = os.path.join(
            FLAGS.semantic_segmentation_folder,
            filenames[i] + '.' + FLAGS.label_format)
        seg_data = tf.compat.v1.gfile.FastGFile(seg_filename, 'rb').read()
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
    

def Example(args, i, imbddata, image_reader):

    imgpath = '{}imdb_crop/{}'.format(args.path,imbddata['full_path'])
    image_data = tf.io.gfile.GFile(imgpath, 'rb').read()
    y,x,c = image_reader.read_image_dims(image_data)
    isMale = int(imbddata['gender'])
    
    scale = args.size/max(y,x)
    #xOffset = int((args.size-scale*x)/2)
    #yOffset = int((args.size-scale*y)/2)

    #M = np.float32([[scale,0,xOffset],[0,scale,yOffset]])
    #dst = cv2.warpAffine(img,M,(args.size,args.size))

    if args.show:
        fig = plt.figure()
        a = fig.add_subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        a.set_title(imbddata['name'])

        a = fig.add_subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        a.set_title('age:{}, gender:{}'.format(imbddata['age'],imbddata['gender']))

        plt.show()

    feature = {
        'subject':  _str_feature(imbddata['name']),
        'height':  tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        'width':  tf.train.Feature(int64_list=tf.train.Int64List(value=[x])),
        'depth':  tf.train.Feature(int64_list=tf.train.Int64List(value=[c])),
        'gender': tf.train.Feature(int64_list=tf.train.Int64List(value=[isMale])),
        'age': tf.train.Feature(float_list=tf.train.FloatList(value=[imbddata['age']])),
        'path': _str_feature(imbddata['full_path']),
        'image': _bytes_feature(image_data)
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

    

def WriteRecords(args, datasets, imbddata):
    '''
        datasets = [{'name':'training', 'ratio':0.7}, {'name':'validation', 'ratio':0.3}]
        imbddata = []
    '''

    # shuffle records between datasets and shards
    random.shuffle(imbddata)
    

    image_reader = build_data.ImageReader('jpeg', channels=3)

    start = 0
    numEntries = len(imbddata)
    for ids, dataset in enumerate(datasets):
        for shard_id in range(args.shards):
            output_filename = os.path.join(args.out, '%s-%05d-of-%05d.tfrecord' % (dataset['name'], shard_id, args.shards))
            if(ids == len(datasets)-1 and shard_id == args.shards-1):
                stop = numEntries
            else:
                groupSize = int(numEntries*dataset['ratio']/args.shards)
                stop = start+groupSize

            print('{} start {} stop {}'.format(output_filename, start, stop))
            pb = ProgressBar(total=stop-start,prefix='{}'.format(dataset['name']), suffix='of {}'.format(stop-start), decimals=3, length=75, fill='%', zfill='-')
            with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(start,stop):
                    example = Example(args, i,imbddata[i],image_reader)
                    tfrecord_writer.write(example.SerializeToString())
                    if i%100 == 0:
                        pb.print_progress_bar(i-start)
            pb.print_progress_bar(stop-start)
            start = stop

            sys.stdout.write('\n')
            sys.stdout.flush()

def matlab2datetime(matlab_datenum):
    try:
        ordinal = datetime.date.fromordinal(int(matlab_datenum))
        matlabDelta = datetime.timedelta(days = 366)
        fraction = datetime.timedelta(days=int(matlab_datenum)%1)
        python_datetime = ordinal - matlabDelta + fraction
    except:
        python_datetime = None

    #ordinal = int(matlab_datenum) - 366
    #if(ordinal > 1):
    #    ordinal
    #    python_datetime = datetime.date.fromordinal(ordinal) + datetime.timedelta(days=matlab_datenum%1)
    #    python_datetime = datetime.fromordinal(int(matlab_datenum)) + datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    #else:
    #    python_datetime = None

    return python_datetime

def LoadMatToDict(args, face_score_threshold = 1.0):
    mat_contents = sio.loadmat(args.path+'imdb.mat')
    
    imbddata = []
    numEntries = len(mat_contents['imdb']['dob'][0][0][0])

    for i in range(numEntries):
        entry = {'dob':mat_contents['imdb']['dob'][0][0][0][i],
                 'photo_taken':mat_contents['imdb']['photo_taken'][0][0][0][i],
                 'full_path':np.array2string(mat_contents['imdb']['full_path'][0][0][0][i][0]).replace('\'',''),
                 'gender':mat_contents['imdb']['gender'][0][0][0][i],
                 'name':mat_contents['imdb']['name'][0][0][0][i][0],
                 'face_location':mat_contents['imdb']['face_location'][0][0][0][i][0].tolist(),
                 'face_score':mat_contents['imdb']['face_score'][0][0][0][i],
                 'second_face_score':mat_contents['imdb']['second_face_score'][0][0][0][i],
                 }

        dob = matlab2datetime(entry['dob'])
        photo = datetime.datetime(entry['photo_taken'],7,1).date()
        if dob is not None:
            entry['age'] = (photo-dob).days/365.25
        else:
            entry['age'] = None

        if entry['face_score'] >= face_score_threshold and not math.isnan(entry['gender']) and entry['age'] is not None:
            imbddata.append(entry)
    return imbddata

def main(args):

    imbddata = LoadMatToDict(args)

    #with open(args.path + "imbddata.json") as json_file:
    #    imbddata = json.load(json_file)
    
    WriteRecords(args, args.sets, imbddata)
   
    print('exit')

if __name__ == '__main__':
    args = parse_arguments()
    main(args)