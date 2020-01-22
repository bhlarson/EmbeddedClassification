import os
import argparse
import json
import tensorflow as tf

def feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def main(args):
    imbddata = {}
    with open(args.path + "/imbddata.json") as json_file:
        imbddata = json.load(json_file)

    for i in range(0,len(imbddata['gender'])):
        img = imbddata['full_jpath'][i]
        features_dataset = tf.data.Dataset.from_tensor_slices((feature(imbddata['gender'][i]), 
            feature(imbddata['age'][i], feature2, feature3))

    print('exit')
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path', type=str,
        default='C:\data\datasets\imdb',
        help='Path to data directory')

    parser.add_argument('--out', type=str,
        default='C:\data\datasets\imdb',
        help='Path to output directory')

    parser.add_argument('--image_wildcard', type=str,
        default='.tif',
        help='Image file wildcard e.g.: .tif')

    parser.add_argument('--annotation_wildcard', type=str,
        default='_cls.png',
        help='Image file wildcard e.g.: _cls.png')

    parser.add_argument('--set_probability', nargs='+', type=float,
        default= [0.7, 0.3],
        help='set probability min=0.0, max = 1.0')

    parser.add_argument('--datasets', nargs='+', type=str,
        default=['training', 'validation'],
        help='set probability min=0.0, max = 1.0')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)