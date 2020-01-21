import os
from os.path import join, getsize
import glob
import argparse
import json
import datetime
import numpy as np
import pandas as pd
from shutil import copyfile

def list_dirs(top, maxdepth):
    dirs, nondirs = [], []

    for iTop in top:
        entries = os.scandir(iTop)
        for entry in entries:
            (dirs if entry.is_dir() else nondirs).append(entry.path)

    if maxdepth > 1:
        outDirs = list_dirs(dirs, maxdepth-1)
    else:
        outDirs = dirs

    return outDirs

def matlab2datetime(matlab_datenum):
    try:
        ordinal = datetime.date.fromordinal(int(matlab_datenum))
        matlabDelta = datetime.timedelta(days = 366)
        fraction = datetime.timedelta(days=matlab_datenum%1)
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

def main(args):
    imbddata = {}
    with open(args.path + "/imdb_name.json") as json_file:
        imbddata['name'] = json.load(json_file)

    with open(args.path + "/imdb_dob.json") as json_file:
        imbddata['dob'] = json.load(json_file)
        
    with open(args.path + "/imdb_photo_taken.json") as json_file:
        imbddata['photo_taken'] = json.load(json_file)
        
    with open(args.path + "/imdb_full_path.json") as json_file:
        imbddata['full_path'] = json.load(json_file)
        
    with open(args.path + "/imdb_gender.json") as json_file:
        imbddata['gender'] = json.load(json_file)
        
    with open(args.path + "/imdb_face_location.json") as json_file:
        imbddata['face_location'] = json.load(json_file)
        
    with open(args.path + "/imdb_face_score.json") as json_file:
        imbddata['face_score'] = json.load(json_file)
        
    with open(args.path + "/imdb_second_face_score.json") as json_file:
        imbddata['second_face_score'] = json.load(json_file)
        
    with open(args.path + "/imdb_celeb_names.json") as json_file:
        imbddata['celeb_names'] = json.load(json_file)

    with open(args.path + "/imdb_celeb_id.json") as json_file:
        imbddata['celeb_id'] = json.load(json_file)

    imbddata['age'] = [None] * len(imbddata['dob'])
    for i in range(0,len(imbddata['photo_taken'])):
        dob = matlab2datetime(imbddata['dob'][i])
        photo = datetime.datetime(imbddata['photo_taken'][i],7,1).date()
        if dob is not None:
            dt = (photo-dob).days/365.25
            imbddata['age'][i] = dt


    with open(args.outpath + '/imbddata.json', 'w') as outfile:
        json.dump(imbddata, outfile)
 
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path', type=str,
        default='/store/datasets/imdb-wiki/imdb',
        help='Path to data directory')

    parser.add_argument('--outpath', type=str,
        default='/store/datasets/imdb-wiki/imdb',
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