import os
from os.path import join, getsize
import glob
import argparse
import json
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
    day = datetime.fromordinal(int(matlab_datenum))
    dayfrac = datetime.timedelta(days=matlab_datenum%1) - datetime.timedelta(days = 366)
    return day + dayfrac

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

    imbddata['age'] =  pd.to_datetime(np.array(imbddata['photo_taken'],7,1)) - pd.to_datetime(np.array(imbddata['dob'])-719529, unit='D')


    wImage = '{}/*{}'.format(args.path, args.image_wildcard)
    images = glob.glob(wImage)
    wAnnotation = '{}/*{}'.format(args.path, args.annotation_wildcard)
    annotations = glob.glob(wAnnotation)

    for dataset in args.datasets:
        imagesDir = '{}/images/{}'.format(args.outpath,dataset)
        annotationsDir = '{}/annotations/{}'.format(args.outpath,dataset)
        if not os.path.exists(imagesDir):
            os.makedirs(imagesDir)
        if not os.path.exists(annotationsDir):
            os.makedirs(annotationsDir)

    for imageName in images:
        imageBaseName = os.path.splitext(imageName.split("/")[-1])[0]
        annotationName = '{}/{}{}'.format(args.path,imageBaseName,args.annotation_wildcard)
        if annotationName in annotations:
            inst_rand = np.random.uniform()

            set_probability = 0.0
            for i, dataset in enumerate(args.datasets):
                set_probability += args.set_probability[i]
                if(inst_rand < set_probability):

                    imgSrc = imageName
                    annSrc = annotationName
                    imgDest = '{}/images/{}/{}.tif'.format(args.outpath,dataset,imageBaseName)
                    annDest = '{}/annotations/{}/{}.png'.format(args.outpath,dataset,imageBaseName)
                    copyfile(imgSrc, imgDest)
                    copyfile(annSrc, annDest)
                    break  
        

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--path', type=str,
        default='/store/datasets/imdb-wiki/imdb',
        help='Path to data directory')

    parser.add_argument('--outpath', type=str,
        default='/store/TrainingSet/20191211_GV_FV_C_W_SS',
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