#!/usr/bin/env python3
#!/usr/bin/env python3

##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Yaoyao Liu
## NUS School of Computing
## Email: yaoyao.liu@nus.edu.sg
## Copyright (c) 2019
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import os
import numpy as np
import csv
import glob
import cv2
from shutil import copyfile
from tqdm import tqdm
from pathlib import Path
import logging
import pandas as pd

# argument parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--tar_dir',  type=str)
parser.add_argument('--imagenet_dir',  type=str)
parser.add_argument('--image_resize',  type=int,  default=84)
parser.add_argument('--num_classes', type=int, default=1000)
parser.add_argument('--num_instances_per_class', type=int, default=100)

args = parser.parse_args()

class FSImageNetGenerator(object):
    def __init__(self, input_args):
        self.input_args = input_args
        if self.input_args.tar_dir is not None:
            print('Untarring ILSVRC2012 package')
            self.imagenet_dir = './imagenet'
            if not os.path.exists(self.imagenet_dir):
                os.mkdir(self.imagenet_dir)
            os.system('tar xvf ' + str(self.input_args.tar_dir) + ' -C ' + self.imagenet_dir)
        elif self.input_args.imagenet_dir is not None:
            self.imagenet_dir = self.input_args.imagenet_dir
        else:
            print('You need to specify the ILSVRC2012 source file path')
        self.fs_dir = './fs_imagenet'
        if not os.path.exists(self.fs_dir):
            os.mkdir(self.fs_dir)
        self.image_resize = self.input_args.image_resize
        self._read_synset_keys()
        self._generate_split()

    def _read_synset_keys(self, path="./synset_words.txt"):
        path = Path(path)
        with open(p, "r") as f:
            self.synset_keys = [line.split(" ")[0] for line in f]
        logging.info(f"Read synset text file with {len(self.synset_keys)} number of keys.")

    def untar(self):
        for idx, keys in enumerate(self.synset_keys):
            print('Untarring ' + keys)
            os.system('tar xvf ' + self.imagenet_dir + '/' + keys + '.tar -C ' + self.fs_dir)
        logging.info('All the tar files are untarred')

    def process_original_files(self):
        self.processed_img_dir = './processed_images'
        split_lists = ['train', 'val', 'test']
        csv_files = ['./csv_files/train.csv','./csv_files/val.csv', './csv_files/test.csv']

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for this_split in split_lists:
            filename = './csv_files/' + this_split + '.csv'
            this_split_dir = self.processed_img_dir + '/' + this_split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)
            with open(filename) as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                next(csv_reader, None)
                images = {}
                print('Reading IDs....')

                for row in tqdm(csv_reader):
                    if row[1] in images.keys():
                        images[row[1]].append(row[0])
                    else:
                        images[row[1]] = [row[0]]

                print('Writing photos....')
                for cls in tqdm(images.keys()):
                    this_cls_dir = this_split_dir + '/' + cls
                    if not os.path.exists(this_cls_dir):
                        os.makedirs(this_cls_dir)

                    lst_files = []
                    for file in glob.glob(self.mini_dir + "/*"+cls+"*"):
                        lst_files.append(file)

                    lst_index = [int(i[i.rfind('_')+1:i.rfind('.')]) for i in lst_files]
                    index_sorted = sorted(range(len(lst_index)), key=lst_index.__getitem__)

                    index_selected = [int(i[i.index('.') - 4:i.index('.')]) for i in images[cls]]
                    selected_images = np.array(index_sorted)[np.array(index_selected) - 1]
                    for i in np.arange(len(selected_images)):
                        if self.image_resize==0:
                            copyfile(lst_files[selected_images[i]],os.path.join(this_cls_dir, images[cls][i]))
                        else:
                            im = cv2.imread(lst_files[selected_images[i]])
                            im_resized = cv2.resize(im, (self.image_resize, self.image_resize), interpolation=cv2.INTER_AREA)
                            cv2.imwrite(os.path.join(this_cls_dir, images[cls][i]),im_resized)

if __name__ == "__main__":
    dataset_generator = MiniImageNetGenerator(args)
    dataset_generator.untar_mini()
    dataset_generator.process_original_files()
