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
import csv
import glob
import logging
import os
from os import listdir
from pathlib import Path
from shutil import copyfile

import cv2
import numpy as np
from tqdm import tqdm

# argument parser
parser = argparse.ArgumentParser(description="")
parser.add_argument("--tar_dir", type=str)
parser.add_argument("--imagenet_dir", type=str)
parser.add_argument("--image_resize", type=int, default=84)
parser.add_argument("--num_classes", type=int, default=1000)
parser.add_argument("--num_instances_per_class", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()

rng = np.random.default_rng(args.seed)


class FSImageNetGenerator(object):
    def __init__(self, input_args):
        self.tar_dir = input_args.tar_dir
        self.imagenet_dir = input_args.imagenet_dir
        self.image_resize = input_args.image_resize
        self.num_classes = input_args.num_classes
        self.num_instances_per_class = input_args.num_instances_per_class
        self.input_args = input_args
        if self.tar_dir is not None:
            print("Untarring ILSVRC2012 package")
            self.imagenet_dir = "./imagenet"
            if not os.path.exists(self.imagenet_dir):
                os.mkdir(self.imagenet_dir)
            os.system(
                "tar xvf " + str(self.tar_dir) + " -C " + self.imagenet_dir
            )
        elif self.imagenet_dir is not None:
            self.imagenet_dir = self.imagenet_dir
        else:
            logging.info("You need to specify the ILSVRC2012 source file path")
        self.fs_dir = "./fs_imagenet"
        if not os.path.exists(self.fs_dir):
            os.mkdir(self.fs_dir)
        self.image_resize = self.image_resize
        self._read_synset_keys()
        self._generate_split()

    def _read_synset_keys(self):
        path = Path("./synset_keys.txt")
        with open(path, "r") as f:
            self.synset_keys = np.array([line.split(" ")[0] for line in f])
        logging.info(
            f"Read synset text file with {len(self.synset_keys)} number of keys."
        )

    def _generate_split(self):
        """Split the classes randomly into train / val / test"""
        # Split is 64 / 16 / 20
        rng.shuffle(self.synset_keys)
        n = self.num_classes

        self.train_synset_keys = self.synset_keys[: int(0.64 * n)]
        self.valid_synset_keys = self.synset_keys[int(0.64 * n) : int(0.80 * n)]
        self.test_synset_keys = self.synset_keys[int(0.80 * n) :]
        logging.info("Generated split")

    def untar(self):
        for key in self.synset_keys:
            logging.info("Untarring " + key)
            os.system(
                "tar xvf " + self.imagenet_dir + "/" + key + ".tar -C " + self.fs_dir
            )
        logging.info("All the tar files are untarred")

    def process_original_files(self):
        self.processed_img_dir = "./processed_images"
        split_lists = ["train", "valid", "test"]

        if not os.path.exists(self.processed_img_dir):
            os.makedirs(self.processed_img_dir)

        for split_synset_keys, split in zip(
            [self.train_synset_keys, self.valid_synset_keys, self.test_synset_keys],
            split_lists,
        ):
            this_split_dir = self.processed_img_dir + "/" + split
            if not os.path.exists(this_split_dir):
                os.makedirs(this_split_dir)

            print("Writing photos....")
            for cls in tqdm(split_synset_keys):
                this_cls_dir = this_split_dir + "/" + cls
                if not os.path.exists(this_cls_dir):
                    os.makedirs(this_cls_dir)

                cls_instance_files = np.array(list(glob.glob(self.fs_dir + "/*" + cls + "*")))[:self.num_instances_per_class]
                # Randomize what is kept
                rng.shuffle(cls_instance_files)
                cls_instance_index = np.array([
                    int(filename.split("_")[1].split(".")[0]) for filename in cls_instance_files
                ])

                for image_file in cls_instance_files:
                    if self.image_resize == 0:
                        copyfile(
                            image_file,
                            os.path.join(this_cls_dir, image_file),
                        )
                    else:
                        im = cv2.imread(image_file)
                        im_resized = cv2.resize(
                            im,
                            (self.image_resize, self.image_resize),
                            interpolation=cv2.INTER_AREA,
                        )
                        cv2.imwrite(
                            os.path.join(this_cls_dir, image_file), im_resized
                        )


if __name__ == "__main__":
    dataset_generator = FSImageNetGenerator(args)
    dataset_generator.untar()
    dataset_generator.process_original_files()
