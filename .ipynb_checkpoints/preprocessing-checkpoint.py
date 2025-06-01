"""

Description: This file houses code to preprocess our data.

2025-05-23

"""

import argparse
from os.path import join
import os

#TODO check unused argument read_dir
def normalize_dataset(read_dir, write_dir):

    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    #TODO use of  undefined variables train_dir, imagenet_mean and imagenet_std
    for dataset in os.listdir(train_dir):
        dataset_dir = join(train_dir, dataset)
        write_dataset_dir = join(write_dir, dataset)

        if not os.path.isdir(write_dataset_dir):
            os.makedirs(write_dataset_dir)

        for image in os.listdir(dataset_dir):
            image_path = join(dataset_dir, image)

            image_arr = plt.imread(image_path)
            norm_im_arr = (image_arr - imagenet_mean) / imagenet_std
            # want to add padding here but need to adjust aspect ratio first
            write_path = join(write_dataset_dir, image)
            plt.imsave(write_dir, norm_im_arr)

def normalize_data(data_dir):

    #TODO check unused variables imagenet_mean, imagenet_std, etc.
    imagenet_mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, ...]
    imagenet_std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, ...]

    train_dir = join(data_dir, "train")
    processed_dir = join(data_dir, "processed_train")
   
    normalize_dataset(train_dir, processed_dir)

    train_dir = join(data_dir, "test")
    processed_dir = join(data_dir, "processed_test")
   
    normalize_dataset(train_dir, processed_dir)

def preprocess_data(data_dir):

    pass

if __name__ == "__main__":

    # initializing arparser
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument(
        "-d", 
        "--data_dir", 
        default = "/home/icip_2025/data/"
        type = str, 
        help = "where the data is downloaded to"
    )

    args = parser.parse_args()

    preprocess_data(args.data_dir)