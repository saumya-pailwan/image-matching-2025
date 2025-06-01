"""

Description: This file houses code to preprocess our data.

2025-05-23

"""

import argparse
import numpy as np

import torch.nn.functional as F
import torch

import cv2

import pandas as pd 
from os.path import join
import os

eps = 1.0e-16


def resize_images(images, shapes):

    max_size = [640, 368] # make this 640 x 368
    
    images = np.array([cv2.resize(image, tuple(max_size), interpolation = cv2.INTER_CUBIC) for image in images])

    return images

    

def pad_images(images, shapes):

    max_size = [1042, 774] # make 1042 x 774

    # resizing all those images too big
    sizes = shapes[:, 0] * shapes[:, 1]
    big_image_inds = np.where(sizes > 774 * 1042)[0]

    resized_images = []
    shapes = []

    for i, image in enumerate(images):
        if i in big_image_inds:
            image = cv2.resize(image, tuple(max_size), interpolation = cv2.INTER_CUBIC)

        resized_images.append(image)
        shapes.append([image.shape[0], image.shape[1]])

    images = resized_images
    # resizing all images proportionally to 
    shapes = np.array(shapes)
    resize_shapes = np.ceil(shapes * np.array([[368 / 774, 640 / 1042]])) # getting proportion to resize to

    # resizing images
    images = [cv2.resize(image, shape, interpolation = cv2.INTER_CUBIC) for shape, image in zip(resize_shapes, images)]
    shapes = resize_shapes

    # padding images
    padding = (max_size - shapes) / 2

    top_padding = np.ceil(padding[:, 0]) # top padding is rounded up and bottom is rounded down, same for left vs right
    bottom_padding = np.floor(padding[:, 0])
    left_padding = np.ceil(padding[:, 1])
    right_padding = np.floor(padding[:, 1])

    images = [cv2.copyMakeBorder(image, t, b, l, r, cv2.BORDER_CONSTANT, value = (255, 255, 255)) for image, t, b, l, r in zip(top_padding, bottom_padding, left_padding, right_padding, images)]
    images = np.array(images)


    return images
    


def preprocess_training_data(read_dir, write_dir, collate_fcn = "resizing"):

    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    train_images = []
    train_shapes = []
    train_inds = []

    valid_images = []
    valid_shapes = []
    valid_inds = []

    labels = pd.read_csv(join(os.path.dirname(read_dir), "train_labels.csv"))

    # iterating over training dataset
    for dataset in os.listdir(train_dir):
        dataset_dir = join(train_dir, dataset)

        train_or_valid = np.random.random() > 0.15

        label_inds = np.where(labels["dataset"] == dataset)[0]

        # getting label information to split df
        if train_or_valid == True:
            train_inds.extend(label_inds)
        else:
            valid_inds.extend(label_inds)

        # reading in images
        for image in os.listdir(dataset_dir):
            image_path = join(dataset_dir, image)

            image_arr = plt.imread(image_path) # h x w x 3

            if image_arr.shape[1] / image_arr.shape[0] < 1: # making everything landscape
                image_arr = image_arr.transpose(axes = (0, 1))
            
            if train_or_valid == True:
                train_images.append(image_arr)
                train_shapes.append(image_arr.shape)
            else:
                valid_images.append(image_arr)
                valid_shapes.append(image_arr.shape)


    train_images = np.array(train_images)
    train_shapes = np.array(train_shapes)

    valid_images = np.array(valid_images)
    valid_shapes = np.array(valid_shapes)

    # computing statistics
    train_pixels = train_images.reshape(train_images.shape[0], -1, 3)

    mean = train_images.mean(axis = (0, 1, 2), keepdims = True)
    std = train_images.std(axis = (0, 1, 2), keepdims = True)

    # normalizing images
    normalized_train_images = (train_images - mean) / (std + eps)
    normalized_valid_images = (valid_images - mean) / (std + eps)

    # resizing or padding
    collate_fcns = {
        "resizing": resize_images,
        "padding": pad_images
    }

    normalized_train_images = collate_fcns[collate_fcn](normalized_train_images)
    normalized_valid_images = collate_fcns[collate_fcn](normalized_valid_images)

    # creating labels
    train_labels = labels.iloc(train_inds)
    valid_labels = labels.iloc(valid_inds)

    
    # writing dataset to disk
    write_dataset_dir = join(write_dir, dataset)

    if not os.path.isdir(write_dataset_dir):
        os.makedirs(write_dataset_dir)

    train_labels.to_csv(join(write_dataset_dir, "train_labels.csv"))
    valid_labels.to_csv(join(write_dataset_dir, "valid_labels.csv"))

    np.savez(join(write_dataset_dir, "train.npz"), train = normalized_train_images, valid = normalized_valid_images, mean = mean, std = std)


def preprocess_testing_data(read_dir, write_dir, collate_fcn = "resizing"):

    if not os.path.isdir(write_dir):
        os.makedirs(write_dir)

    images = []
    shapes = []

    labels = pd.read_csv(join(os.path.dirname(read_dir), "train_labels.csv")) # confusing, but yes testing labels are in this file

    # iterating over training dataset
    for dataset in os.listdir(read_dir):
        dataset_dir = join(read_dir, dataset)


        label_inds = np.where(labels["dataset"] == dataset)[0]

        # reading in images
        for image in os.listdir(dataset_dir):
            image_path = join(dataset_dir, image)

            image_arr = plt.imread(image_path) # h x w x 3

            if image_arr.shape[1] / image_arr.shape[0] < 1: # making everything landscape
                image_arr = image_arr.transpose(axes = (0, 1))
            
 
            images.append(image_arr)
            shapes.append(image_arr.shape)



    images = np.array(images)
    shapes = np.array(shapes)

    # loading statistics
    stats = np.load(join(os.path.dirname(read_dir), "processed_train", "train.npz"))
    mean = stats["mean"]
    std = stats["std"]

    # normalizing images
    normalized_images = (images - mean) / (std + eps)

    # resizing or padding
    collate_fcns = {
        "resizing": resize_images,
        "padding": pad_images
    }

    normalized_train_images = collate_fcns[collate_fcn](normalized_images)
    normalized_valid_images = collate_fcns[collate_fcn](normalized_valid_images)

    # creating labels
    test_labels = labels.iloc(inds)
    
    # writing dataset to disk
    write_dataset_dir = join(write_dir, dataset)

    if not os.path.isdir(write_dataset_dir):
        os.makedirs(write_dataset_dir)

    test_labels.to_csv(join(write_dataset_dir, "test_labels.csv"))

    np.savez(join(write_dataset_dir, "test.npz"), test = normalized_images)



def preprocess_data(data_dir, collate_fcn):

    train_dir = join(data_dir, "train")
    processed_dir = join(data_dir, "processed_train")
   
    preprocess_training_data(train_dir, processed_dir, collate_fcn)

    test_dir = join(data_dir, "test")
    processed_dir = join(data_dir, "processed_test")
   
    preprocess_training_data(test_dir, processed_dir, collate_fcn)


if __name__ == "__main__":

    # initializing arparser
    parser = argparse.ArgumentParser()

    # adding arguments
    parser.add_argument(
        "-d", # short hand
        "--data_dir", 
        default = "/home/icip_2025/data/"
        type = str, 
        help = "where the data is downloaded to"
    )

    parser.add_argument(
        "-f", # short hand
        "--collate_fcn", 
        default = "resizing"
        type = str, 
        help = "either resizing or padding"
    )

    args = parser.parse_args()

    preprocess_data(**vars(args))