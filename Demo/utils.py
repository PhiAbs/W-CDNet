import os
import csv
import cv2
import numpy as np
import pandas as pd
from sklearn.utils import class_weight
import imgaug as ia
from imgaug import augmenters as iaa
import random
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input


def load_images_minibatch(input_paths, input_size):
    """
    Load images for each minibatch individually and resize them

    Args:
    - input_paths: paths to images of current minibatch
    - input_size: image size

    Returns:
    - input_images: images
    """
    input_images = list(map(
        lambda x: image.load_img(x, color_mode='rgb', target_size=input_size),
        input_paths
    ))
    input_images = np.array(list(map(
        lambda x: image.img_to_array(x),
        input_images
    )))

    return input_images


def image_augmentation(images):
    """Augment images during training.

    Args: 
    - images: images to be augmented

    Returns:
    - images_augmented: well, the augmented images
    """
    # inspired by https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
    def sometimes(aug): return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential([
        # apply the following augmenters to most images
        sometimes(iaa.Affine(
            # scale images to 98-102% of their size, individually per axis
            scale={"x": (0.98, 1.02), "y": (0.98, 1.02)},
            # translate by -1 to +1 percent (per axis)
            translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
            # rotate by -1 to +1 degrees
            rotate=(-1, 1),
        )),
        iaa.SomeOf((0, 3), [
            iaa.OneOf([
                iaa.GaussianBlur((0, 1.0)),
                # blur images with a sigma between 0 and 3.0
                iaa.AverageBlur(k=(3, 5)),
                # blur image using local means with kernel sizes between 2 and 7
                iaa.MedianBlur(k=(3, 5)),
                # blur image using local medians with kernel sizes between 2 and 7
            ]),
            iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),
            # sharpen images,
            iaa.AdditiveGaussianNoise(loc=0,
                                      scale=(0.0, 0.01 * 255),
                                      per_channel=0.5),
            # add gaussian noise to images
            iaa.OneOf([
                iaa.Dropout((0.01, 0.05), per_channel=0.5),
                # randomly remove up to 10% of the pixels
                iaa.CoarseDropout((0.01, 0.03),
                                  size_percent=(0.01, 0.02),
                                  per_channel=0.2),
            ]),
            iaa.Add((-20, 20), per_channel=0.5),
            sometimes(iaa.ElasticTransformation(alpha=(0.5, 2),
                                                sigma=0.5)),
            # move pixels locally around (with random strengths)
            sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
            # sometimes move parts of the image around
            sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
            # sometimes cut out part of the image
            sometimes(iaa.Cutout(nb_iterations=(0, 3),
                                 size=(0.1, 0.3),
                                 squared=False,
                                 cval=(0, 255),
                                 fill_per_channel=0.5)),
        ],
            random_order=True)
    ],
        random_order=True)

    images_augmented = seq.augment_images(images)

    return images_augmented


def image_augmentation_for_both_images(images_curr, images_prev):
    """Apply the same transformation to both images in order to keep them aligned

    """
    # inspired by https://www.kaggle.com/mpalermo/keras-pipeline-custom-generator-imgaug
    # apply up-down flip in 1/4 of all images
    # apply left-right flip in 1/4 of all images
    if random.randint(0, 3) == 0:
        images_curr = iaa.flip.fliplr(images_curr)
        images_prev = iaa.flip.fliplr(images_prev)

    if random.randint(0, 3) == 0:
        images_curr = iaa.flip.flipud(images_curr)
        images_prev = iaa.flip.flipud(images_prev)

    return images_curr, images_prev


def image_augmentation_add_border(images_curr, images_prev):
    """Mirror the image edges. This should make it easier to detect objects close to the border.
    directly resize the images to the input shape!

    Args:
    - images_curr: batch containing the current images. shape: [batch_size, rows, cols, channels]
    - images_prev: batch containing the previous images. shape: [batch_size, rows, cols, channels]
    """
    border_size = 10
    image_size = (np.shape(images_curr)[2], np.shape(images_curr)[1])
    batch_size = np.shape(images_curr)[0]

    for idx in range(batch_size):
        images_curr[idx, :, :, :] = cv2.resize(cv2.copyMakeBorder(
            images_curr[idx, :, :, :], border_size, border_size, border_size, border_size, borderType=cv2.BORDER_REFLECT), image_size)
        images_prev[idx, :, :, :] = cv2.resize(cv2.copyMakeBorder(
            images_prev[idx, :, :, :], border_size, border_size, border_size, border_size, borderType=cv2.BORDER_REFLECT), image_size)

    return images_curr, images_prev


def load_data_for_aicd(classes_file, events_list, dataset_root):
    """Load classes, labels, image ids, etc. (Data needed for training)

    Args:
    - classes_file: csv file name (contains class information)
    - events_list: csv file containing event information
    - dataset_root: path to folder containing images
    """
    # read data from csv files
    classes_df = pd.read_csv(classes_file)
    classes = classes_df['label_ids_extended'].tolist()
    class_descriptions = classes_df['description'].tolist()
    num_classes = len(classes)
    print('number of classes: ' + str(num_classes))

    image_ids_current_paths = []
    image_ids_previous_paths = []
    label_ids_extended = []
    label_ids_extended_lists = []

    # load image names
    with open(events_list, 'r') as csv_file:
        reader = csv.DictReader(csv_file, delimiter=',')

        unusable_images_counter = 0

        for idx, row in enumerate(reader):
            image_id_current_path = os.path.join(
                dataset_root, str(row['image_ids_current']))
            image_id_previous_path = os.path.join(
                dataset_root, str(row['image_ids_previous']))

            # check if the image id paths actually link to an image and if the class label is available
            # If it does, we store the path
            # also check image size.
            if os.path.isfile(image_id_current_path) \
                    and os.path.isfile(image_id_previous_path)\
                    and os.path.getsize(image_id_current_path) > 10\
                    and os.path.getsize(image_id_previous_path) > 10:

                image_ids_current_paths.append(image_id_current_path)
                image_ids_previous_paths.append(image_id_previous_path)
                label_ids_extended.append(row['label_ids_extended'])

            else:
                unusable_images_counter += 1
                print('faulty image: Will not train current image id')
                print(str(image_id_current_path))

        print('total unusable images: ' + str(unusable_images_counter))

    soft_multi_hot_labels = []
    label_list_for_class_balancing = []

    # label_ids_extended are actually lists in the form of
    # strings. convert them to lists and make soft multi-hot vectors out of
    # them by using the 'classes' vector as output array
    for idx, _ in enumerate(label_ids_extended):
        # convert string to list
        label_ids_list = label_ids_extended[idx].split()

        # append the list for the current trashevent to the list of all trashevents
        label_ids_extended_lists.append(label_ids_list)

        # soft multi-hot vector
        soft_multi_hot_label = np.zeros((num_classes,))

        for k, label_id in enumerate(label_ids_list):
            label_index = classes.index(int(label_id))
            soft_multi_hot_label[label_index] = 1

            # append all label ids to a loooong list which can be used to compute class weights (for balancing)
            label_list_for_class_balancing.append(int(label_id))

        soft_multi_hot_labels.append(soft_multi_hot_label)

    # compute class weights in order to make up for underrepresented classes!
    try:
        class_weights = class_weight.compute_class_weight(
            'balanced', np.array(classes), np.array(label_list_for_class_balancing))
    except:
        print('could not compute class weights')
        class_weights = np.array([1, 1, 1])
    class_weights = dict(enumerate(class_weights))

    return(
        np.array(soft_multi_hot_labels),
        np.array(image_ids_current_paths),
        np.array(image_ids_previous_paths),
        np.array(classes),
        class_descriptions,
        num_classes,
        class_weights,
        np.array(label_ids_extended_lists),
    )


class CustomDataGenerator(Sequence):
    def __init__(
            self,
            input_paths_current,
            input_paths_previous,
            labels,
            batch_size,
            input_size
    ):
        self.input_paths_current = input_paths_current
        self.input_paths_previous = input_paths_previous
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size

    def __len__(self):
        return (np.ceil(len(self.input_paths_current) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        self.input_images_previous = load_images_minibatch(
            self.input_paths_previous[(idx * self.batch_size):((idx+1) * self.batch_size)], self.input_size)

        self.input_images_current = load_images_minibatch(
            self.input_paths_current[(idx * self.batch_size):((idx+1) * self.batch_size)], self.input_size)

        # add border around image (output has same shape as input!!)
        self.input_images_current, self.input_images_previous = image_augmentation_add_border(
            self.input_images_current, self.input_images_previous)

        # augment the images
        # TODO turn on image augmentation again!!
        self.input_images_previous = image_augmentation(
            self.input_images_previous)
        self.input_images_current = image_augmentation(
            self.input_images_current)

        # apply flips to both the current and previous image simultaneously
        self.input_images_current, self.input_images_previous = image_augmentation_for_both_images(
            self.input_images_current, self.input_images_previous)

        self.minibatch_data = [vgg16_preprocess_input(
            self.input_images_previous), vgg16_preprocess_input(self.input_images_current)]

        return self.minibatch_data, self.labels[(idx * self.batch_size):((idx+1) * self.batch_size)]


class CustomDataGeneratorForValidation(Sequence):
    def __init__(
            self,
            input_paths_current,
            input_paths_previous,
            labels,
            batch_size,
            input_size
    ):
        self.input_paths_current = input_paths_current
        self.input_paths_previous = input_paths_previous
        self.labels = labels
        self.batch_size = batch_size
        self.input_size = input_size

    def __len__(self):
        return (np.ceil(len(self.input_paths_current) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        self.input_images_previous = load_images_minibatch(
            self.input_paths_previous[(idx * self.batch_size):((idx+1) * self.batch_size)], self.input_size)

        self.input_images_current = load_images_minibatch(
            self.input_paths_current[(idx * self.batch_size):((idx+1) * self.batch_size)], self.input_size)

        # add border around image (output has same shape as input!!)
        self.input_images_current, self.input_images_previous = image_augmentation_add_border(
            self.input_images_current, self.input_images_previous)

        self.minibatch_data = [vgg16_preprocess_input(
            self.input_images_previous), vgg16_preprocess_input(self.input_images_current)]

        return self.minibatch_data, self.labels[(idx * self.batch_size):((idx+1) * self.batch_size)]
