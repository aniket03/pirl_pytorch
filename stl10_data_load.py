"""
Script obtained from: https://github.com/mttk/STL10/blob/master/stl10_input.py
"""

from __future__ import print_function

import os, sys, tarfile, errno
import numpy as np
from PIL import Image

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './raw_stl10/'
STL10_TRAIN_IMG_DIR = './stl10_data/train/'
STL10_TEST_IMG_DIR = './stl10_data/test/'
STL10_UNLABELLED_IMG_DIR = './stl10_data/unlabelled/'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary files with image data
TRAIN_DATA_PATH = os.path.join(DATA_DIR ,'stl10_binary/train_X.bin')
TEST_DATA_PATH = os.path.join(DATA_DIR , 'stl10_binary/test_X.bin')
UNLABELLED_DATA_PATH = os.path.join(DATA_DIR, 'stl10_binary/unlabeled_X.bin')

# path to the binary files with labels
TRAIN_LABEL_PATH = os.path.join(DATA_DIR, 'stl10_binary/train_y.bin')
TEST_LABEL_PATH = os.path.join(DATA_DIR, 'stl10_binary/test_y.bin')


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def save_image(image, name):
    image = Image.fromarray(image)
    image.save(name + '.jpeg')


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def save_images(images_dir, images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        if labels is not None:
            directory = os.path.join(images_dir, str(labels[i]))
        else:
            directory = images_dir

        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass

        filename = os.path.join(directory, str(i))
        # print(filename)
        save_image(image, filename)
        i = i + 1


if __name__ == "__main__":

    # download data if needed
    download_and_extract()

    # Read train-images and labels and save to disk
    images = read_all_images(TRAIN_DATA_PATH)
    print(images.shape)

    labels = read_labels(TRAIN_LABEL_PATH)
    print(labels.shape)

    save_images(STL10_TRAIN_IMG_DIR, images, labels)

    # Read test-images and labels and save to disk
    images = read_all_images(TEST_DATA_PATH)
    print(images.shape)

    labels = read_labels(TEST_LABEL_PATH)
    print(labels.shape)

    save_images(STL10_TEST_IMG_DIR, images, labels)

    # Read unlabelled images and save to disk
    images = read_all_images(UNLABELLED_DATA_PATH)
    labels = None
    print (images.shape)

    save_images(STL10_UNLABELLED_IMG_DIR, images, labels)
