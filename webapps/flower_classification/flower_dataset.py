# labels are yeilded by path
# TODO: shuffle

from os import listdir
from os.path import isfile, join
from random import shuffle
import numpy as np
from skimage.transform import resize
from scipy import misc

size = (100, 100)
train_ratio = .7

class Flower_Dataset:
    data_dir = ""

    classes = [] # in alphabetical order
    classes_count = 0

    data_count = 0
    train_data_count = 0
    test_data_count = 0

    train_data_paths = []
    test_data_paths = []

    train_data_index = 0
    test_data_index = 0

    def __init__(self):
        self.init()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.init()

    def init(self):
        self.classes = listdir(self.data_dir)
        self.classes_count = len(self.classes)
        for a_class in self.classes:
            example_paths = listdir(self.data_dir + '/' + a_class + '/')
            example_count = len(example_paths)
            train_example_count = int(train_ratio * example_count)
            test_example_count = example_count - train_example_count
            for file_name_index in range(int(train_example_count)):
                self.train_data_paths.append(a_class + "/" + example_paths[file_name_index])
            for file_name_index in range(train_example_count, example_count):
                self.test_data_paths.append(a_class + "/" + example_paths[file_name_index])
        # shuffle
        self.train_data_count = len(self.train_data_paths)
        self.test_data_count = len(self.test_data_paths)
        shuffle(self.train_data_paths)
        shuffle(self.test_data_paths)

    def get_next_train_batch(self, examples):
        data = {'examples': [], 'labels': []}
        diff = (self.train_data_index + examples) - self.train_data_count
        if diff <= 0:
            indexes_to_use = range(self.train_data_index, self.train_data_index + examples)
        else:
            indexes_to_use = list(range(self.train_data_index, self.train_data_count)) + list(range(diff))
        for example in indexes_to_use:
            if (self.train_data_paths[example] != 'mosaic/23mosaicvirus4.jpg'):
                im = misc.imread(self.data_dir + '/' + self.train_data_paths[example], mode='RGB')
                im = resize(im/255, size) #getdata, putdata
                data['examples'].append(im)
                data['labels'].append(self.classes.index(self.train_data_paths[example].split('/')[0]))
            self.train_data_index = example
        return data

    def get_next_test_batch(self, examples):
        data = {'examples': [], 'labels': []}
        diff = (self.test_data_index + examples) - self.test_data_count
        if diff <= 0:
            indexes_to_use = range(self.test_data_index, self.test_data_index + examples)
        else:
            indexes_to_use = list(range(self.test_data_index, self.test_data_count)) + list(range(diff))
        for example in indexes_to_use:
            if (self.test_data_paths[example] != 'mosaic/23mosaicvirus4.jpg'):
                im = misc.imread(self.data_dir + '/' + self.test_data_paths[example], mode='RGB')
                im = resize(im/255, size)
                data['examples'].append(im)
                data['labels'].append(self.classes.index(self.test_data_paths[example].split('/')[0]))
            self.test_data_index = example
        return data
