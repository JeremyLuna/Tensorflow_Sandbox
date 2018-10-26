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

    train_examples = {"index": None,
                      "example_info": None}
    test_examples = {"index": None,
                     "example_info": None}

    def __init__(self, data_dir, portion_for_training, aug_mirror_x):
        self.data_dir = data_dir
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

    def get_next_batch(data_stream, # self.train_examples or self.test_examples
                       examples):   # number of examples in the batch
        batch_examples = {'examples': [], 'labels': []}
        diff = (data_stream["index"] + examples) - len(data_stream["example_info"])
        if diff <= 0:
            indexes_to_use = range(data_stream["index"], data_stream["index"] + examples)
        else:
            indexes_to_use = list(range(data_stream["index"], len(data_stream["example_info"]))) + list(range(diff))
        for example in indexes_to_use:
            try:
                im = misc.imread(self.data_dir + '/' + self.train_data_paths[example], mode='RGB')
                im = resize(im/255, size) # getdata, putdata
                batch_examples['examples'].append(im)
                batch_examples['labels'].append(self.classes.index(self.train_data_paths[example].split('/')[0]))
            except:
                print("unreadable example: " + self.train_data_paths[example])
        self.train_data_index = example+1
        return batch_examples
