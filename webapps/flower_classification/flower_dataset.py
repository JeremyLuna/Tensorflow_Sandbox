'''
TODO:
    does get get_intermediate_directories work?
    should I augment test set?
    might have infinite loop in program
    how to handle size parameter
'''

from os import listdir, walk
from os.path import isfile, join
from random import shuffle
import numpy as np
from skimage.transform import resize
from scipy import misc
import copy
import tensorflow as tf

class Flower_Dataset:
    data_dir = ""
    train_ratio = None

    classes = [] # in alphabetical order? todo
    classes_count = 0
    examples_count = 0

    train_examples = {"index": 0,
                      "count": None,
                      "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}
    test_examples = {"index": 0,
                     "count": None,
                     "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}

    size = (100, 100)

    def __init__(self,
                 data_dir,
                 train_ratio, # .7
                 augmentation_functions):
        self.data_dir = data_dir
        self.train_ratio = train_ratio

        # get list of all image file paths
        print("initializing dataset")
        paths = []
        print("getting paths")
        for (dirpath, dirnames, filenames) in walk(self.data_dir):
            paths += map(lambda a: dirpath + "/" + a, filenames)
        self.examples_count = len(paths)
        if self.examples_count == 0:
            print("incorrect dataset path")
            exit()
        # only use forward slashes
        print("normalizing paths")
        paths = list(map(Flower_Dataset.normalize_path, paths))

        # returns ["a/s/d/f"] from ["data_dir/a/s/d/f/img.png"]
        print("getting classes")
        self.classes = list(set(map(lambda p: self.get_intermediate_directories(p), paths)))
        self.classes_count = len(self.classes)

        # shuffle paths
        print("shuffling")
        shuffle(paths)

        # divide up paths between training and testing
        print("making train and test sets")
        self.train_examples["count"] = int(self.train_ratio * self.examples_count)
        self.test_examples["count"] = self.examples_count - self.train_examples["count"]
        for path in paths[:self.train_examples["count"]]:
            self.train_examples["example_info"].append({"path": path, "augmentation_functions": []})
        for path in paths[self.train_examples["count"]:]:
            self.test_examples["example_info"].append({"path": path, "augmentation_functions": []})

        print("record augmentations")
        for augmentation_function in augmentation_functions:
            a = copy.deepcopy(self.train_examples["example_info"])
            for e in a:
                e["augmentation_functions"].append(augmentation_function)
            self.train_examples["example_info"] += a

        print("shuffling")
        shuffle(self.train_examples["example_info"])
        print("finished initializing dataset")

    def get_next_batch(self,
                       data_stream, # self.train_examples or self.test_examples
                       examples):   # number of examples in the batch
        print("getting next batch")
        batch_examples = {'examples': [], 'labels': []}
        diff = (data_stream["index"] + examples) - data_stream["count"]
        if diff <= 0:
            indexes_to_use = range(data_stream["index"], data_stream["index"] + examples)
        else:
            indexes_to_use = list(range(data_stream["index"], data_stream["count"])) + list(range(diff))
        for example_index in indexes_to_use:
            try:
                print("reading")
                # im = misc.imread(data_stream["example_info"][example_index]["path"], mode='RGB')
                im = tf.image.decode_image(data_stream["example_info"][example_index]["path"], channels=3, dtype=tf.float32)
                print(im)
                input()
                print("resizing")
                im = resize(im/255, self.size) # TODO use tf.image.crop_and_resize
                print("augmenting")
                for augmentation_function in data_stream["example_info"][example_index]["augmentation_functions"]:
                    im = augmentation_function(im)
                batch_examples['examples'].append(im)
                label = self.get_intermediate_directories(data_stream["example_info"][example_index]["path"])
                batch_examples['labels'].append(self.classes.index(label))
            except Exception as e:
                print(e)
                print("unreadable example: " + data_stream["example_info"][example_index]["path"])
        data_stream["index"] = example_index+1
        print("got batch")
        return batch_examples

    def get_intermediate_directories(self, path):
        return '/'.join(path[len(self.data_dir):].split('/')[:-1])

    def normalize_path(path):
        return path.replace('\\', '/')
