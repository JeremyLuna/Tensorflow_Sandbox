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


class Flower_Dataset:
    data_dir = ""

    classes = [] # in alphabetical order? todo
    classes_count = 0
    examples_count = 0

    train_examples = {"index": None,
                      "count": None,
                      "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}
    test_examples = {"index": None,
                     "count": None,
                     "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}

    size = (100, 100)

    def __init__(self,
                 data_dir,
                 train_ratio, # .7
                 augmentation_functions):
        self.data_dir = data_dir

        # get list of all image file paths
        paths = []
        for (dirpath, dirnames, filenames) in walk(self.data_dir):
            print(dirpath)
            paths += map(lambda a: dirpath + "/" + a, filenames)
        examples_count = len(paths)
        if example_count == 0:
            print("incorrect dataset path")

        # returns ["a/s/d/f"] from ["data_dir/a/s/d/f/img.png"]
        self.classes = list(set(map(lambda p: self.get_intermediate_directories(p), paths)))
        self.classes_count = len(self.classes)

        # shuffle
        shuffle(paths)

        # divide up paths between training and testing
        self.train_examples["count"] = int(self.train_ratio * self.examples_count)
        self.test_examples["count"] = self.examples_count - self.train_examples["count"]
        for path in paths[:self.train_examples["count"]]:
            self.train_examples["example_info"].append({"path": path, "augmentations": []})
        for path in paths[self.train_examples["count"]:]:
            self.test_examples["example_info"].append({"path": path, "augmentations": []})

        # record augmentations
        for augmentation_function in augmentation_functions:
            for example_info in self.train_examples["example_info"]:
                example_info["augmentations"].append(augmentation_function) # TODO: will this inf loop?

    def get_next_batch(data_stream, # self.train_examples or self.test_examples
                       examples):   # number of examples in the batch
        batch_examples = {'examples': [], 'labels': []}
        diff = (data_stream["index"] + examples) - data_stream["count"]
        if diff <= 0:
            indexes_to_use = range(data_stream["index"], data_stream["index"] + examples)
        else:
            indexes_to_use = list(range(data_stream["index"], data_stream["count"])) + list(range(diff))
        for example_index in indexes_to_use:
            try:
                im = misc.imread(self.train_examples["example_info"]["path"], mode='RGB')
                im = resize(im/255, size)
                for augmentation_function in self.train_examples["example_info"]["augmentation_functions"]
                    im = augmentation_function(im)
                batch_examples['examples'].append(im)
                label = self.get_intermediate_directories(self.train_examples["example_info"]["path"])
                batch_examples['labels'].append(this.classes.index(label))
            except:
                print("unreadable example: " + self.train_data_paths[example_index])
        self.train_examples["index"] = example_index+1
        return batch_examples

    def get_intermediate_directories(self, path):
        return path[len(self.data_dir):].split('/')[:-1].join('/')
