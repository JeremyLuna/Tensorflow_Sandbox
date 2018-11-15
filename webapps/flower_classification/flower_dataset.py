'''
TODO:
    "23mosaicvirus4.jpeg" this file is BROKE
    no overloading?
    how to automate size parameter
    numpy.save the dataset...
'''
import os
from os import walk
from random import shuffle
import copy
import tensorflow as tf
import pickle
import numpy as np

class Flower_Dataset:
    dataset_config = {"dataset_dir": None,
        "batch_size": None,
        "image_size": None,
        "train_ratio": None,
        "augmentation_functions": None,
        "np_dataset_dir": None, # directory for serialized dataset
        "classes": [],
        "classes_count": 0,
        "source_examples_count": 0,
        "train_examples_count": 0,
        "test_examples_count": 0,
        'batches_count': {'train':0, 'test':0}}

    # object variables
    batch_index = {'train': 0, 'test': 0}

    def __init__(self,
                 dataset_dir,
                 batch_size,
                 image_size,
                 train_ratio, # .7
                 augmentation_functions):

        print("configuring dataset")
        self.dataset_config = {"dataset_dir": dataset_dir,
            "batch_size": batch_size,
            "image_size": image_size,
            "train_ratio": train_ratio,
            "augmentation_functions": augmentation_functions,
            "np_dataset_dir": None,
            "classes": [],
            "classes_count": 0,
            "source_examples_count": 0,
            "train_examples_count": 0,
            "test_examples_count": 0,
            'batches_count': {'train':0, 'test':0}}

        # get list of all image file paths
        paths = []
        for (dirpath, dirnames, filenames) in walk(dataset_dir):
            paths += map(lambda a: dirpath + "/" + a, filenames)
        self.dataset_config['source_examples_count'] = len(paths)
        if self.dataset_config['source_examples_count'] == 0:
            print("incorrect dataset path")
            exit()
        # only use forward slashes
        paths = list(map(os.path.normpath, paths))
        # returns ["a/s/d/f"] from ["dataset_dir/a/s/d/f/img.png"]
        self.dataset_config["classes"] = list(set(map(lambda p: self.get_intermediate_directories(p), paths)))
        self.dataset_config["classes_count"] = len(self.dataset_config["classes"])
        self.dataset_config["classes"].sort() # classes change order between runs without this

        # shuffle paths
        shuffle(paths)
        train_examples = {"index": 0,
        "count": None,
        "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}
        test_examples = {"index": 0,
        "count": None,
        "example_info": []} # should be list of {"path": None, "augmentations": [functions to apply to it to augment it]}
        # divide up paths between training and testing
        train_examples_count = int(train_ratio * self.dataset_config['source_examples_count'])
        for path in paths[:train_examples_count]:
            train_examples["example_info"].append({"path": path, "augmentation_functions": []})
        for path in paths[train_examples_count:]:
            test_examples["example_info"].append({"path": path, "augmentation_functions": []})

        for augmentation_function in augmentation_functions:
            a = copy.deepcopy(train_examples["example_info"])
            for e in a:
                e["augmentation_functions"].append(augmentation_function)
            train_examples["example_info"] += a

        # record augmented amount
        train_examples['count'] = len(train_examples["example_info"])
        self.dataset_config['train_examples_count'] = train_examples['count']
        test_examples['count'] = len(test_examples["example_info"])
        self.dataset_config['test_examples_count'] = test_examples['count']
        shuffle(train_examples["example_info"])

        # check if dataset is made
        up = os.path.dirname
        self.dataset_config['np_dataset_dir'] = up(os.path.normpath(dataset_dir)) + "/serialized"
        np_dataset_dir = self.dataset_config['np_dataset_dir']
        try:
            if pickle.load(open(np_dataset_dir + "/dataset_config.pkl", 'rb')) == self.dataset_config:
                print('serialized dataset found')
                return
        except:
            pass

        print('serializing dataset')
        train_batches_count = int(self.dataset_config['train_examples_count']/batch_size)
        self.dataset_config['batches_count']['train'] = train_batches_count
        for batch_number in range(train_batches_count):
            batch = self.get_next_batch(train_examples, batch_size)
            np.save(self.dataset_config['np_dataset_dir']+"train_images_"+batch_number+".npy",batch['examples'], allow_pickle=True)
            np.save(self.dataset_config['np_dataset_dir']+"train_labels_"+batch_number+".npy",batch['labels'], allow_pickle=True)
        test_batches_count = int(self.dataset_config['test_examples_count']/batch_size)
        self.dataset_config['batches_count']['test'] = test_batches_count
        for batch_number in range(test_batches_count):
            batch = self.get_next_batch(test_examples, batch_size)
            np.save(self.dataset_config['np_dataset_dir']+"test_images_"+batch_number+".npy",batch['examples'], allow_pickle=True)
            np.save(self.dataset_config['np_dataset_dir']+"test_labels_"+batch_number+".npy",batch['labels'], allow_pickle=True)

        pickle.dump(self.dataset_config, open(np_dataset_dir + "/dataset_config.pkl", 'wb'))
        print('finished initializing dataset')

    def get_next_batch(self,
                       data_stream, # self.train_examples or self.test_examples
                       examples):   # number of examples in the batch
        batch_examples = {'examples': [], 'labels': []}
        diff = (data_stream["index"] + examples) - data_stream["count"]
        if diff <= 0:
            indexes_to_use = range(data_stream["index"], data_stream["index"] + examples)
        else:
            indexes_to_use = list(range(data_stream["index"], data_stream["count"])) + list(range(diff))
        for example_index in indexes_to_use:
            example_path = data_stream["example_info"][example_index]["path"]
            im = tf.read_file(example_path)
            # do not use the generalized function tf.image.decode_images. It
            # does not return a tensor with a shape, which is needed for tf.image.resize_images
            im = tf.cond(tf.image.is_jpeg(im),
                lambda: tf.image.decode_jpeg(im, channels=3),
                lambda: tf.image.decode_png(im, channels=3))
            im = tf.image.central_crop(im, central_fraction=1)
            image_size = self.dataset_config['image_size']
            im = tf.image.resize_images(im, image_size)
            for augmentation_function in data_stream["example_info"][example_index]["augmentation_functions"]:
                im = augmentation_function(im)
            batch_examples['examples'].append(im/255)
            label = self.get_intermediate_directories(data_stream["example_info"][example_index]["path"])
            batch_examples['labels'].append(self.dataset_config['classes'].index(label))
            # np.stack(sess.run())
        data_stream["index"] = example_index+1
        return batch_examples

    def get_next_batch(self, mode): # mode = "train" | "test"
        mode = '/'+mode
        data_dir = self.dataset_config['np_dataset_dir''train_images_'+batch_number+'.npy']
        x = np.load(data_dir+mode+"_images_"+self.batch_index[mode]+".npy")
        y = np.load(data_dir+mode+"_labels_"+self.batch_index[mode]+".npy")
        self.batch_index[mode] += 1
        self.batch_index[mode] %= self.dataset_config['batches_count'][mode]

    def get_intermediate_directories(self, path):
        return '/'.join(path[len(self.dataset_config['dataset_dir']):].split('/')[:-1])
