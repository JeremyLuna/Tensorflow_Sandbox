# reference:
#     Use this methodology then convert:
#       (new guide) https://www.tensorflow.org/hub/tutorials/image_retraining
#       (old guide) https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md
#     https://github.com/Zehaos/MobileNet
#     https://www.tensorflow.org/js/tutorials/conversion/import_saved_model
#     https://codelabs.developers.google.com/codelabs/tensorflowjs-teachablemachine-codelab/index.html
# model taken from
#     tensorflowjs_converter --input_format=tf_hub 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' ./mobilenet/web_model

import os

image_dir = "I:/Rahnemoonfar group/Datasets/plant_disease/"
''' possible directories
dataset_dir = "C:/datasets/plant_disease/"
dataset_dir = "F:/programming/bina/plant_disease/"
dataset_dir = "G:/programming/bina/plant_disease/"
'''

learning_rate = 0.0001
testing_percentage = 20
validation_percentage = 20
train_batch_size = 32
validation_batch_size = -1
flip_left_right = "True"
random_scale = 30
random_brightness = 30
eval_step_interval = 100
how_many_training_steps = 600
architecture = "mobilenet_1.0_224"


os.system("python tensorflow/examples/image_retraining/retrain.py"
    + " --image_dir " + image_dir
    + " --learning_rate=" + learning_rate
    + " --testing_percentage=" + testing_percentage
    + " --validation_percentage=" + validation_percentage
    + " --train_batch_size=" + train_batch_size
    + " --validation_batch_size=" + validation_batch_size
    + " --flip_left_right " + flip_left_right
    + " --random_scale=" + random_scale
    + " --random_brightness=" + random_brightness
    + " --eval_step_interval=" +  eval_step_interval
    + " --how_many_training_steps=" + how_many_training_steps
    + " --architecture " + architecture)
