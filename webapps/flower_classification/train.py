# reference:
#     program uses this methodology then converts model:
#       (new guide) https://www.tensorflow.org/hub/tutorials/image_retraining
#       (old guide) https://hackernoon.com/creating-insanely-fast-image-classifiers-with-mobilenet-in-tensorflow-f030ce0a2991
# this can be used to view node names from console (or you can use tensorboard)
#   [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
# for tensorboard: in .../trained_model/ run
#   tensorboard --logdir .\

import os

image_dir = "D:/plant_disease"
''' possible directories
image_dir = "I:/Rahnemoonfar group/Datasets/plant_disease/"
image_dir = "F:/programming/bina/plant_disease/"
image_dir = "G:/programming/bina/plant_disease/"
'''
output_graph = "./trained_model/output_graph"
intermediate_output_graphs_dir = "./trained_model/intermediate_output_graphs"
intermediate_store_frequency = '0'
output_labels = "./trained_model/output_labels"
summaries_dir = "./trained_model/summaries"
how_many_training_steps = '2000'
learning_rate = '0.0001'
testing_percentage = '20'
validation_percentage = '20'
eval_step_interval = '100'
train_batch_size = '32'
test_batch_size = '-1'
validation_batch_size = '-1'
print_misclassified_test_images = "True"
bottleneck_dir = "./trained_model/bottleneck"
# final_tensor_name = "final_tensor"
flip_left_right = "True"
random_crop = '0'
random_scale = '30'
random_brightness = '30'
tfhub_module = 'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2'
saved_model_dir = "./trained_model/saved_model"
logging_verbosity = 'INFO'

os.system("python trainer/retrain.py"
    + " --image_dir " + image_dir
    + " --output_graph " + output_graph
    + " --intermediate_output_graphs_dir " + intermediate_output_graphs_dir
    + " --intermediate_store_frequency " + intermediate_store_frequency
    + " --output_labels=" + output_labels
    + " --summaries_dir=" + summaries_dir
    + " --how_many_training_steps " + how_many_training_steps
    + " --learning_rate " + learning_rate
    + " --testing_percentage " + testing_percentage
    + " --validation_percentage " + validation_percentage
    + " --eval_step_interval " +  eval_step_interval
    + " --train_batch_size " + train_batch_size
    + " --test_batch_size " + test_batch_size
    + " --validation_batch_size " + validation_batch_size
    + " --print_misclassified_test_images " + print_misclassified_test_images
    + " --bottleneck_dir " + bottleneck_dir
    # + " --final_tensor_name " + final_tensor_name
    + " --flip_left_right " + flip_left_right
    + " --random_crop " + random_crop
    + " --random_scale " + random_scale
    + " --random_brightness " + random_brightness
    + " --tfhub_module " + tfhub_module
    + " --saved_model_dir " + saved_model_dir
    + " --logging_verbosity " + logging_verbosity)

os.system("tensorflowjs_converter --input_format=tf_saved_model --output_node_names=\"final_result\" trained_model/saved_model js_model")
