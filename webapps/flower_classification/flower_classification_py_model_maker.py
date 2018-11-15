# saves an mnist model in the form "SavedModel"
# converted previously using the command:
# tensorflowjs_converter --input_format=tf_saved_model --output_node_names="net_out/BiasAdd" py_model_saves js_model_saves
# use python -m cProfile .\flower_classification_py_model_maker.py to profile

# 38.64% accuracy

import tensorflow as tf
import numpy as np
from flower_dataset import Flower_Dataset

epochs = 20
dataset_dir = "C:/datasets/plant_disease/"
''' possible directories
dataset_dir = "F:/programming/bina/plant_disease/"
dataset_dir = "G:/programming/bina/plant_disease/"
dataset_dir = "I:/Rahnemoonfar group/Datasets/plant_disease/"
'''
batch_size = 200
image_size = (128, 128)
log_level = 2
train_ratio = .7
augmentation_functions = [tf.image.flip_left_right]

dataset = Flower_Dataset(dataset_dir=dataset_dir,
                         batch_size=batch_size,
                         image_size=image_size,
                         train_ratio=train_ratio,
                         augmentation_functions=augmentation_functions)

x = tf.placeholder('float', [None, dataset.size[0], dataset.size[1], 3])
y = tf.placeholder('int64', [None])

net = x
net = tf.reshape(net, [-1, dataset.size[0], dataset.size[1], 3]) # TODO: why am I reshaping it?
net = tf.layers.conv2d(net,
    filters = 16,
    kernel_size = 7,
    padding = 'same',
    activation = tf.nn.relu)
net = tf.layers.conv2d(net,
    filters = 32,
    kernel_size = 7,
    padding = 'same',
    activation = tf.nn.relu)
net = tf.layers.conv2d(net,
    filters = 16,
    kernel_size = 7,
    padding = 'same',
    activation = tf.nn.relu)
net = tf.reshape(net, [-1, dataset.size[0]*dataset.size[1]*16])
net = tf.layers.dense(net,
    units = 64,
    activation = tf.nn.relu)
net = tf.layers.dense(net,
    units = dataset.classes_count, name = "net_out")

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(y, dataset.classes_count), net))

with tf.name_scope('OPTIMIZATION'):
    optimizer = tf.train.AdamOptimizer().minimize(loss)


# train
print("bout to train")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = int(dataset.train_examples["count"]/batch_size)
    for epoch in range(epochs):
        print("EPOCH: ", epoch+1, "/", epochs)
        for step in range(steps):
            print("reading batch...", end='\n')
            data = dataset.get_next_batch('train')
            y_data = data['labels']
            x_data = sess.run(data['examples'])
            x_data = np.stack(x_data)
            print("done")
            o, l = sess.run([optimizer, loss], feed_dict = {x: x_data, y: y_data})
            if log_level > 0:
                print("STEP: ", step+1, "/", steps, " STEP LOSS: ", l)

    correct_c = 0
    test_steps = int(dataset.test_examples["count"]/batch_size)
    for test_steps in range(test_steps):
        data = dataset.get_next_batch('test')
        x_data, y_data = data['examples'], data['labels']
        correct = sess.run(tf.count_nonzero(tf.equal(tf.argmax(net, 1), y)),
                            feed_dict = {x: x_data, y: y_data})
        correct_c += correct
    accuracy = correct_c / dataset.test_examples["count"]
    print("Test Accuracy: ", accuracy)

    # this line shows the names in the nodes, so I can find the node to
    # use as the output in the js model
    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]

    tf.saved_model.simple_save(sess, "py_model_saves", inputs = {"x": x}, outputs = {"net_out": net})
