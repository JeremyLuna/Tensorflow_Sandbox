# saves an mnist model in the form "SavedModel"
# converted previously using the command:
# tensorflowjs_converter --input_format=tf_saved_model --output_node_names="net_out/BiasAdd" py_model_saves js_model_saves
# use python -m cProfile .\flower_classification_py_model_maker.py to profile

# 38.64% accuracy

import tensorflow as tf
import numpy as np
from flower_dataset import Flower_Dataset
from scipy import misc

batch_size = 100
epochs = 20
log_level = 2
'''
dataset = Flower_Dataset("I:/Rahnemoonfar group/Datasets/plant_disease/",
dataset = Flower_Dataset("F:/programming/bina/plant_disease/",
dataset = Flower_Dataset("G:/programming/bina/plant_disease/",
'''
dataset = Flower_Dataset("C:/datasets/plant_disease/",
                         .7,
                         [np.fliplr])

x = tf.placeholder('float', [None, 100, 100, 3])
y = tf.placeholder('int64', [None])

net = x
net = tf.reshape(net, [-1, 100, 100, 3]) # TODO: why am I reshaping it?
net = tf.layers.conv2d(net,
    filters = 8,
    kernel_size = 7,
    padding = 'same',
    activation = tf.nn.relu)
net = tf.reshape(net, [-1, 100*100*8])
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
            data = dataset.get_next_batch(dataset.train_examples, batch_size)
            x_data, y_data = data['examples'], data['labels']
            o, l = sess.run([optimizer, loss], feed_dict = {x: x_data, y: y_data})
            if log_level > 0:
                print("STEP: ", step+1, "/", steps, " STEP LOSS: ", l)

    correct_c = 0
    test_steps = int(dataset.test_examples["count"]/batch_size)
    for test_steps in range(test_steps):
        data = dataset.get_next_batch(dataset.test_examples, batch_size)
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
