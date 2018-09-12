import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 1000
epochs = 2
log_level = 2
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

net = x
net = tf.reshape(net, [-1, 28, 28, 1])
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
    filters = 8,
    kernel_size = 7,
    padding = 'same',
    activation = tf.nn.relu)
net = tf.reshape(net, [-1, 8*28*28])
net = tf.layers.dense(net,
    units = 64,
    activation = tf.nn.relu)
net = tf.layers.dense(net,
    units = 10)

loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(y, net))

with tf.name_scope('OPTIMIZATION'):
    optimizer = tf.train.AdamOptimizer().minimize(loss)

#train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    steps = int(mnist.train.num_examples/batch_size)
    for epoch in range(epochs):
        print("EPOCH: ", epoch+1, "/", epochs)
        for step in range(steps):
            x_data, y_data = mnist.train.next_batch(batch_size)
            o, l = sess.run([optimizer, loss], feed_dict = {x: x_data, y: y_data})
            if log_level > 0:
                print("STEP: ", step+1, "/", steps, " STEP LOSS: ", l)
    print(end-start, " time elapsed")

    correct_c = 0
    test_steps = int(mnist.test.num_examples/batch_size)
    for test_steps in range(test_steps):
        x_data, y_data = mnist.test.next_batch(batch_size)
        correct = sess.run(tf.count_nonzero(tf.equal(tf.argmax(net, 1), tf.argmax(y, 1))),
                            feed_dict = {x: x_data, y: y_data})
        correct_c += correct
    accuracy = correct_c / mnist.test.num_examples
    print("Test Accuracy: ", accuracy)

    tf.saved_model.simple_save(sess, DIRECTORY, inputs = {"x": x}, outputs = {"net": net})
