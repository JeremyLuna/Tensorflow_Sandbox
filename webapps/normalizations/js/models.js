conv_demo_model = "\n\
input = tf.input({shape: [2]});\n\
\n\
net = input;\n\
net = tf.layers.dense({units: 6, activation: 'relu', useBias: true}).apply(net);\n\
net = tf.layers.dense({units: 6, activation: 'relu', useBias: true}).apply(net);\n\
net = tf.layers.dense({units: 2, activation: 'linear', useBias: true}).apply(net);\n\
\n\
output = net;\n\
model = tf.model({inputs: input, outputs: output});\n\
\n\
model.compile({loss: tf.losses.softmaxCrossEntropy, optimizer: 'sgd'});\n\
";

euclidian_model = "\n\
input = tf.input({shape: [2]});\n\
\n\
net = input;\n\
feature = tf.variable(tf.randomNormal(shape=[4, 2], mean=0.5, stdDev=0.2));\n\
net = tf.sub().apply(feature, input);\n\
net = tf.mul().apply(net, net);\n\
net = tf.sum().apply(net, 1);\n\
net = tf.sqrt().apply(net);\n\
net = tf.layers.dense({units: 2, activation: 'linear', useBias: true}).apply(net);\n\
\n\
output = net;\n\
model = tf.model({inputs: input, outputs: output});\n\
\n\
model.compile({loss: tf.losses.softmaxCrossEntropy, optimizer: 'sgd'});\n\
";
