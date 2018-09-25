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
feature = tf.variable(tf.randomNormal(shape=[2, 8], mean=0, stdDev=0.5));\n\
weights = tf.variable(tf.randomNormal(shape=[8, 2], mean=1.0, stdDev=0.2));\n\
biases = tf.variable(tf.randomNormal(shape=[1, 2], mean=1.0, stdDev=0.2));\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = tf.expandDims(input, 2);\n\
    net = tf.sub(tf.expandDims(feature, 0), net);\n\
    net = tf.mul(net, net);\n\
    net = tf.sqrt(tf.sum(net, 1));\n\
    net = tf.relu(net);\n\
    net = tf.matMul(net, weights).add(biases);\n\
    net = tf.relu(net);\n\
    return net;\n\
  });\n\
};\n\
\n\
function loss(prediction, target){\n\
return tf.tidy(function(){\n\
  return tf.losses.softmaxCrossEntropy(prediction, target);\n\
});\n\
}\n\
optimizer = tf.train.sgd(.04);\n\
";
