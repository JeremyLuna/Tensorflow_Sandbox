// TODO: make angular and weighted models

dense_model = "\
weights_1 = tf.variable(tf.randomNormal(shape=[2, 16], mean=0.0, stdDev=0.2));\n\
biases_1 = tf.variable(tf.randomNormal(shape=[1, 16], mean=0.5, stdDev=0.2));\n\
weights_2 = tf.variable(tf.randomNormal(shape=[16, 16], mean=0.0, stdDev=0.2));\n\
biases_2 = tf.variable(tf.randomNormal(shape=[1, 16], mean=0.5, stdDev=0.2));\n\
weights_3 = tf.variable(tf.randomNormal(shape=[16, 2], mean=0.0, stdDev=0.2));\n\
biases_3 = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = input;\n\
    net = tf.matMul(net, weights_1).add(biases_1);\n\
    net = tf.relu(net);\n\
    net = tf.matMul(net, weights_2).add(biases_2);\n\
    net = tf.relu(net);\n\
    net = tf.matMul(net, weights_3).add(biases_3);\n\
    return net;\n\
  });\n\
};\n\
\n\
function loss(prediction, target){\n\
return tf.tidy(function(){\n\
  return tf.losses.softmaxCrossEntropy(prediction, target);\n\
});\n\
}\n\
\n\
optimizer = tf.train.sgd(.01);\n\
";

euclidian_model = "\
feature = tf.variable(tf.randomNormal(shape=[1, 2, 8], mean=0.0, stdDev=0.2));\n\
weights = tf.variable(tf.randomNormal(shape=[8, 2], mean=0.0, stdDev=0.2));\n\
biases = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = tf.expandDims(input, 2);\n\
    net = tf.sub(feature, net);\n\
    net = tf.norm(net, ord=2, axis=1, keepDims=false);\n\
    net = tf.matMul(net, weights).add(biases);\n\
    return net;\n\
  });\n\
};\n\
\n\
function loss(prediction, target){\n\
return tf.tidy(function(){\n\
  return tf.losses.softmaxCrossEntropy(prediction, target);\n\
});\n\
}\n\
\n\
optimizer = tf.train.sgd(.01);\n\
";

angle_model = "\
feature = tf.variable(tf.randomNormal(shape=[1, 2, 8], mean=0.0, stdDev=0.2));\n\
weights = tf.variable(tf.randomNormal(shape=[8, 2], mean=0.0, stdDev=0.2));\n\
biases = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = tf.expandDims(input, 2);\n\
    net = tf.norm(net, ord=2, axis=1, keepDims=false);\n\
    net = tf.sub(feature, net);\n\
    net = tf.norm(net, ord=2, axis=1, keepDims=false);\n\
    net = tf.matMul(net, weights).add(biases);\n\
    return net;\n\
  });\n\
};\n\
\n\
function loss(prediction, target){\n\
return tf.tidy(function(){\n\
  return tf.losses.softmaxCrossEntropy(prediction, target);\n\
});\n\
}\n\
\n\
optimizer = tf.train.sgd(.01);\n\
";
