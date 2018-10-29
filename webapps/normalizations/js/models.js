// TODO: make angular and weighted models

dense_model = "\
weights_1 = tf.variable(tf.randomNormal(shape=[2, 8], mean=0.0, stdDev=0.2));\n\
biases_1 = tf.variable(tf.randomNormal(shape=[1, 8], mean=0.5, stdDev=0.2));\n\
weights_2 = tf.variable(tf.randomNormal(shape=[8, 2], mean=0.0, stdDev=0.2));\n\
biases_2 = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = input;\n\
    net = tf.matMul(net, weights_1).add(biases_1);\n\
    net = tf.relu(net);\n\
    net = tf.matMul(net, weights_2).add(biases_2);\n\
    return net;\n\
  });\n\
};\n\
\n\
optimizer = tf.train.adam(.01);\n\
";

// these two seem to discriminate by angle
manhattan_model = "\
feature = tf.variable(tf.randomUniform(shape=[1, 2, 8], minval=-2, maxval=2));\n\
weights = tf.variable(tf.randomUniform(shape=[8, 2], minval=-2, maxval=2));\n\
biases = tf.variable(tf.randomUniform(shape=[1, 2], minval=-2, maxval=2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = tf.expandDims(input, 2);\n\
    net = tf.sub(feature, net);\n\
    net = tf.tanh(tf.norm(net, ord=1, axis=1, keepDims=false));\n\
    net = tf.matMul(net, weights).add(biases);\n\
    return net;\n\
  });\n\
};\n\
\n\
optimizer = tf.train.adam(.01);\n\
";

euclidian_model = "\
feature = tf.variable(tf.randomUniform(shape=[1, 2, 8], minval=-2, maxval=2));\n\
weights = tf.variable(tf.randomUniform(shape=[8, 2], minval=0, maxval=1));\n\
biases = tf.variable(tf.randomUniform(shape=[1, 2], minval=-2, maxval=2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    net = tf.expandDims(input, 2);\n\
    net = tf.sub(feature, net);\n\
    net = tf.tanh(tf.norm(net, ord=2, axis=1, keepDims=false));\n\
    net = tf.matMul(net, weights).add(biases);\n\
    return net;\n\
  });\n\
};\n\
\n\
optimizer = tf.train.adam(.01);\n\
";

angular_model = "\
feature = tf.variable(tf.randomNormal(shape=[2, 8], mean=0.0, stdDev=0.2));\n\
weights = tf.variable(tf.randomNormal(shape=[8, 2], mean=0.0, stdDev=0.2));\n\
biases = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    // angle between input vector and 8 other vectors\n\
    net = tf.matMul(input, feature);\n\
    net = tf.div(net, tf.norm(feature, ord=2, axis=0, keepDims=false));\n\
    net = tf.div(net, tf.norm(input, ord=2, axis=1, keepDims=false).expandDims(1));\n\
    net = tf.tanh(net);\n\
    // weighted sum of angles\n\
    net = tf.matMul(net, weights).add(biases);\n\
    return net;\n\
  });\n\
};\n\
\n\
optimizer = tf.train.adam(.01);\n\
";

merged_model = "\
dense_features = tf.variable(tf.randomNormal(shape=[2, 2], mean=0.0, stdDev=0.2));\n\
dense_bias = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
manhattan_features = tf.variable(tf.randomUniform(shape=[1, 2, 2], minval=-2, maxval=2));\n\
euclidian_features = tf.variable(tf.randomUniform(shape=[1, 2, 2], minval=-2, maxval=2));\n\
angle_features = tf.variable(tf.randomNormal(shape=[2, 2], mean=0.0, stdDev=0.2));\n\
\n\
weights = tf.variable(tf.randomNormal(shape=[8, 2], mean=0.0, stdDev=0.2));\n\
biases = tf.variable(tf.randomNormal(shape=[1, 2], mean=0.5, stdDev=0.2));\n\
\n\
predict = function(input){\n\
  return tf.tidy(function(){\n\
    // dense layer\n\
    dense_net = tf.matMul(input, dense_features).add(dense_bias);\n\
    dense_net = tf.relu(dense_net);\n\
    \n\
    // manhattan layer\n\
    manhattan_net = tf.expandDims(input, 2);\n\
    manhattan_net = tf.sub(manhattan_features, manhattan_net);\n\
    manhattan_net = tf.tanh(tf.norm(manhattan_net, ord=1, axis=1, keepDims=false));\n\
    \n\
    // euclidian layer\n\
    euclidian_net = tf.expandDims(input, 2);\n\
    euclidian_net = tf.sub(euclidian_features, euclidian_net);\n\
    euclidian_net = tf.tanh(tf.norm(euclidian_net, ord=2, axis=1, keepDims=false));\n\
    \n\
    // angular layer\n\
    angle_net = tf.matMul(input, angle_features);\n\
    angle_net = tf.div(angle_net, tf.norm(angle_features, ord=2, axis=0, keepDims=false));\n\
    angle_net = tf.div(angle_net, tf.norm(input, ord=2, axis=1, keepDims=false).expandDims(1));\n\
    angle_net = tf.tanh(angle_net);\n\
    \n\
    \n\
    // weighted sum + bias of all features\n\
    net = tf.concat([dense_net, manhattan_net, euclidian_net, angle_net], 1)\n\
    net = tf.matMul(net, weights).add(biases);\n\
    \n\
    return net;\n\
  });\n\
};\n\
\n\
optimizer = tf.train.adam(.01);\n\
";
