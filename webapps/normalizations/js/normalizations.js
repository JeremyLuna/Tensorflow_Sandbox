// this is the convjs visualization tool, with a tensorflowjs backend
// learned from https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

// TODO: put dataset back in buttons, hardcode optimizer
// TODO: refactor
// TODO: change layout
// TODO: fix mouse position
//   https://www.html5canvastutorials.com/advanced/html5-canvas-mouse-coordinates/

var dataset, N;
var density= 5.0;
var ss = 50.0; // canvas is like a 4x4 or something
var netx;
pixel_count = 0;
row_pixel_count = 0;
column_pixel_count = 0;
plot_value = 'class'; // 'class'|'softmax'|'confidence'

// create neural net

function loss(prediction, target){
  return tf.tidy(function(){
    return tf.losses.softmaxCrossEntropy(prediction, target);
  });
}

function train (inputs, labels){
  optimizer.minimize(function() {
    predictions = predict(inputs);
    stepLoss = loss(predictions, labels);
    document.getElementById("loss_display").innerHTML = "Loss: " + stepLoss;
    return stepLoss;
  });
};

async function reload() {
  eval(document.getElementById('layerdef').value);
}

function myinit() {
    var im = [];
    pixel_count = 0;

    for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
      if (cx > row_pixel_count) row_pixel_count = cx;
      for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
        if (cy > column_pixel_count) column_pixel_count = cy;
        pixel_count++;
        im.push([]);
        im[pixel_count-1].push((x-WIDTH/2)/ss);
        im[pixel_count-1].push((y-HEIGHT/2)/ss);
      }
    }
    netx = tf.tensor(im);
}

async function update(){// TODO: minibatches
  while (true) { // epochs
    await draw();
    tf.tidy(() => {
        x = tf.tensor(dataset['data']);
        y = tf.oneHot(tf.tensor(dataset['labels']).asType('int32'), 2).asType('float32');
        // y = dataset['labels']
        train(x, y);
    });
  }
}

function mouseClick(x, y, shiftPressed, ctrlPressed){

  // x and y transformed to data space coordinates
  var xt = (x-WIDTH/2)/ss;
  var yt = (y-HEIGHT/2)/ss;

  if(ctrlPressed) {
    // remove closest data point
    var mink = -1;
    var mind = 99999;
    for(var k=0, n=dataset['data'].length;k<n;k++) {
      var dx = dataset['data'][k][0] - xt;
      var dy = dataset['data'][k][1] - yt;
      var d = dx*dx+dy*dy;
      if(d < mind || k==0) {
        mind = d;
        mink = k;
      }
    }
    if(mink>=0) {
      dataset['data'].splice(mink, 1);
      dataset['labels'].splice(mink, 1);
      N -= 1;
    }

  } else {
    // add datapoint at location of click
    dataset['data'].push([xt, yt]);
    dataset['labels'].push(shiftPressed ? 1 : 0);
    N += 1;
  }

}

function keyDown(key){
}

function keyUp(key) {
}

function load_dataset(new_dataset){
    dataset = new_dataset;
    N = dataset['labels'].length;
}

function load_model(model){
    document.getElementById('layerdef').value = model;
    reload();
}

load_model(dense_model);
load_dataset(simple_dataset());
reload();
NPGinit();
