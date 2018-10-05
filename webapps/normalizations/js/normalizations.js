// this is the convjs visualization tool, with a tensorflowjs backend
// learned from https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

// TODO: put each thing in text boxes and buttons, and automate their creation
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

async function draw(){
    // clear canvas
    // do I need this? makes it flicker but might clear memory
    // ctx.clearRect(0,0,WIDTH,HEIGHT);

    // draw decisions in the grid
    var gridstep = 2;
    var gridx = [];
    var gridy = [];
    var gridl = [];

    var a = await tf.softmax(predict(netx)).data();
    if (plot_value == 'class'){
      for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
        for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
          blue = a[(cx*(column_pixel_count + 1)+ cy)*2];
          red = a[(cx*(column_pixel_count + 1) + cy)*2 + 1];
          if(blue > red){
            ctx.fillStyle = 'rgb(200, 100, 100)';
          }else{
            ctx.fillStyle = 'rgb(100, 100, 200)';
          }
          ctx.fillRect(x-density/2-1, y-density/2-1, density+2, density+2);
        }
      }
    }else if(plot_value == 'softmax'){
      for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
        for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
          blue = a[(cx*(column_pixel_count + 1)+ cy)*2];
          red = a[(cx*(column_pixel_count + 1) + cy)*2 + 1];
          ctx.fillStyle = 'rgb(' + blue*255 +',0,'+ red*255 +')';
          ctx.fillRect(x-density/2-1, y-density/2-1, density+2, density+2);
        }
      }
    }else if(plot_value == 'confidence'){
      for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
        for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
          blue = a[(cx*(column_pixel_count + 1)+ cy)*2];
          red = a[(cx*(column_pixel_count + 1) + cy)*2 + 1];
          confidence = Math.abs(blue-red);
          if(blue > red){
            ctx.fillStyle = 'rgb(' + confidence*255 +',0,'+ 0 +')';
          }else{
            ctx.fillStyle = 'rgb(' + 0 +',0,'+ confidence*255 +')';
          }
          ctx.fillRect(x-density/2-1, y-density/2-1, density+2, density+2);
        }
      }
    }


    // draw axes
    ctx.beginPath();
    ctx.strokeStyle = 'rgb(50,50,50)';
    ctx.lineWidth = 1;
    ctx.moveTo(0, HEIGHT/2);
    ctx.lineTo(WIDTH, HEIGHT/2);
    ctx.moveTo(WIDTH/2, 0);
    ctx.lineTo(WIDTH/2, HEIGHT);
    ctx.stroke();

    // draw datapoints.
    ctx.strokeStyle = 'rgb(0,0,0)';
    ctx.lineWidth = 1;
    for(var i=0;i<N;i++) {
      if(labels[i]==1) ctx.fillStyle = 'rgb(0,0,255)';
      else ctx.fillStyle = 'rgb(255,0,0)';
      drawCircle(dataset['data'][i][0]*ss+WIDTH/2, dataset['data'][i][1]*ss+HEIGHT/2, 5.0);
    }
    return true;
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

function load_dataset(){
    switch (document.getElementById("dataset_selector").value) {
        case "Simple":
            dataset = simple_dataset();
            break;
        case "Cluster":
            dataset = cluster_dataset();
            break;
        case "Circle":
            dataset = circle_dataset();
            break;
        case "Telephone Pole":
            dataset = telephone_pole_dataset();
            break;
        case "Spiral":
            dataset = spiral_dataset();
            break;
        case "Angular":
            dataset = angular_dataset();
            break;
        case "Random":
            dataset = random_dataset();
            break;
    }
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
