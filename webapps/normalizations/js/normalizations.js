// this is the convjs visualization tool, with a tensorflowjs backend
// learned from https://cs.stanford.edu/people/karpathy/convnetjs/demo/classify2d.html

var data, labels, N;
var ss = 50.0; // scale for drawing
var netx;
batch_size_res = 0;
batch_size_x_res = 0;
batch_size_y_res = 0;

viscanvas = document.getElementById('viscanvas');
visctx = viscanvas.getContext('2d');
visWIDTH = viscanvas.width;
visHEIGHT = viscanvas.height;

// create neural net
var net;
var t = "\n\
input = tf.input({shape: [2]});\n\
\n\
dense1 = tf.layers.dense({units: 6, activation: 'relu', useBias: true});\n\
dense2 = tf.layers.dense({units: 6, activation: 'relu', useBias: true});\n\
dense3 = tf.layers.dense({units: 2, activation: 'linear', useBias: true});\n\
\n\
output = dense3.apply(dense2.apply(dense1.apply(input)));\n\
net = tf.model({inputs: input, outputs: output});\n\
\n\
net.compile({loss: tf.losses.softmaxCrossEntropy, optimizer: 'sgd'});\n\
";

async function reload() {
  eval(document.getElementById('layerdef').value);
  // enter buttons for layers
  // var t = '';
  // for(var i=1;i<net.layers.length-1;i++) { // ignore input and regression layers (first and last)
  //   var butid = "button" + i;
  //   t += "<input id=\""+butid+"\" value=\"" + net.layers[i].layer_type + "(" + net.layers[i].out_depth + ")" +"\" type=\"submit\" onclick=\"updateLix("+i+")\" style=\"width:80px; height: 30px; margin:5px;\";>";
  // }
  // document.getElementById('layer_ixes').innerHTML = t;
  // document.getElementById('cyclestatus').value = 'drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')';
  NPGinit(1000);
}
function updateLix(newlix) {
  lix = newlix;
  d0 = 0;
  d1 = 1; // reset these
  document.getElementById('cyclestatus').value = 'drawing neurons ' + d0 + ' and ' + d1 + ' of layer with index ' + lix + ' (' + net.layers[lix].layer_type + ')';

}

function myinit() {
    var im = [];
    var density= 5.0;
    for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
      if (cx > batch_size_x_res) batch_size_x_res = cx;
      for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
        if (cy > batch_size_y_res) batch_size_y_res = cy;
        batch_size_res++;
        im.push([]);
        im[batch_size_res-1].push((x-WIDTH/2)/ss);
        im[batch_size_res-1].push((y-HEIGHT/2)/ss);
      }
    }
    netx = tf.tensor(im);
}

function random_data(){
  data = [];
  labels = [];
  for(var k=0;k<40;k++) {
    data.push([convnetjs.randf(-3,3), convnetjs.randf(-3,3)]); labels.push(convnetjs.randf(0,1) > 0.5 ? 1 : 0);
  }
  N = labels.length;
}

function original_data(){
  data = [];
  labels = [];
  data.push([-0.4326  ,  1.1909 ]); labels.push(1);
  data.push([3.0, 4.0]); labels.push(1);
  data.push([0.1253 , -0.0376   ]); labels.push(1);
  data.push([0.2877 ,   0.3273  ]); labels.push(1);
  data.push([-1.1465 ,   0.1746 ]); labels.push(1);
  data.push([1.8133 ,   1.0139  ]); labels.push(0);
  data.push([2.7258 ,   1.0668  ]); labels.push(0);
  data.push([1.4117 ,   0.5593  ]); labels.push(0);
  data.push([4.1832 ,   0.3044  ]); labels.push(0);
  data.push([1.8636 ,   0.1677  ]); labels.push(0);
  data.push([0.5 ,   3.2  ]); labels.push(1);
  data.push([0.8 ,   3.2  ]); labels.push(1);
  data.push([1.0 ,   -2.2  ]); labels.push(1);
  N = labels.length;
}

// function circle_data() {
//   data = [];
//   labels = [];
//   for(var i=0;i<50;i++) {
//     var r = convnetjs.randf(0.0, 2.0);
//     var t = convnetjs.randf(0.0, 2*Math.PI);
//     data.push([r*Math.sin(t), r*Math.cos(t)]);
//     labels.push(1);
//   }
//   for(var i=0;i<50;i++) {
//     var r = convnetjs.randf(3.0, 5.0);
//     //var t = convnetjs.randf(0.0, 2*Math.PI);
//     var t = 2*Math.PI*i/50.0
//     data.push([r*Math.sin(t), r*Math.cos(t)]);
//     labels.push(0);
//   }
//   N = data.length;
// }
//
// function spiral_data() {
//   data = [];
//   labels = [];
//   var n = 100;
//   for(var i=0;i<n;i++) {
//     var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
//     var t = 1.25*i/n*2*Math.PI + convnetjs.randf(-0.1, 0.1);
//     data.push([r*Math.sin(t), r*Math.cos(t)]);
//     labels.push(1);
//   }
//   for(var i=0;i<n;i++) {
//     var r = i/n*5 + convnetjs.randf(-0.1, 0.1);
//     var t = 1.25*i/n*2*Math.PI + Math.PI + convnetjs.randf(-0.1, 0.1);
//     data.push([r*Math.sin(t), r*Math.cos(t)]);
//     labels.push(0);
//   }
//   N = data.length;
// }

async function update(){
  for(var iters=0;iters<1000;iters++) { // epochs
    draw();
    x = tf.tidy(() => {
      return tf.tensor(data);
    });
    y = tf.tidy(() => {
      return tf.oneHot(tf.tensor(labels).asType('int32'), 2).asType('float32');
    });
    await net.fit(x, y, {batchSize: N});
  }
}

// function cycle() {
//   var selected_layer = net.layers[lix];
//   d0 += 1;
//   d1 += 1;
//   if(d1 >= selected_layer.out_depth) d1 = 0; // and wrap
//   if(d0 >= selected_layer.out_depth) d0 = 0; // and wrap
//   document.getElementById('cyclestatus').value = 'drawing neurons ' + d0 + ' and ' + d1 + ' of layer #' + lix + ' (' + net.layers[lix].layer_type + ')';
// }

var lix = 4; // layer id to track first 2 neurons of
var d0 = 0; // first dimension to show visualized
var d1 = 1; // second dimension to show visualized
async function draw(){
    ctx.clearRect(0,0,WIDTH,HEIGHT);
    // draw decisions in the grid
    var density= 5.0;
    var gridstep = 2;
    var gridx = [];
    var gridy = [];
    var gridl = [];

    var a = await net.predict(netx, {batchSize: batch_size_res}).data();
    for(var x=0.0, cx=0; x<=WIDTH; x+= density, cx++) {
      for(var y=0.0, cy=0; y<=HEIGHT; y+= density, cy++) {
        if(a[(cx*batch_size_y_res + cy)*2] > a[(cx*batch_size_y_res + cy)*2 + 1]) ctx.fillStyle = 'rgb(250, 150, 150)';
        else ctx.fillStyle = 'rgb(150, 250, 150)';
        ctx.fillRect(x-density/2-1, y-density/2-1, density+2, density+2);
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

    // draw representation transformation axes for two neurons at some layer
    var mmx = cnnutil.maxmin(gridx);
    var mmy = cnnutil.maxmin(gridy);
    visctx.clearRect(0,0,visWIDTH,visHEIGHT);
    visctx.strokeStyle = 'rgb(0, 0, 0)';
    var n = Math.floor(Math.sqrt(gridx.length)); // size of grid. Should be fine?
    var ng = gridx.length;
    var c = 0; // counter
    visctx.beginPath()
    for(var x=0;x<n;x++) {
      for(var y=0;y<n;y++) {

        // down
        var ix1 = x*n+y;
        var ix2 = x*n+y+1;
        if(ix1 >= 0 && ix2 >= 0 && ix1 < ng && ix2 < ng && y<n-1) { // check oob
          var xraw = gridx[ix1];
          xraw1 = visWIDTH*(gridx[ix1] - mmx.minv)/mmx.dv;
          yraw1 = visHEIGHT*(gridy[ix1] - mmy.minv)/mmy.dv;
          xraw2 = visWIDTH*(gridx[ix2] - mmx.minv)/mmx.dv;
          yraw2 = visHEIGHT*(gridy[ix2] - mmy.minv)/mmy.dv;
          visctx.moveTo(xraw1, yraw1);
          visctx.lineTo(xraw2, yraw2);
        }

        // and draw its color
        if(gridl[ix1]) visctx.fillStyle = 'rgb(250, 150, 150)';
        else visctx.fillStyle = 'rgb(150, 250, 150)';
        var sz = density * gridstep;
        visctx.fillRect(xraw1-sz/2-1, yraw1-sz/2-1, sz+2, sz+2);

        // right
        var ix1 = (x+1)*n+y;
        var ix2 = x*n+y;
        if(ix1 >= 0 && ix2 >= 0 && ix1 < ng && ix2 < ng && x <n-1) { // check oob
          var xraw = gridx[ix1];
          xraw1 = visWIDTH*(gridx[ix1] - mmx.minv)/mmx.dv;
          yraw1 = visHEIGHT*(gridy[ix1] - mmy.minv)/mmy.dv;
          xraw2 = visWIDTH*(gridx[ix2] - mmx.minv)/mmx.dv;
          yraw2 = visHEIGHT*(gridy[ix2] - mmy.minv)/mmy.dv;
          visctx.moveTo(xraw1, yraw1);
          visctx.lineTo(xraw2, yraw2);
        }

      }
    }
    visctx.stroke();

    // draw datapoints.
    ctx.strokeStyle = 'rgb(0,0,0)';
    ctx.lineWidth = 1;
    for(var i=0;i<N;i++) {
      if(labels[i]==1) ctx.fillStyle = 'rgb(100,200,100)';
      else ctx.fillStyle = 'rgb(200,100,100)';

      drawCircle(data[i][0]*ss+WIDTH/2, data[i][1]*ss+HEIGHT/2, 5.0);

      // also draw transformed data points while we're at it
      // netx[0][0] = data[i][0];
      // netx[0][1] = data[i][1];
      // console.log("predicting2");
      // var a = net.predict(tf.tensor(netx), {batchSize: 1});
      // console.log("predicting2 done");
      // var xt = visWIDTH * (net.layers[lix].out_act.w[d0] - mmx.minv) / mmx.dv; // in screen coords
      // var yt = visHEIGHT * (net.layers[lix].out_act.w[d1] - mmy.minv) / mmy.dv; // in screen coords
      if(labels[i]==1) visctx.fillStyle = 'rgb(100,200,100)';
      else visctx.fillStyle = 'rgb(200,100,100)';
      visctx.beginPath();
      // visctx.arc(xt, yt, 5.0, 0, Math.PI*2, true);
      visctx.closePath();
      visctx.stroke();
      visctx.fill();
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
    for(var k=0, n=data.length;k<n;k++) {
      var dx = data[k][0] - xt;
      var dy = data[k][1] - yt;
      var d = dx*dx+dy*dy;
      if(d < mind || k==0) {
        mind = d;
        mink = k;
      }
    }
    if(mink>=0) {
      console.log('splicing ' + mink);
      data.splice(mink, 1);
      labels.splice(mink, 1);
      N -= 1;
    }

  } else {
    // add datapoint at location of click
    data.push([xt, yt]);
    labels.push(shiftPressed ? 1 : 0);
    N += 1;
  }

}

function keyDown(key){
}

function keyUp(key) {
}

document.getElementById('layerdef').value = t;
// circle_data();
original_data();
reload();
