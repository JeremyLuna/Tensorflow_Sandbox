//Simple game engine
//Author: Andrej Karpathy
//License: BSD
//This function does all the boring canvas stuff. To use it, just create functions:
//myinit()          gets called once in beginning
//update()          gets called every frame
//draw()            gets called every frame
//mouseClick(x, y)  gets called on mouse click
//keyUp(keycode)    gets called when key is released
//keyDown(keycode)  gets called when key is pushed

var canvas;
var ctx;
var WIDTH;
var HEIGHT;

function drawBubble(x, y, w, h, radius)
{
  var r = x + w;
  var b = y + h;
  ctx.beginPath();
  ctx.strokeStyle="black";
  ctx.lineWidth="2";
  ctx.moveTo(x+radius, y);
  ctx.lineTo(x+radius/2, y-10);
  ctx.lineTo(x+radius * 2, y);
  ctx.lineTo(r-radius, y);
  ctx.quadraticCurveTo(r, y, r, y+radius);
  ctx.lineTo(r, y+h-radius);
  ctx.quadraticCurveTo(r, b, r-radius, b);
  ctx.lineTo(x+radius, b);
  ctx.quadraticCurveTo(x, b, x, b-radius);
  ctx.lineTo(x, y+radius);
  ctx.quadraticCurveTo(x, y, x+radius, y);
  ctx.stroke();
}

function drawRect(x, y, w, h){
  ctx.beginPath();
  ctx.rect(x,y,w,h);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

function drawCircle(x, y, r){
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI*2, true);
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
}

//uniform distribution integer
function randi(s, e) {
  return Math.floor(Math.random()*(e-s) + s);
}

//uniform distribution
function randf(s, e) {
  return Math.random()*(e-s) + s;
}

//normal distribution random number
function randn(mean, variance) {
  var V1, V2, S;
  do {
    var U1 = Math.random();
    var U2 = Math.random();
    V1 = 2 * U1 - 1;
    V2 = 2 * U2 - 1;
    S = V1 * V1 + V2 * V2;
  } while (S > 1);
  X = Math.sqrt(-2 * Math.log(S) / S) * V1;
  X = mean + Math.sqrt(variance) * X;
  return X;
}

function eventClick(e) {

  //get position of cursor relative to top left of canvas
  var x;
  var y;
  if (e.pageX || e.pageY) {
    x = e.pageX;
    y = e.pageY;
  } else {
    x = e.clientX + document.body.scrollLeft + document.documentElement.scrollLeft;
    y = e.clientY + document.body.scrollTop + document.documentElement.scrollTop;
  }
  x -= canvas.offsetLeft;
  y -= canvas.offsetTop;

  //call user-defined callback
  mouseClick(x, y, e.shiftKey, e.ctrlKey);
}

//event codes can be found here:
//http://www.aspdotnetfaq.com/Faq/What-is-the-list-of-KeyCodes-for-JavaScript-KeyDown-KeyPress-and-KeyUp-events.aspx
function eventKeyUp(e) {
  var keycode = ('which' in e) ? e.which : e.keyCode;
  keyUp(keycode);
}

function eventKeyDown(e) {
  var keycode = ('which' in e) ? e.which : e.keyCode;
  keyDown(keycode);
}

function NPGinit(time_m){
  //takes miliseconds per training
  canvas = document.getElementById('NPGcanvas');
  ctx = canvas.getContext('2d');
  WIDTH = canvas.width;
  HEIGHT = canvas.height;
  canvas.addEventListener('click', eventClick, false);

  //canvas element cannot get focus by default. Requires to either set
  //tabindex to 1 so that it's focusable, or we need to attach listeners
  //to the document. Here we do the latter
  document.addEventListener('keyup', eventKeyUp, true);
  document.addEventListener('keydown', eventKeyDown, true);

  myinit();
  NPGtick();
}

function NPGtick() {
    update();
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
