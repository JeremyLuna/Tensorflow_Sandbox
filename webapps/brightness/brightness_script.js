// learning from:
// https://www.html5rocks.com/en/tutorials/getusermedia/intro/
// https://js.tensorflow.org/tutorials/webcam-transfer-learning.html :
//      https://github.com/tensorflow/tfjs-examples/tree/master/webcam-transfer-learning
// https://medium.com/tensorflow/a-gentle-introduction-to-tensorflow-js-dba2e5257702

// TODO: use
//      window.screen.availHeight
//      window.screen.availWidth

// stream from camera
const video = document.getElementById("webcam");
navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {video.srcObject = stream});
video.width = 320;
video.height = 240;

function calc_brightness(){
      // Reads the image as a Tensor from the webcam <video> element.
      net = tf.fromPixels(video);
      net = tf.mean(net);
      document.getElementById("brightness").innerHTML = net;
}

setInterval(calc_brightness, 1000);
