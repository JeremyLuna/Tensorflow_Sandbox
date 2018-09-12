// learning from https://www.tensorflow.org/guide/saved_model#models
// https://github.com/tensorflow/tfjs-converter
// https://js.tensorflow.org/api/0.11.2/#loadModel


// stream from camera
const video = document.getElementById("webcam");
navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {video.srcObject = stream});
video.width = 28;
video.height = 28;

function calc_brightness(){
      // Reads the image as a Tensor from the webcam <video> element.
      net = tf.fromPixels(video);
      net = tf.mean(net);
      document.getElementById("digit").innerHTML = net;
}

setInterval(calc_brightness, 1000);
