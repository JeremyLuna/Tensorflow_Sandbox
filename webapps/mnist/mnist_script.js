// learning from
// saving tf python models:
//      https://www.tensorflow.org/guide/saved_model#models
// converting to tf js model:
//      https://github.com/tensorflow/tfjs-converter
// loading tf js model
//      https://js.tensorflow.org/api/0.11.2/#loadModel


// stream from camera
const video = document.getElementById("webcam");
navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {video.srcObject = stream});
video.width = 28;
video.height = 28;

function calc_brightness(){
      // Reads the image as a Tensor from the webcam <video> element.
      net = tf.fromPixels(video);
      net = tf.mean(net, [2])
      document.getElementById("digit").innerHTML = net;
}

setInterval(calc_brightness, 1000);
