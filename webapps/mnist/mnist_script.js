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

const loaded_model = await tf.loadFrozenModel(
    "https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/mnist/js_model_saves/tensorflowjs_model.pb",
    "https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/mnist/js_model_saves/weights_manifest.json");

function calc_brightness(){
      // Reads the image as a Tensor from the webcam <video> element.
      preprocessor = tf.fromPixels(video);
      preprocessor = tf.mean(preprocessor, 2);
      preprocessor = tf.expandDims(preprocessor, 0);
      output = loaded_model.predict(preprocessor);
      // TODO: take max, to just return the esimate
      // TODO: normalize in preprocessor?
      document.getElementById("digit").innerHTML = output;
}

setInterval(calc_brightness, 1000);
