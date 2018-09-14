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

model = "temp";
let interval_id;

async function setup(){
    try {
        model = await tf.loadModel("https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/mnist/saved_model/js_saved_model.json",
                               strict = false);
    } catch (err) {
        console.log("Error: ", err);
    }
}

function calc_brightness(){
    console.log("Model: ", model);
      // Reads the image as a Tensor from the webcam <video> element.
      preprocessor = tf.fromPixels(video);
      preprocessor = tf.mean(preprocessor, 2);
      preprocessor = tf.expandDims(preprocessor, 0);
      preprocessor = tf.expandDims(preprocessor, 3);
      output = model.predict(preprocessor);
      output = tf.argMax(output);
      document.getElementById("digit").innerHTML = output.get(0);
}

setup();

// calc_brightness();
interval_id = setInterval(calc_brightness, 1000);
