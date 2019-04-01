// learning from
// user upload:
//      https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file
//      tf.io.browserFiles()
//      https://js.tensorflow.org/api/0.13.3/#loadModel
//      https://simpl.info/getusermedia/sources/
//      https://js.tensorflow.org/tutorials/model-save-load.html

// stream from camera
const video = document.getElementById("webcam");

function loadFile(filePath) {
  var result = null;
  var xmlhttp = new XMLHttpRequest();
  xmlhttp.open("GET", filePath, false);
  xmlhttp.send();
  if (xmlhttp.status==200) {
    result = xmlhttp.responseText;
  }
  return result;
}

let model;
let interval_id;
var classes = loadFile("https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/flower_classification/trained_model/output_labels");
classes = classes.split("\n");

async function setup(){
    try {
        document.getElementById("disease").innerHTML = "Loading Model ...";
        storage_dir = "https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/flower_classification/mobilenet/web_model/";
        model_dir = "tensorflowjs_model.pb";
        weights_dir = "weights_manifest.json";
        model = await tf.loadFrozenModel(storage_dir+model_dir, storage_dir+weights_dir);
    } catch (err) {
        console.log("Error: ", err);
    }
}

document.getElementById("disease").innerHTML = "Model Loaded";

function calc_disease(){
    // Reads the image as a Tensor from the webcam <video> element.
    preprocessor = tf.fromPixels(video).asType('float32');
    preprocessor = tf.div(preprocessor, 255);
    preprocessor = tf.expandDims(preprocessor, 0);
    output = model.predict(preprocessor);
    output = tf.argMax(output, 1);
    output = classes[output.get([0])];
    document.getElementById("disease").innerHTML = "Disease: " + output;
}

setup();

interval_id = setInterval(calc_disease, 1000);
