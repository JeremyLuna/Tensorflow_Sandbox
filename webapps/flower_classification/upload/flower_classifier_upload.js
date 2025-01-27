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
let source_image;
let interval_id;
var classes = loadFile("https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/flower_classification/trained_model/output_labels");
classes = classes.split("\n");

// load model
async function setup(){
    try {
        document.getElementById("disease").innerHTML = "Loading Model ...";
        storage_dir = "https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/flower_classification/js_model/";
        model_dir = "tensorflowjs_model.pb";
        weights_dir = "weights_manifest.json";
        model = await tf.loadFrozenModel(storage_dir+model_dir, storage_dir+weights_dir);
        document.getElementById("disease").innerHTML = "Model Loaded";
    } catch (err) {
        console.log("Error: ", err);
    }
}

function calc_disease(){
    // Reads the image as a Tensor from the file
    preprocessor = tf.fromPixels(document.getElementById('input_image')).asType('float32');
    // crop middle square
    shape = preprocessor.shape
    diff = shape[0] - shape[1]
    start = Math.abs(Math.floor(diff/2))
    if (shape[0] > shape[1]){
      preprocessor = tf.slice(preprocessor, [start, 0, 0], [shape[1], shape[1], 3])
    } else {
      preprocessor = tf.slice(preprocessor, [0, start, 0], [shape[0], shape[0], 3])
    }
    // resize to whatever the network takes
    preprocessor = tf.image.resizeBilinear(preprocessor, [224, 224])
    preprocessor = tf.div(preprocessor, 255);
    preprocessor = tf.expandDims(preprocessor, 0);
    output = model.predict(preprocessor);
    output = tf.argMax(output, 1);
    output = classes[output.get([0])];
    document.getElementById("disease").innerHTML = "Disease: " + output;
}

function handleFileSelect(evt) {
  var files = evt.target.files;
  var f = files[0];

  var url_reader = new FileReader();
  url_reader.onload = (function(theFile) {
    return function(e) {
      document.getElementById('image_display').innerHTML = ['<img src="', e.target.result,
                                                            '" title="', theFile.name,
                                                            '"id=input_image />'].join('');
      setTimeout(calc_disease,100);
    };
  })(f);
  url_reader.readAsDataURL(f);
}

setup();
document.getElementById('input_image_file').addEventListener('change', handleFileSelect, false);
