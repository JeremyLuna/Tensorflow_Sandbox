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
classes.pop();
logit_datapoints = classes.map(function(classe) {return {y: 0, label: classe}});

var chart = new CanvasJS.Chart("chartContainer", {
	animationEnabled: true,

	title:{
		text:"Plant Disease"
	},
	axisX:{
		interval: 1
	},
	axisY2:{
		interlacedColor: "rgba(1,77,101,.2)",
		gridColor: "rgba(1,77,101,.1)",
		title: "Confidence",
    suffix: "%"
	},
	data: [{
		type: "bar",
		name: "Plant/Disease",
		axisYType: "secondary",
		color: "#014D65",
		dataPoints: logit_datapoints
	}]
});


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
    // Reads the image as a Tensor from the webcam <video> element.
    preprocessor = tf.fromPixels(video).asType('float32');
    preprocessor = tf.div(preprocessor, 255);
    preprocessor = tf.expandDims(preprocessor, 0);
    logits = model.predict(preprocessor);
    output = tf.argMax(logits, 1);
    output = classes[output.get([0])];
    document.getElementById("disease").innerHTML = "Disease: " + output;
    for (let i=0; i < chart.options.data[0].dataPoints.length; i++){
      chart.options.data[0].dataPoints[i].y = logits.dataSync()[i];
    }
    chart.render();
}

setup();

interval_id = setInterval(calc_disease, 1000);
