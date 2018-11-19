// learning from
// user upload:
//      https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file
//      tf.io.browserFiles()
//      https://js.tensorflow.org/api/0.13.3/#loadModel
//      https://simpl.info/getusermedia/sources/
//      https://js.tensorflow.org/tutorials/model-save-load.html
// saving tf python models:
//      https://www.tensorflow.org/guide/saved_model#models
// converting to tf js model:
//      https://github.com/tensorflow/tfjs-converter
// loading tf js model
//      https://js.tensorflow.org/api/0.11.2/#loadModel actually loadFrozenModel
// TODO: video in canvas
//      https://stackoverflow.com/questions/4429440/html5-display-video-inside-canvas
//      http://html5doctor.com/video-canvas-magic/
// TODO: drawing on canvas
//      https://www.w3schools.com/Html/html5_canvas.asp
// Other dataset "plant village":
//      https://github.com/spMohanty/PlantVillage-Dataset

// stream from camera
const video = document.getElementById("webcam");

model = "temp";
let interval_id;
var classes = ['anthracnose', 'aphids', 'black _spot', 'botrytis_blight', 'cercospora', 'crown_gall', 'downy_mildew', 'healthy', 'leaf_cutting_bee', 'mosaic', 'powdery_mildew', 'rose_chafer', 'rose_rosette', 'rust', 'stem_cankers', 'thrips', 'two-spotted_mite'];

async function setup(){
    try {
        storage_dir = "https://jeremyluna.github.io/Tensorflow_Sandbox/webapps/flower_classification/js_model_saves/";
        model_dir = "tensorflowjs_model.pb";
        weights_dir = "weights_manifest.json";
        model = await tf.loadFrozenModel(storage_dir+model_dir, storage_dir+weights_dir);
    } catch (err) {
        console.log("Error: ", err);
    }
}

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