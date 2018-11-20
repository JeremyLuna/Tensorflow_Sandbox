// learning from
// user upload:
//      https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input/file
//      tf.io.browserFiles()
//      https://js.tensorflow.org/api/0.13.3/#loadModel
//      https://simpl.info/getusermedia/sources/
//      https://js.tensorflow.org/tutorials/model-save-load.html

let model;
let source_image;
let interval_id;
var classes = ['anthracnose', 'aphids', 'black _spot', 'botrytis_blight', 'cercospora', 'crown_gall', 'downy_mildew', 'healthy', 'leaf_cutting_bee', 'mosaic', 'powdery_mildew', 'rose_chafer', 'rose_rosette', 'rust', 'stem_cankers', 'thrips', 'two-spotted_mite'];

// load model
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
    // Reads the image as a Tensor from the file
    // preprocessor = tf.fromPixels(video).asType('float32');
    // preprocessor = tf.div(preprocessor, 255);
    // preprocessor = tf.expandDims(preprocessor, 0);
    // output = model.predict(preprocessor);
    // output = tf.argMax(output, 1);
    // output = classes[output.get([0])];
    // document.getElementById("disease").innerHTML = "Disease: " + output;
}

function handleFileSelect(evt) {
  var files = evt.target.files;
  var f = files[0];
  var url_reader = new FileReader();
  url_reader.onload = (function(theFile) {
    return function(e) {
      document.getElementById('image_display').innerHTML = ['<img src="', e.target.result,'" title="', theFile.name, '" width="50" />'].join('');
    };
  })(f);
  reader.readAsDataURL(f);

  var array_reader = new FileReader();
  array_reader.readAsArrayBuffer(f);
  source_image = array_reader.result;
}

function update(evt){
  handleFileSelect(evt);
  calc_disease();
  console.log(source_image);
}

setup();
document.getElementById('input_image').addEventListener('change', update, false);

// interval_id = setInterval(calc_disease, 1000);
