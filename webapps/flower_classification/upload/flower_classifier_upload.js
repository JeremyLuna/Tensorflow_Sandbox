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
    preprocessor = tf.fromPixels(document.getElementById('input_image')).asType('float32');
    // TODO: crop middle square
    // TODO: resize to whatever the network takes
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
      calc_disease();
    };
  })(f);
  url_reader.readAsDataURL(f);
}

function update(evt){
  handleFileSelect(evt);
}

setup();
document.getElementById('input_image_file').addEventListener('change', update, false);

// interval_id = setInterval(calc_disease, 1000);
