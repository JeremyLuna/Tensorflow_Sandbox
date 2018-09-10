// Notice there is no 'import' statement. 'tf' is available on the index-page
// because of the script tag above.

// learning from:
// https://www.html5rocks.com/en/tutorials/getusermedia/intro/
// https://js.tensorflow.org/tutorials/webcam-transfer-learning.html

function hasGetUserMedia() {
  return !!(navigator.mediaDevices &&
    navigator.mediaDevices.getUserMedia);
}

if (!hasGetUserMedia()) {
  alert('getUserMedia() is not supported by your browser');
} else {
  document.write("<video autoplay></video>")
  const video = document.querySelector('video');
  navigator.mediaDevices.getUserMedia({video:true}).
  then((stream) => {video.srcObject = stream});
}

// document.write("<p>loading</p>");
//
// const mobilenet = await tf.loadModel(
//   'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
//
// document.write("<p>done</p>");
// //The input size is [null, 224, 224, 3]
// const input_s = mobilenet.inputs[0].shape;
//
// //The output size is [null, 1000]
// const output_s = mobilenet.outputs[0].shape;
// document.write("<p>running</p>");
// var pred = mobilenet.predict(tf.zeros([1, 224, 224, 3]));
// document.write("<p>done</p>");
// document.write("<p> ", pred.argMax(), " </p>");
