// Notice there is no 'import' statement. 'tf' is available on the index-page
// because of the script tag above.

// learning from:
// https://www.html5rocks.com/en/tutorials/getusermedia/intro/
// https://js.tensorflow.org/tutorials/webcam-transfer-learning.html :
//      https://github.com/tensorflow/tfjs-examples/tree/master/webcam-transfer-learning
// https://medium.com/tensorflow/a-gentle-introduction-to-tensorflow-js-dba2e5257702

import {Webcam} from './webcam';

navigator.mediaDevices.getUserMedia({video: true}).then((stream) => {video.srcObject = stream});

// A webcam class that generates Tensors from the images from the webcam.
const webcam = new Webcam(document.getElementById('webcam'));
webcam.setup();

// while (true) {
//     document.getElementById("brightness").innerHTML = "Have a nice day!";
// }
