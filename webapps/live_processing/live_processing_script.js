// Notice there is no 'import' statement. 'tf' is available on the index-page
// because of the script tag above.

// learning from https://www.html5rocks.com/en/tutorials/getusermedia/intro/

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
