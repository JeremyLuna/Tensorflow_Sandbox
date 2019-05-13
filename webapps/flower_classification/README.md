This webapp has two pages:
  camera - use a webcam to classify
  upload - upload an image to classify

To train on a dataset:
  delete trained_model and js_model
    these files will be generated
    the model is pretrained and downloaded from the INTERNET <3
  open train.py
    change image_dir to the path of your dataset
    models can be chosen from https://tfhub.dev/s?module-type=image-classification
    change any of the parameters as you like
  from this directory, run
    python train.py

output.txt can also be deleted. I just wanted to look at
the console output once.
