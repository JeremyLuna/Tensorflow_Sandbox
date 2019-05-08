This webapp has two pages:
  camera - use a webcam to classify
  upload - upload an image to classify

To train on a dataset:
  delete trained_model and js_model
  open train.py
    change image_dir to the path of your dataset
    change any of the parameters as you like
  from this directory, run
    python train.py
