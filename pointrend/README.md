### PointRend single human segmentation wrapper
Usage: `python pointrend.py <image_file1> <image_file2> ...`

Segmentation mask `<image_file>_mask.png` will be written for each file if a human is detected.

### Setup
`pip install numpy opencv-python torch`
You'll also need to install detectron2 for the appropriate CUDA+pytorch versions as described:
https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

