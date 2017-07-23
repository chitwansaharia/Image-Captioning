# Image-Captioning

Working on the analysis of state of the art Image Captioning Models using a pretrained VGG-19 Model on the MSCOCO Dataset.

## Requirements

Tensorflow
MSCOCO Dataset (can be downloaded from http://mscoco.org/dataset/)
pycocotools (clone from https://github.com/pdollar/coco)
PIL

## Processing Image

The MSCOCO images are of varying dimensions. To use the pretrained VGG-19 net we need to reshape the image to the input to the input shape of VGG net (224x224x3). 
For this, we reshape while maintaining the aspect ratio of images. (Padding image if necessary)
Use the following for saving the processed image in the required directory (the input and the oputput files can be changed inside the code)
``` bash 
python image_process.py 
```




