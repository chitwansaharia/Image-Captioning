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
python image_process.py '<train/valid>'
```
## Processing Captions

The pycocotool assigns a caption id to each caption and each image has 5 captions. The text-process.py takes the captions (in the form of words through the pycocotool) and then builds a vocabulary (15000 tokens) and stores vocabulary and the index form captions in a pkl file (the path of the file can be changed in the code).
``` bash 
python text-process.py '<train/valid>'
```
## Training the model

The model used for training model/image_captioning_model.py uses the pretrained VGG-net which is trainable through the vgg_train flag. It uses a LSTM decoder for caption generation. The configuration of the network can be changed from the config/config.py file. For running the model use the following command. (An example command has been give in script.sh).
``` bash 
python train_hybrid.py --save_path="<save-path>" --log_path="<log-path>"
```



