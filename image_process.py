import matplotlib
import os
matplotlib.use('Agg')
from pycocotools.coco import COCO
import sys
from PIL import Image
import tensorflow as tf
import pdb

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("open_path",None,"The open file for images")
flags.DEFINE_string("save_path",None,"the save path for the modified images")

FLAGS = flags.FLAGS

assert(FLAGS.open_path)
assert(FLAGS.save_path)

open_path = FLAGS.open_path
save_path = FLAGS.save_path

image_list = [file for file in os.listdir(open_path) if os.path.isfile(os.path.join(open_path,file))]

final_size = 224,224

if not os.path.exists(save_path):
	os.makedirs(save_path)

for image_list_item in image_list:
	image_path = os.path.join(open_path,image_list_item)
	image = Image.open(image_path)
	img_size = image.size
	new_image = Image.new('RGB',final_size)
	image.thumbnail(final_size,Image.ANTIALIAS)
	new_image.paste(image,(0,0))
	new_image.save(os.path.join(save_path,image_list_item))
	pdb.set_trace()




