import matplotlib
import os
matplotlib.use('Agg')
from pycocotools.coco import COCO
dataType='train2014'
annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_%s.json'%(dataType)

coco = COCO(annFile)

from PIL import Image

image_id_list = coco.getImgIds()

image_dict = coco.loadImgs(image_id_list)

final_size = 224,224

os.makedirs('train2014_modified')

for image_dict_item in image_dict:
	image_path = os.path.join('/data/lisatmp4/chitwan/mscoco/train2014',image_dict_item['file_name'])
	image = Image.open(image_path)
	img_size = image.size
	new_image = Image.new('RGB',final_size)
	image.thumbnail(final_size,Image.ANTIALIAS)
	new_image.paste(image,(0,0))
	new_image.save(os.path.join('/data/lisatmp4/chitwan/mscoco/train2014_modified',image_dict_item['file_name']))




