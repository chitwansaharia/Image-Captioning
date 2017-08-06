import matplotlib
import os
matplotlib.use('Agg')
from pycocotools.coco import COCO
import sys
from PIL import Image


final_size = 224,224

if str(sys.argv[1]) == 'valid':
	dataType = 'val2014'
	open_path = '/data/lisatmp4/chitwan/mscoco/val2014'
	save_path = '/data/lisatmp4/chitwan/mscoco/val2014_modified'
	save_dir = 'val2014_modified'
elif str(sys.argv[1]) == 'train':
	dataType = 'train2014'
	open_path = '/data/lisatmp4/chitwan/mscoco/train2014'
	save_path = '/data/lisatmp4/chitwan/mscoco/train2014_modified'
	save_dir = 'train2014_modified'
elif str(sys.argv[1]) == 'newimage':
	print("Hello")
	image = Image.open('images/a.jpg')
	new_image = Image.new('RGB',final_size)
	image.thumbnail(final_size,Image.ANTIALIAS)
	new_image.paste(image,(0,0))
	new_image.save('images/a_new.jpg')
	exit(0)
else:
	print("Bad Argument")
	exit(0)


annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_%s.json'%(dataType)

coco = COCO(annFile)


image_id_list = coco.getImgIds()

image_dict = coco.loadImgs(image_id_list)


os.makedirs(save_path)

for image_dict_item in image_dict:
	image_path = os.path.join(open_path,image_dict_item['file_name'])
	image = Image.open(image_path)
	img_size = image.size
	new_image = Image.new('RGB',final_size)
	image.thumbnail(final_size,Image.ANTIALIAS)
	new_image.paste(image,(0,0))
	new_image.save(os.path.join(save_path,image_dict_item['file_name']))




