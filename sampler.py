#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")

import tensorflow as tf

import sys
import os
from models import image_captioning_sampler
from train_iter import *
import imp
import pdb

from pycocotools.coco import COCO
from tensorflow_vgg import utils

import cPickle

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("log_path",None,"Log Directory path")


FLAGS = flags.FLAGS


annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_val2014.json'
coco = COCO(annFile)

imgid_list = coco.getImgIds()
image = coco.loadImgs(imgid_list[150]) 
print(image[0]['file_name'])

image = utils.load_image(os.path.join('/data/lisatmp4/chitwan/mscoco/val2014_modified/',image[0]['file_name']))


vocabulary = cPickle.load(open('/data/lisatmp4/chitwan/mscoco/caption_processed/vocabulary.pkl','r'))



def main(_):
    model_config = imp.load_source('config', 'config/config.py').config().image


    log_path = os.path.join(FLAGS.log_path,'hybrid_log')

    with tf.Graph().as_default():
        main_model = image_captioning_sampler.ImageCaptioning(model_config)

        sv = tf.train.Supervisor( logdir=log_path, init_feed_dict=main_model.init_feed_dict())
        with sv.managed_session() as session:
                final_caption = main_model.sample(session, image,vocabulary, is_training=False, verbose=True)
        print(' '.join(word for word in final_caption))


if __name__ == "__main__":
    tf.app.run()
