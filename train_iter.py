import sys
import numpy  as np
import os, gc
import cPickle
import copy
import logging


import threading
import Queue

import collections
import os

import matplotlib
import os
matplotlib.use('Agg')
from pycocotools.coco import COCO
from PIL import Image
from tensorflow_vgg import utils

import pdb


logger = logging.getLogger(__name__)

class SSFetcher(threading.Thread):
    def __init__(self, parent):
        threading.Thread.__init__(self)
        self.parent = parent

    def run(self):
        diter = self.parent
        offset = 0 
        i = 0
        while not diter.exit_flag:
            last_batch = False
            image_batch = np.zeros((diter.batch_size,224,224,3))
            caption_batch = np.zeros((diter.batch_size,diter.max_caption_length))
            mask = np.zeros((diter.batch_size,diter.max_caption_length))
            counter = 0
            while counter < diter.batch_size:
                if offset == diter.num_images:
                    if not diter.use_infinite_loop:
                        print("Hello")
                        last_batch = True
                        diter.queue.put(None)
                        return
                    else:
                    # Infinite loop here, we reshuffle the indexes
                    # and reset the offset
                    # self.rng.shuffle(self.indexes)
                        offset = 0
                        print("End")

                
                image = utils.load_image(os.path.join(diter.image_path,diter.image_dict[offset]['file_name']))
                caption_ids = diter.coco.getAnnIds(diter.image_dict[offset]['id'])
                for _id in caption_ids:
                    # pdb.set_trace()
                    caption = diter.processed_captions[_id]
                    if len(caption) > diter.max_caption_length:
                        caption = caption[:diter.max_caption_length-1]
                        caption.append(412)
                    mask[counter,:len(caption)] = 1
                    caption_batch[counter,:len(caption)] = caption
                    image_batch[counter,:,:,:] = image.reshape((224, 224, 3))
                    counter += 1
                offset += 1

            if counter == diter.batch_size:
                print(i)
                diter.queue.put((image_batch,caption_batch,mask))
                i+=1

            if last_batch:
                diter.queue.put(None)
                return

class SSIterator(object):
    def __init__(self,
                 batch_size,
                 max_caption_length,
                 seed,
                 use_infinite_loop=True,
                 dtype="int32"):

        self.batch_size = batch_size
        self.max_caption_length = max_caption_length

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.load_files()
        self.exit_flag = False

    def load_files(self):
        self.dataType='train2014'
        self.annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_%s.json'%(self.dataType)
        self.coco = COCO(self.annFile)
        image_id_list = self.coco.getImgIds()
        self.image_dict = self.coco.loadImgs(image_id_list)
        self.image_path = "/data/lisatmp4/chitwan/mscoco/train2014_modified/"
        self.caption_path = "/data/lisatmp4/chitwan/mscoco/caption_processed/processed_captions.pkl"
        self.processed_captions = cPickle.load(open(self.caption_path,'r'))
        self.num_images = len(self.image_dict)

    def start(self):
        self.exit_flag = False
        self.queue = Queue.Queue(maxsize = 100)
        self.gather = SSFetcher(self)
        self.gather.daemon = True
        # print("Hello")
        self.gather.start()

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if self.exit_flag:
            return None
        
        batch = self.queue.get()
        if not batch:
            self.exit_flag = True
            # print("Okay")
        return batch
