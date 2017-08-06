import matplotlib
matplotlib.use('Agg')
from pycocotools.coco import COCO 
import numpy as np
import pdb
import cPickle
from nltk.tokenize import word_tokenize
import itertools
from collections import Counter
import sys
import tensorflow as tf
import os

vocab_size = 15000

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("annFile",None,"The annotation file (from mscoco)")
flags.DEFINE_string("save_path",None,"the save  path for processed dialogues")
flags.DEFINE_string("make_vocabulary","False","true if you want to make new vocabulary")
flags.DEFINE_string("vocab_path",None,"path of vocabulary file")


FLAGS = flags.FLAGS

assert(FLAGS.annFile)
assert(FLAGS.save_path)
assert(FLAGS.vocab_path)

annFile = FLAGS.annFile
save_path = FLAGS.save_path
vocab_path = FLAGS.vocab_path

coco = COCO(annFile)

image_ids = coco.getImgIds()
ann_ids = coco.getAnnIds()
captions = coco.loadAnns(ann_ids)

if not os.path.exists(save_path):
	os.makedirs(save_path)


if not os.path.exists(vocab_path):
	os.makedirs(vocab_path)



def vocabulary():
	pdb.set_trace()
	text = [item['caption'].encode('ascii','ignore') for item in captions]
	tokenised_text = map(lambda text_item :word_tokenize(text_item), text)
	all_words = list(itertools.chain.from_iterable(tokenised_text))
	word_dict = dict(Counter(all_words).most_common(vocab_size))
	index_to_word = dict(enumerate(word_dict.keys()))
	word_to_index = {v: k for k, v in index_to_word.items()}
	word_to_index['UNKNOWN'] = 15000
	word_to_index['STOP_TOKEN'] = word_to_index['.']
	word_to_index.pop('.',None)
	word_to_index['START_TOKEN'] = 15001
	with open(os.path.join(vocab_path,"Vocabulary.pkl"),'wb') as file:
		cPickle.dump(word_to_index, file)

if FLAGS.make_vocabulary == "True":
	vocabulary()


with open(os.path.join(vocab_path,"Vocabulary.pkl"),'rb') as file:
		word_to_index = cPickle.load(file)

tokenized_index = []
text = [(item['id'],item['caption'].encode('ascii','ignore')) for item in captions]
tokenised_text = dict(map(lambda text_item :(text_item[0],word_tokenize(text_item[1])), text))
final_dict = {}
total = len(tokenised_text)
count = 0
for _id,item in tokenised_text.items():
	ind_sent = []
	ind_sent.append(word_to_index['START_TOKEN'])
	for word in item:
		if word in word_to_index.keys():
			ind_sent.append(word_to_index[word])
		elif word == '.':
			ind_sent = ind_sent
		else:
			ind_sent.append(word_to_index['UNKNOWN'])
	ind_sent.append(word_to_index['STOP_TOKEN'])
	final_dict[_id] = ind_sent
	if count % 5000 == 0:
		print("Complete Percent " + str(round(count*100/total,3)))
	count += 1
	pdb.set_trace()

with open(os.path.join(save_path,"Processed_Captions.pkl"),'wb') as file:
	cPickle.dump(final_dict, file)










