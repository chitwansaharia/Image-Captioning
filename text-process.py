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

if str(sys.argv[1]) == 'valid':
	dataType = 'val2014'
	save_path = '/data/lisatmp4/chitwan/mscoco/caption_processed/processed_captions_valid.pkl'
elif str(sys.argv[1]) == 'train':
	dataType = 'train2014'
	save_path = '/data/lisatmp4/chitwan/mscoco/caption_processed/processed_captions_train.pkl'
else:
	print("Bad argument")
	exit(0)

annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_%s.json'%(dataType)
coco = COCO(annFile)

image_ids = coco.getImgIds()
ann_ids = coco.getAnnIds()
captions = coco.loadAnns(ann_ids)

def vocabulary():
	text = [item['caption'].encode('ascii','ignore') for item in captions]
	tokenised_text = map(lambda text_item :word_tokenize(text_item), text)
	all_words = list(itertools.chain.from_iterable(tokenised_text))
	vocab_size  = 15000
	word_dict = dict(Counter(all_words).most_common(vocab_size))
	index_to_word = dict(enumerate(word_dict.keys()))
	word_to_index = {v: k for k, v in index_to_word.items()}
	word_to_index['UNKNOWN'] = 15000
	word_to_index['STOP_TOKEN'] = word_to_index['.']
	word_to_index.pop('.',None)
	word_to_index['START_TOKEN'] = 15001
	with open('/data/lisatmp4/chitwan/mscoco/caption_processed/vocabulary.pkl','wb') as file:
		cPickle.dump(word_to_index, file)

with open('/data/lisatmp4/chitwan/mscoco/caption_processed/vocabulary.pkl','rb') as file:
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
	# pdb.set_trace()

with open(save_path,'wb') as file:
	cPickle.dump(final_dict, file)










