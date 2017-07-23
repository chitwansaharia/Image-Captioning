from tensorflow_vgg import vgg19 
from tensorflow_vgg import utils
import tensorflow as tf
import numpy as np
from SS_Dataset import *
batch_size = 100
seed = 1234

iterator = SSIterator(batch_size,seed,False)

iterator.start()


with tf.device('/gpu:0'):
	sess = tf.Session()

	images = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

	vgg = vgg19.Vgg19('./tensorflow_vgg/Vgg19.npy')
	vgg.build(images)
	sess.run(tf.global_variables_initializer())

	batch = iterator.next()
	counter = 0
	file_id = 1
	vals_collect = []
	while batch != None:
		counter += 1
		vals = sess.run(vgg.fc7, feed_dict={images: batch[0]})
		vals = np.concatenate((vals,batch[1].reshape(batch_size,1)),axis = 1)
		vals_collect.append(vals)
		if counter % 100 ==0:
			file_name = "/data/lisatmp4/chitwan/mscoco/fc7_data/fc7_file_" + str(file_id)  + '.npy'
			np.save(file_name,vals_collect)
			vals_collect = []
			file_id += 1
		print("Iteration: " + str(counter) + " finished")
		batch = iterator.next()

	file_name = "/data/lisatmp4/chitwan/mscoco/fc7_data/fc7_file_" + str(file_id)  + '.npy'
	np.save(file_name,vals_collect)


# pdb.set_trace()




