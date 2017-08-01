from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
sys.path.append('/u/sahariac/Image-Captioning/')

import time
import numpy as np
import tensorflow as tf
from tools import my_lib
from models.lstm_cell import myLSTMCell, _linear
import pdb
from tensorflow_vgg import vgg19_trainable as vgg19

vgg_fc7_layer = 4096


class ImageCaptioning(object):

    def __init__(self, config, scope_name=None, device='gpu'):
        self.config = config
        self.scope = scope_name or "HybridDialogModel"

        self.create_placeholders()
        self.global_step = \
            tf.contrib.framework.get_or_create_global_step()

        self.metrics = {}
        if device == 'gpu':
            tf.device('/gpu:0')
        else:
            tf.device('/cpu:0')

        with tf.variable_scope(self.scope):
            self.build_model()

    def create_placeholders(self):
        batch_size = self.config.batch_size
        max_tokens_per_caption = self.config.max_tokens_per_caption
        image_height = self.config.image_height
        image_width = self.config.image_width
        image_channels = self.config.image_channels

        # input_data placeholders
        self.conv_inputs = tf.placeholder(
            tf.float32, shape=[1,image_height,image_width,image_channels], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(
            tf.int64, shape=[1,1], name="decoder_inputs")
        
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name="phase_train")
        self.vgg_train = tf.placeholder(tf.bool, name="vgg_train")



    def build_model(self):
        config = self.config
        batch_size = config.batch_size
        decoder_units = config.decoder_units
        vocab_size = config.vocab_size
        input_size = config.input_size
        max_tokens_per_caption = config.max_tokens_per_caption

        rand_uni_initializer = \
            tf.random_uniform_initializer(
                -self.config.init_scale, self.config.init_scale)


        self.initial_state = {}


        self.vgg = vgg19.Vgg19('/u/sahariac/Image-Captioning/tensorflow_vgg/Vgg19.npy')

        self.vgg.build(self.conv_inputs,self.vgg_train)

        weights_1 = tf.get_variable("weights_1", [vgg_fc7_layer, decoder_units], dtype=tf.float32)
        bias_1 = tf.get_variable("bias_1", [decoder_units], dtype=tf.float32)

        weights_2 = tf.get_variable("weights_2", [vgg_fc7_layer, decoder_units], dtype=tf.float32)
        bias_2 = tf.get_variable("bias_2", [decoder_units], dtype=tf.float32)

        vgg_fc7_c = tf.add(tf.matmul(self.vgg.fc7,weights_1),bias_1)
        vgg_fc7_h = tf.add(tf.matmul(self.vgg.fc7,weights_2),bias_2)

        self.metrics["vggfc7"] = self.vgg.fc7

        self.metrics["final_vgg_state"]  = tf.nn.rnn_cell.LSTMStateTuple(vgg_fc7_c,vgg_fc7_h)

        embedding = tf.get_variable("embedding",[vocab_size,input_size])

        decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs)


        decoder_inputs = tf.nn.dropout(decoder_inputs, self.keep_prob)

        decoder_cell = myLSTMCell(decoder_units, forget_bias=1.0, state_is_tuple=True)

        state = self.initial_state["decoder_state"] = decoder_cell.zero_state(1,tf.float32) 

        self.decoder_outputs = []
        with tf.variable_scope("decoder_lstm", initializer=rand_uni_initializer):
            (cell_output, state) = decoder_cell([decoder_inputs[:,0,:]], state)
            self.decoder_outputs.append(cell_output)
        self.metrics["final_state"] = state
        

        full_conn_layers = [tf.reshape(tf.concat(axis=1, values=self.decoder_outputs), [-1, decoder_units])]


        self.metrics["full_conn"] = full_conn_layers[-1]        
        with tf.variable_scope("output_layer"):
            self.model_logits = tf.contrib.layers.fully_connected(
                    inputs=full_conn_layers[-1],
                    num_outputs=vocab_size,
                    activation_fn=None,
                    weights_initializer=rand_uni_initializer,
                    biases_initializer=rand_uni_initializer,
                    trainable=True)

            self.metrics["model_prob"] = tf.nn.softmax(self.model_logits)


    def model_vars(self):
        return self.tvars

    def init_feed_dict(self):
        return {self.phase_train.name: True}


    def sample(self, session, image,vocabulary, is_training=False, verbose=False):
        epoch_metrics = {}
        keep_prob = 1.0
        fetches = {
                    "model_prob" : self.metrics["model_prob"],
                    "final_state" : self.metrics["final_state"]

        }
        phase_train = 1.0

        index_to_word = {v: k for k, v in vocabulary.iteritems()}
        image = image.reshape([1,224,224,3])
        final_vgg_state = session.run(self.metrics["final_vgg_state"],feed_dict={self.conv_inputs.name : image,self.vgg_train.name : False})
        
        feed_dict = {}
        feed_dict[self.initial_state["decoder_state"]] = final_vgg_state
        token = np.zeros((1,1),dtype=np.int)
        token[0,0] = vocabulary['START_TOKEN']
        feed_dict[self.decoder_inputs.name] = token
        feed_dict[self.keep_prob.name] = 1.0
        vals = session.run(fetches,feed_dict)
        last_state = vals["final_state"]
        final_caption = ['START_TOKEN']
        final_caption.append(index_to_word[np.argmax(vals["model_prob"])])

        while index_to_word[np.argmax(vals["model_prob"])] != 'STOP_TOKEN':


            feed_dict = {}
            feed_dict[self.initial_state["decoder_state"]] = last_state
            token[0,0] = np.argmax(vals["model_prob"])
            feed_dict[self.decoder_inputs.name] =  token
            feed_dict[self.keep_prob.name] = 1.0
            vals = session.run(fetches, feed_dict)
            last_state = vals["final_state"]
            final_caption.append(index_to_word[np.argmax(vals["model_prob"])])

        return final_caption[1:-1]

            

        


