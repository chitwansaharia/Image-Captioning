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
from tenforflow_vgg import vgg19_traininable as vgg19

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
            self.compute_loss_and_metrics()
            self.compute_gradients_and_train_op()

    def create_placeholders(self):
        batch_size = self.config.batch_size
        max_tokens_per_caption = self.config.max_tokens_per_caption
        uttr_emb_size = self.config.uttr_emb_size
        image_height = self.config.image_height
        image_width = self.config.image_width
        image_channels = self.config.image_channels

        # input_data placeholders
        self.conv_inputs = tf.placeholder(
            tf.int64, shape=[batch_size,image_height,image_width,image_channels], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(
            tf.int64, shape=[batch_size,max_tokens_per_caption], name="decoder_inputs")
        self.y = tf.placeholder(
            tf.int64, shape=[batch_size,max_tokens_per_caption], name="targets")
        self.wts = tf.placeholder(
            tf.float32, shape=[batch_size,max_tokens_per_caption], name="weights")
        self.mask_decoder =  tf.placeholder(tf.float32,shape=[batch_size,max_tokens_per_caption,self.config.decoder_units],name="mask_decoder")

        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.phase_train = tf.placeholder(tf.bool, name="phase_train")
        self.vgg_train = tf.placeholder(tf.bool, name="vgg_train")
        self.iter_count = tf.Variable(initial_value = 0,dtype=tf.int64, name="iter_count")
        self.inc = tf.assign_add(self.iter_count, 1, name='increment')
        self.refresh = tf.assign(self.iter_count,0)



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

        self.vgg = vgg19.Vgg19('../tensorflow_vgg/Vgg19.npy')

        self.vgg.build(self.conv_inputs, self.vgg_train)

        weights_1 = tf.get_variable("weights_1", [vgg_fc7_layer, decoder_units], dtype=tf.float32)
        bias_1 = tf.get_variable("bias_1", [decoder_units], dtype=tf.float32)

        vgg_fc7 = tf.add(tf.matmul(self.vgg.fc7,weights_1),bias_1)

        


  

        # embedding = tf.constant(value=embedding_map,name="embedding")

        embedding = tf.get_variable("embedding",[vocab_size,input_size],name="embedding")

        decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs)



        decoder_inputs = tf.nn.dropout(decoder_inputs, self.keep_prob)


        decoder_cell = myLSTMCell(decoder_units, forget_bias=1.0, state_is_tuple=True)
        state = vgg_fc7

        self.decoder_outputs = []

        with tf.variable_scope("decoder_lstm", initializer=rand_uni_initializer):
            for time_step in range(max_tok_per_utr):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = decoder_cell([decoder_inputs[:,time_step,:]], state)
                self.decoder_outputs.append(cell_output)

        full_conn_layers = [tf.reshape(tf.concat(axis=1, values=self.decoder_outputs), [-1, decoder_units])]

        # pdb.set_trace()
        #full_conn_layers = [tf.stack(outputs, name='stacked_output')]
        with tf.variable_scope("output_layer"):
            self.model_logits = tf.contrib.layers.fully_connected(
                inputs=self.model_logits,
                num_outputs=vocab_size,
                activation_fn=None,
                weights_initializer=rand_uni_initializer,
                biases_initializer=rand_uni_initializer,
                trainable=True)

            self.metrics["model_prob"] = tf.nn.softmax(self.model_logits)

    def compute_loss_and_metrics(self):
        # entropy_loss = tf.contrib.seq2seq.sequence_loss(
        #    logits=self.model_logits,
        #    targets=self.y,
        #    weights=self.wts,
        #    average_across_timesteps=False,
        #    average_across_batch=False)

        entropy_loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.model_logits],
            [tf.reshape(self.y, [-1])],
            [tf.reshape(self.wts, [-1])],
            average_across_timesteps=False)

        self.metrics["loss"] = tf.reduce_sum(entropy_loss)

    def compute_gradients_and_train_op(self):
        tvars = self.tvars = my_lib.get_scope_var(self.scope)
        my_lib.get_num_params(tvars)
        grads = tf.gradients(self.metrics["loss"], tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.config.max_grad_norm)

        self.metrics["grad_sum"] = tf.add_n([tf.reduce_sum(g) for g in grads])

        optimizer = tf.train.AdamOptimizer(learning_rate = self.config.learning_rate)
        self.train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=self.global_step)

    def model_vars(self):
        return self.tvars

    def init_feed_dict(self):
        return {self.phase_train.name: True}

    def run_epoch(self, session,reader, is_training=False, verbose=False):
        start_time = time.time()
        epoch_metrics = {}
        keep_prob = 1.0
        fetches = {
            "loss": self.metrics["loss"],
            "grad_sum": self.metrics["grad_sum"]

        }

        if is_training:
            if verbose:
                print("\nTraining...")
            fetches["train_op"] = self.train_op
            keep_prob = self.config.keep_prob
            phase_train = True
        else:
            phase_train = False
            if verbose:
                print("\nEvaluating...")



        i, total_loss, grad_sum, total_words = 0, 0.0, 0.0, 0.0
        state = {}
        for k, v in self.initial_state.items():
            state[k] = session.run(v)

        reader.start()

        i = 0
        print("Reading till the Bactch Number")
        iter_count_val = session.run(self.iter_count)
        while i < iter_count_val:
            batch = reader.next()
            i+=1
        print("Resuming at Batch Number = %d" % iter_count_val)
        batch = reader.next()
        while batch != None:

            session.run(self.inc)
            feed_dict = {}
            feed_dict[self.y.name] = batch["caption_batch_y"]
            feed_dict[self.wts.name] = batch["mask"]
            feed_dict[self.decoder_inputs.name] = batch["caption_batch_x"]
            feed_dict[self.keep_prob.name] = keep_prob
            feed_dict[self.phase_train.name] = phase_train
            feed_dict[self.conv_inputs.name] = batch['image_batch']

            # pdb.set_trace()


            # if is_training:
            #     feed_dict[self.decoder_input_decider.name] = np.random.random_sample()
            # else:
            #     feed_dict[self.decoder_input_decider.name] = 1

            state = {}
            for k, v in self.initial_state.items():
                state[k] = session.run(v)

            for k in state:
                feed_dict[self.initial_state[k].c] = state[k].c
                feed_dict[self.initial_state[k].h] = state[k].h

            state = {}
            for k, v in self.initial_state.items():
                state[k] = session.run(v)

            vals = session.run(fetches, feed_dict)

            total_loss += vals["loss"]
            grad_sum += vals["grad_sum"]
            total_words += np.sum(batch["mask"]) * 1.0

            i += 1
            percent_complete = (i * 100.0) / self.config.iters_per_epoch
            perplexity = np.exp(total_loss / total_words)

            if verbose:
                print(
                    "% Complete :", round(percent_complete, 0),
                    "HybridDialogModel model : perplexity :", round(perplexity, 3),
                    "loss :", round((total_loss / total_words), 3), \
                    # "grad_sum: ", round(grad_sum, 3), \
                    "words/sec :", round((total_words) / (time.time() - start_time), 0),
                    "Gradient :", round(vals["grad_sum"],3))
            batch = reader.next()

        session.run(self.refresh)

        epoch_metrics["loss"] = round(np.exp(total_loss / total_words), 3)
        return epoch_metrics


    def run_step(self, session, word_id, model_state=None):
        epoch_metrics = {}
        keep_prob = 1.0
        phase_train = False
        fetches = {
            "model_prob": self.metrics["model_prob"],
            # "final_state_encoder_lstm": self.metrics["final_state_encoder_lstm"],
            "final_state_decoder_lstm": self.metrics["final_state_decoder_lstm"]
        }

        state = {}
        if not model_state:
            for k in self.initial_state:
                state[k] = session.run(self.initial_state[k])
        else:
            for k in self.initial_state:
                state[k] = model_state["final_state_{}".format(k)]

        feed_dict = {}
        feed_dict[self.x.name] = np.array([[word_id]])
        feed_dict[self.wts.name] = np.ones((1, 1))
        for k in self.initial_state:
            feed_dict[self.initial_state[k].c] = state[k].c
            feed_dict[self.initial_state[k].h] = state[k].h
        feed_dict[self.keep_prob.name] = keep_prob
        feed_dict[self.phase_train.name] = phase_train

        vals = session.run(fetches, feed_dict)
        vals["model_prob"] = np.squeeze(vals["model_prob"])

        return vals
