from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf
from tools import my_lib
from models.lstm_cell import myLSTMCell, _linear
import pdb



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



        self.initial_state = {}
        self.initial_state["token_encoder_lstm"] = token_encoder_cell.zero_state(batch_size, tf.float32)
#
        encoder_inputs = self.encoder_inputs


        vocabulary = ptb._build_vocab('/data/lisatmp4/chinna/data/ubuntu/Dataset.dict.pkl')
        file = open("/data/lisatmp4/chinna/Documents/UbuntuData/sg_vectors_ubuntu.txt")


        embedding_map = np.zeros((len(vocabulary.keys()),config.vocab_emd_size),dtype=np.float32)

        line = file.readline()

        for line in file:
            line = line.strip().split()
            if line[0] in vocabulary.keys():
                embedding_map[vocabulary[line[0]],:] = [float(i) for i in line[1:]]

        # embedding = tf.constant(value=embedding_map,name="embedding")

        embedding = tf.Variable(initial_value=embedding_map,name="embedding")

        weights_1 = tf.get_variable("weights_1", [300, 500], dtype=tf.float32)
        bias_1 = tf.get_variable("bias_1", [500], dtype=tf.float32)

        embedding = tf.add(tf.matmul(embedding,weights_1),bias_1)

        encoder_inputs = tf.nn.embedding_lookup(embedding, encoder_inputs)



        encoder_inputs = tf.nn.dropout(encoder_inputs, self.keep_prob)


        state = self.initial_state["token_encoder_lstm"]
        state_values_0 = []
        state_values_1 = []
        cell_output_values = []

        with tf.variable_scope("token_encoder_lstm", initializer=rand_uni_initializer):
            for time_step in range(max_tok_per_utr):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = token_encoder_cell([encoder_inputs[:,time_step,:]], state)
                state_values_0.append(state[0])
                state_values_1.append(state[1])
                cell_output_values.append(cell_output)





        cell_output_values = tf.stack(cell_output_values,1)
        state_values_0 = tf.stack(state_values_0,1)
        state_values_1 = tf.stack(state_values_1,1)

        cell_output_values = tf.reduce_sum(tf.multiply(cell_output_values,self.mask_encoder),1)
        state_values_0 = tf.reduce_sum(tf.multiply(state_values_0,self.mask_encoder),1)
        state_values_1 = tf.reduce_sum(tf.multiply(state_values_1,self.mask_encoder),1)

        final_cell_output = cell_output_values
        self.metrics["final_state_token_encoder_lstm"] = tf.nn.rnn_cell.LSTMStateTuple(state_values_0,state_values_1)






        turn_encoder_cell = myLSTMCell(turn_encoder_units,forget_bias=1.0,state_is_tuple=True)

        state =  self.initial_state["turn_encoder_lstm"] = turn_encoder_cell.zero_state(batch_size,tf.float32)

        with tf.variable_scope("turn_encoder_lstm", initializer=rand_uni_initializer):
            for time_step in range(max_num_turns):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = turn_encoder_cell([final_cell_output], state)


        self.metrics["final_state_turn_encoder_lstm"] = state
        final_cell_output_turn = cell_output


        decoder_cell = myLSTMCell(decoder_units, forget_bias=1.0, state_is_tuple=True)
        state = self.initial_state["decoder_lstm"] = decoder_cell.zero_state(batch_size, tf.float32)
        state_values_1 = []
        state_values_0 = []



        weights_2 = tf.get_variable("weights_2", [500, 500], dtype=tf.float32)
        bias_2 = tf.get_variable("bias_2", [500], dtype=tf.float32)

        embedding = tf.add(tf.matmul(embedding,weights_2),bias_2)

        decoder_inputs = tf.nn.embedding_lookup(embedding, self.decoder_inputs)


        decoder_inputs = tf.nn.dropout(decoder_inputs, self.keep_prob)


        self.decoder_outputs = []

        with tf.variable_scope("decoder_lstm", initializer=rand_uni_initializer):
            for time_step in range(max_tok_per_utr):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (cell_output, state) = decoder_cell([decoder_inputs[:,time_step,:],final_cell_output_turn], state)
                # (cell_output, state) = decoder_cell([decoder_inputs[:,time_step,:]], state)
                state_values_0.append(state[0])
                state_values_1.append(state[1])
                self.decoder_outputs.append(cell_output)

        state_values_0 = tf.stack(state_values_0,1)
        state_values_1 = tf.stack(state_values_1,1)

        state_values_0 = tf.reduce_sum(tf.multiply(state_values_0,self.mask_decoder),1)
        state_values_1 = tf.reduce_sum(tf.multiply(state_values_1,self.mask_decoder),1)

        self.metrics["final_state_decoder_lstm"] = tf.nn.rnn_cell.LSTMStateTuple(state_values_0,state_values_1)

        full_conn_layers = [tf.reshape(tf.concat(axis=1, values=self.decoder_outputs), [-1, decoder_units])]

        # pdb.set_trace()
        #full_conn_layers = [tf.stack(outputs, name='stacked_output')]
        with tf.variable_scope("output_layer"):
            self.model_logits = tf.contrib.layers.fully_connected(
                inputs=full_conn_layers[-1],
                num_outputs=num_penultimate,
                activation_fn=tf.nn.relu,
                weights_initializer=rand_uni_initializer,
                biases_initializer=rand_uni_initializer,
                trainable=True)
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
            "grad_sum": self.metrics["grad_sum"],
            "token_encoder_lstm": self.metrics["final_state_token_encoder_lstm"],
            "turn_encoder_lstm": self.metrics["final_state_turn_encoder_lstm"],
            "decoder_lstm": self.metrics["final_state_decoder_lstm"],

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
            feed_dict[self.y.name] = batch["decoder_outputs"]
            feed_dict[self.wts.name] = batch["mask"]
            feed_dict[self.encoder_inputs.name] = batch["encoder_inputs"]
            feed_dict[self.decoder_inputs.name] = batch["decoder_inputs"]
            feed_dict[self.keep_prob.name] = keep_prob
            feed_dict[self.phase_train.name] = phase_train
            feed_dict[self.mask_encoder] = batch["mask_encoder"]
            feed_dict[self.mask_decoder] = batch["mask_decoder"]

            # pdb.set_trace()


            # if is_training:
            #     feed_dict[self.decoder_input_decider.name] = np.random.random_sample()
            # else:
            #     feed_dict[self.decoder_input_decider.name] = 1

            if batch["reset"]:
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

            # pdb.set_trace()
            for k in state:
                state[k] = vals[k]

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
