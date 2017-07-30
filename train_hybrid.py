#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(1,"/u/sahariac/.local/lib/python2.7/site-packages/")

import tensorflow as tf

import sys
import os
from models import image_captioning_model
# from data_readers import ptb, wiki
from train_iter import *
import imp
# import Data_iter
import pdb

flags = tf.flags
logging = tf.logging

# flags.DEFINE_string("config", "1",
#                     "config params for the model")
flags.DEFINE_string("save_path", None,
                    "base save path for the experiment")
flags.DEFINE_string("eval_only", "False",
                    "Only Evaluate.No Training.")
flags.DEFINE_string("log_path",None,"Log Directory path")


FLAGS = flags.FLAGS


# if FLAGS.config == '1':
#     model_config = imp.load_source('config', 'config/config.py').config().config_1
# elif FLAGS.config == '2':
#     model_config = imp.load_source('config', 'config/config.py').config().config_2
# elif FLAGS.config == '3':
#     model_config = imp.load_source('config', 'config/config.py').config().config_3



def main(_):
    # assert(FLAGS.save_path)
    # sent_config = imp.load_source('config', FLAGS.config).config().lm

    model_config = imp.load_source('config', 'config/config.py').config().image

    file_train_config  = imp.load_source('config', 'config/config.py').config().train
    file_valid_config  = imp.load_source('config', 'config/config.py').config().valid

    print("Config :")
    print(model_config)
    print("\n")

    save_path = os.path.join(FLAGS.save_path, model_config.save_path)
    log_path = os.path.join(FLAGS.log_path,'hybrid_log')

    with tf.Graph().as_default():
        main_model = eval(model_config.model)(model_config)
        # pdb.set_trace()
        # sent_model = eval(sent_config.model)(sent_config,None)

        if model_config.load_mode == "continue":
            if not tf.gfile.Exists(save_path):
                os.makedirs(save_path)
                os.chmod(save_path, 0775)
        model_vars = main_model.model_vars()
        # model_vars.remove(<tf.Variable 'HybridDialogModel/weights_embedding:0' shape=(300, 300) dtype=float32_ref>)
        # model_vars.remove(<tf.Variable 'HybridDialogModel/bias_embedding:0' shape=(300,) dtype=float32_ref>)

        # m = model_vars
        # del model_vars[9]
        # del model_vars[9]
        # pre_train_saver = tf.train.Saver(model_vars)

        # pdb.set_trace()



        def load_pretrain(sess):
            model_ckpt_loc = os.path.join(log_path,"model.ckpt")
            print("Restoring model at...", model_ckpt_loc)

            pre_train_saver.restore(sess, model_ckpt_loc)

        sv = tf.train.Supervisor( logdir=log_path, init_feed_dict=main_model.init_feed_dict())
        
        with sv.managed_session() as session:
            if model_config.load_mode == "best":
                sv.saver.restore(
                    sess=session,
                    save_path=os.path.join(save_path, "best_model.ckpt"))


            i, patience = 0, 0
            best_valid_metric = 1e10

            while patience < model_config.patience and not eval(FLAGS.eval_only):
                i += 1
                # train_reader = Data_iter.get_train_iterator(dial_config)
                # valid_reader = Data_iter.get_validation_iterator(dial_config)
                iterator_train = SSIterator(model_config.batch_size,file_train_config,model_config.max_tokens_per_caption,1234)
                iterator_valid = SSIterator(model_config.batch_size,file_valid_config,model_config.max_tokens_per_caption,1234)
                
                print("\nEpoch: %d" % (i))
                main_model.run_epoch(session, reader = iterator, is_training=True, verbose=True)

                # valid_metrics = main_model.run_epoch(session, reader = valid_reader, verbose=True)

                # if best_valid_metric > valid_metrics["loss"]:
                #     best_valid_metric = valid_metrics["loss"]

                #     print("\nsaving best model...")
                #     sv.saver.save(sess=session, save_path=os.path.join(save_path, "best_model.ckpt"))
                #     patience = 0
                # else:
                #     patience += 1
                #     print("\nLosing patience...")

            # print("\nLoading best model and Evaluating with Test set...")
            # sv.saver.restore(sess=session, save_path=os.path.join(save_path, "best_model.ckpt"))
            # model.run_epoch(session, data=data["test"], verbose=True)

            if FLAGS.save_path and dial_config.load_mode == "fresh":
                print("\nSaving model to %s." % save_path)
                sv.saver.save(session, save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
