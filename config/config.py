import os



class config_container(object):

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            if value:
                yield value

    def __getattr__(self, item):
        return None

    def _to_dict(self):
        d = {}
        for attr, value in self.__dict__.iteritems():
            if isinstance(value, config_container):
                d[attr] = value._to_dict()
            else:
                d[attr] = value
        return d

    def __repr__(self):
        import json
        return json.dumps(self._to_dict(), indent=2)


def sampler_config():
    config = config_container()
    config.start_phrases = ["an", "the", "what"]
    config.num_samples = 1
    config.num_steps = 20
    return config


def default_config():
    config = config_container()
    config.load_mode = "best"
    config.patience = 3
    config.init_scale = 0.1
    config.max_grad_norm = 5
    config.num_layers = 1
    config.num_steps = 100
    config.hidden_size = 800
    config.input_size = 300
    config.s_dim = 1000
    config.uttr_emb_size = config.hidden_size
    config.vocab_emd_size = 300
    config.keep_prob = 0.65
    config.batch_size = 80
    config.vocab_size = 10000
    config.attn_span = 10
    config.max_turn_per_row = 1
    config.max_token_per_utr = 100
    return config

def hybrid_config():
    config = default_config()
    config.vocab_size = 20000
    config.decoder_units = 2000
    config.turn_encoder_units = 500
    config.token_encoder_units = 1000
    config.use_infinite_loop = False
    config.seed = 1234
    config.iters_per_epoch = 5000
    config.num_penultimate = 500
    config.reader = "wiki.wiki_reader"
    config.save_path = "hybrid/default/"
    config.data_path_train = 'Data/100/Train_modified.pkl'
    config.data_path_valid = 'Data/100/Valid_modified.pkl'
    config.model = "end_to_end_dialog_model.HybridDialogModel"
    config.load_mode = "continue"
    config.learning_rate = 0.0002
    return config

def hybrid_attention_config():
    config = hybrid_config()
    config.encoder_units = 500
    config.decoder_units = 1200
    config.model = "hybrid_model_with_attention.HybridDialogModel"
    return config



def hybrid_config_1():
    config = hybrid_config()
    return config

def hybrid_config_2():
    config = hybrid_config()
    config.data_path_train = 'Data/80/Train_modified.pkl'
    config.data_path_valid = 'Data/80/Valid_modified.pkl'
    config.max_token_per_utr = 80
    return config

def hybrid_config_3():
    config = hybrid_config()
    config.decoder_units = 800
    return config


def config():
    config = config_container()
    config.dialog = hybrid_config()
    config.hybrid_config_1 = hybrid_config_1()
    config.hybrid_config_2 = hybrid_config_2()
    config.hybrid_config_3 = hybrid_config_3()
    config.attn = hybrid_attention_config()
    config.sampler = sampler_config()
    return config


if __name__ == '__main__':
    config = config()
    print config
