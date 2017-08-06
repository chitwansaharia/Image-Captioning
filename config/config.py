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
    config.load_mode = "continue"
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

def image_caption_config():
    config = default_config()
    config.batch_size  = 50
    config.max_tokens_per_caption = 45
    config.decoder_units = 2000
    config.vocab_size = 15002
    config.input_size = 300
    config.save_path = '/data/lisatmp4/chitwan/mscoco/saved_weights/'
    config.model = "image_captioning_model.ImageCaptioning"
    config.image_height = 224
    config.image_width = 224
    config.image_channels = 3
    config.learning_rate = 0.0001
    config.num_iters = 414113/config.batch_size
    config.use_infinite_loop = False
    config.save_path = 'Best'
    return config

def train_file():
    config = image_caption_config()
    config.annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_train2014.json'
    config.image_path = '/data/lisatmp4/chitwan/mscoco/train2014_modified'
    config.caption_file = '/data/lisatmp4/chitwan/mscoco/caption_processed/processed_captions_train.pkl'
    return config

def valid_file():
    config = image_caption_config()
    config.annFile = '/data/lisatmp4/chitwan/mscoco/annotations/captions_val2014.json'
    config.image_path = '/data/lisatmp4/chitwan/mscoco/val2014_modified'
    config.caption_file = '/data/lisatmp4/chitwan/mscoco/caption_processed/processed_captions_valid.pkl'
    return config

def image_caption_config_attn():
    config = image_caption_config()
    config.model = "image_captioning_with_attention.ImageCaptioning"
    config.decoder_units = 1024
    return config    


def config():
    config = config_container()
    config.sampler = sampler_config()
    config.image = image_caption_config()
    config.train = train_file()
    config.valid = valid_file()
    config.image_attn = image_caption_config_attn()
    return config


if __name__ == '__main__':
    config = config()
    print config
