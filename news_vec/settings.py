

# Models
LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 1
CNN_FILTER_SIZE = 100
CNN_FILTER_WIDTHS = (3, 4, 5)
ATTN_HIDDEN_SIZE = 200
CLF_EMBED_DIM = 512

# Trainer
BATCH_SIZE = 50
LR = 1e-4
EVAL_EVERY = None
ES_WAIT = 5

try:
    from .local_settings import *
except ModuleNotFoundError:
    pass
