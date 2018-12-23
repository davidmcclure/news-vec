

LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 1

CNN_FILTER_SIZE = 100
CNN_FILTER_WIDTHS = (3, 4, 5)

CLF_EMBED_DIM = 512

try:
    from .local_settings import *
except ModuleNotFoundError:
    pass
