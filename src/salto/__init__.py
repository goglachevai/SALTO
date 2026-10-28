import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")

from .c22 import C22Space, c22_features, c22_features_batch
from .classifier import METRICS, Salto, SaltoStream
from .data import block_split, build_training_set, load_pastila, load_series, subsequences, znorm
from .model import Extractor, rmse_loss
from .mpdist import mpdist_matrix
