import os
import torch
import time
import numpy as np
import json
def set_deterministic(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # torch.set_deterministic(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_ssl_type(ssl_type):
    ssl_book={
        "wav2vec2-base": "facebook/wav2vec2-base",
        "wav2vec2-large": "facebook/wav2vec2-large",
        "wav2vec2-large-robust": "facebook/wav2vec2-large-robust",
        "hubert-large": "facebook/hubert-large-ll60k",
        "hubert-base": "facebook/hubert-base-ls960",
        "wavlm-base": "microsoft/wavlm-base",
        "wavlm-base-plus": "microsoft/wavlm-base-plus",
        "wavlm-large": "microsoft/wavlm-large",
        "data2vec-large": "facebook/data2vec-audio-large-960h",
        "data2vec-base": "facebook/data2vec-audio-base-960h"
    }
    return ssl_book.get(ssl_type, None)