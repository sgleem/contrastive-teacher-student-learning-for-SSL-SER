# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import glob
from tqdm import tqdm
import warnings
import librosa
from multiprocessing import Pool
warnings.filterwarnings('ignore')
# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import AutoModel

def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav
def get_norm_stat_for_wav(wav_list, verbose=False):
    count = 0
    wav_sum = 0
    wav_sqsum = 0
    
    for cur_wav in tqdm(wav_list):
        wav_sum += np.sum(cur_wav)
        wav_sqsum += np.sum(cur_wav**2)
        count += len(cur_wav)
    
    wav_mean = wav_sum / count
    wav_var = (wav_sqsum / count) - (wav_mean**2)
    wav_std = np.sqrt(wav_var)

    return wav_mean, wav_std


seed = 0
test_wav_dir = "data/podcast" # test wav dir
model_dir = "model/wav2vec2-large-robust-12" # dir of wav2vec2.0 model
result_path = "result/w2v_feat.pkl"

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(seed)
np.random.seed(seed)
    
test_wav_path = glob.glob(os.path.join(test_wav_dir, '*.wav'))
with Pool(24) as p:
    wav_list = list(tqdm(p.imap(extract_wav, test_wav_path), total=len(test_wav_path)))
with open(os.path.join(model_dir, 'train_norm_stat.pkl'), 'rb') as f:
    wav_mean, wav_std = pk.load(f)
# wav_mean, wav_std = get_norm_stat_for_wav(wav_list)

# Load model
wav2vec_model= AutoModel.from_pretrained("microsoft/wav2vec2-large-robust")
wav2vec_model.load_state_dict(torch.load(model_dir+"/final_ssl.pt"))
del wav2vec_model.encoder.layers[12:]
wav2vec_model.cuda()
wav2vec_model.eval()

feat_dict = dict()
for utt_idx, cur_wav in enumerate(wav_list):
    wav_id = test_wav_path[utt_idx]
    utt_id = wav_id.split("/")[-1].split(".")[0]
    x = torch.from_numpy((cur_wav - wav_mean) / wav_std).unsqueeze(0)
    x = x.cuda(non_blocking=True).float()
    w2v = wav2vec_model(x).last_hidden_state
    w2v = w2v.squeeze(0).cpu().detach().numpy()
    feat_dict[utt_id] = w2v

with open(result_path, 'wb') as f:
    pk.dump(feat_dict, f)
