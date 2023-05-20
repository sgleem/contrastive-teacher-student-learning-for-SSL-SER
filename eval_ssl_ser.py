# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
import pandas as pd
from tqdm import tqdm
import glob
import librosa
import copy

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import AutoModel

# Self-Written Modules
sys.path.append(os.getcwd())
import net
import utils


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--data_path", type=str, default="/media/kyunster/ssd1/corpus/MSP_Podcast_1.10")
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--model_path", type=str, default="./model/wavlm-large")
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
MODEL_PATH = args.model_path

from collections import defaultdict
corpus_path = args.data_path
audio_path = os.path.join(corpus_path, "Audios")
label_path = os.path.join(corpus_path, "Labels", "labels_concensus.csv")

total_dataset=dict()
total_dataloader=dict()
for dtype in ["test"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=1, shuffle=False, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
del ssl_model.encoder.layers[12:]
ssl_model.freeze_feature_encoder()
ssl_model.load_state_dict(torch.load(MODEL_PATH+"/final_ssl.pt"))
ssl_model.eval(); ssl_model.cuda()

ser_model = net.EmotionRegression(1024, 1024, 1, 3, p=0.5)
ser_model.load_state_dict(torch.load(MODEL_PATH+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()


lm = utils.LogManager()
lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])

min_epoch=0
min_loss=1e10

lm.init_stat()

ssl_model.eval()
ser_model.eval() 
total_pred = [] 
total_y = []
for xy_pair in tqdm(total_dataloader["test"]):
    x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
    y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
    mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()
    
    with torch.no_grad():
        ssl = ssl_model(x, attention_mask=mask).last_hidden_state
        ssl = torch.mean(ssl, dim=1)
        emo_pred = ser_model(ssl)

        total_pred.append(emo_pred)
        total_y.append(y)

# CCC calculation
total_pred = torch.cat(total_pred, 0)
total_y = torch.cat(total_y, 0)
ccc = utils.CCC_loss(total_pred, total_y)
# Logging
lm.add_torch_stat("test_aro", ccc[0])
lm.add_torch_stat("test_dom", ccc[1])
lm.add_torch_stat("test_val", ccc[2])

lm.print_stat()
