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
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--noise_path", type=str, default=None)
parser.add_argument("--snr", type=str, default=None)

parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--ssl_type", type=str, default="wavlm-large")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--accumulation_steps", type=int, default=1)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--lr_decay", type=float, default=0.8)

parser.add_argument("--margin", type=float, default=0.5)
parser.add_argument("--tl_coef", type=float, default=100.0)
parser.add_argument("--cl_coef", type=float, default=10.0)

parser.add_argument("--original_model_path", type=str, default="./temp")
parser.add_argument("--model_path", type=str, default="./temp")
args = parser.parse_args()

utils.set_deterministic(args.seed)
SSL_TYPE = utils.get_ssl_type(args.ssl_type)
assert SSL_TYPE != None, print("Invalid SSL type!")
BATCH_SIZE = args.batch_size
ACCUMULATION_STEP = args.accumulation_steps
assert (ACCUMULATION_STEP > 0) and (BATCH_SIZE % ACCUMULATION_STEP == 0)
EPOCHS=args.epochs
LR=args.lr
LR_DECAY=args.lr_decay
MODEL_PATH = args.model_path
os.makedirs(MODEL_PATH+"/param", exist_ok=True)

from collections import defaultdict
corpus_path = args.data_path
audio_path = os.path.join(corpus_path, "Audios")
label_path = os.path.join(corpus_path, "Labels", "labels_concensus.csv")

total_dataset=dict()
total_dataloader=dict()
for dtype in ["train", "dev"]:
    cur_utts, cur_labs = utils.load_adv_emo_label(label_path, dtype)
    cur_wavs = utils.load_audio(audio_path, cur_utts)
    if dtype == "train":
        cur_wav_set = utils.WavSet(cur_wavs)
        cur_wav_set.save_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
    else:
        if dtype == "dev":
            wav_mean = total_dataset["train"].datasets[0].wav_mean
            wav_std = total_dataset["train"].datasets[0].wav_std
        elif dtype == "test":
            wav_mean, wav_std = utils.load_norm_stat(MODEL_PATH+"/train_norm_stat.pkl")
        cur_wav_set = utils.WavSet(cur_wavs, wav_mean=wav_mean, wav_std=wav_std)
    ########################################################
    cur_bs = BATCH_SIZE // ACCUMULATION_STEP if dtype == "train" else 1
    is_shuffle=True if dtype == "train" else False
    ########################################################
    cur_emo_set = utils.ADV_EmoSet(cur_labs)
    total_dataset[dtype] = utils.CombinedSet([cur_wav_set, cur_emo_set])
    total_dataloader[dtype] = DataLoader(
        total_dataset[dtype], batch_size=cur_bs, shuffle=is_shuffle, 
        pin_memory=True, num_workers=4,
        collate_fn=utils.collate_fn_wav_lab_mask
    )
# Load noise sound
noise_wav_path = glob.glob(args.noise_path+"/*.wav")
wav_path = noise_wav_path[args.seed]
train_noise_sample, _ = librosa.load(wav_path, sr=16000) # Assuming that the noise sound file is 30s
train_noise_sample = torch.Tensor([train_noise_sample]).float()

wav_path = noise_wav_path[args.seed+1]
dev_noise_sample, _ = librosa.load(wav_path, sr=16000) # Assuming that the noise sound file is 30s
dev_noise_sample = torch.Tensor([dev_noise_sample]).float()

augment_snr = args.snr

print("Loading pre-trained ", SSL_TYPE, " model...")

ssl_model = AutoModel.from_pretrained(SSL_TYPE)
del ssl_model.encoder.layers[12:]
ssl_model.load_state_dict(torch.load(args.original_model_path+"/final_ssl.pt"))
ssl_model.feature_extractor._freeze_parameters()
ssl_model.eval(); ssl_model.cuda()

T_ssl_model = AutoModel.from_pretrained(SSL_TYPE)
del T_ssl_model.encoder.layers[12:]
T_ssl_model.load_state_dict(torch.load(args.original_model_path+"/final_ssl.pt"))
T_ssl_model.eval(); T_ssl_model.cuda()

ser_model = net.EmotionRegression(1024, 1024, 1, 3, p=0.5)
ser_model.load_state_dict(torch.load(args.original_model_path+"/final_ser.pt"))
ser_model.eval(); ser_model.cuda()

ssl_opt = torch.optim.Adam(ssl_model.parameters(), LR)
ser_opt = torch.optim.Adam(ser_model.parameters(), LR)

scaler = GradScaler()
ssl_opt.zero_grad(set_to_none=True)
ser_opt.zero_grad(set_to_none=True)

ssl_sch = optim.lr_scheduler.ExponentialLR(ssl_opt, gamma=LR_DECAY)
ser_sch = optim.lr_scheduler.ExponentialLR(ser_opt, gamma=LR_DECAY)

lm = utils.LogManager()
lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val"])
lm.alloc_stat_type_list(["dev_aro", "dev_dom", "dev_val"])
lm.alloc_stat_type_list(["train_CL"])
lm.alloc_stat_type_list(["train_TL"])

min_epoch=0
min_loss=1e10

InfoNCEloss = utils.InfoNCE(negative_mode='unpaired')

for epoch in range(EPOCHS):
    print("Epoch: ", epoch, "LR: ", ssl_sch.get_lr())
    lm.init_stat()
    ssl_model.train()
    ser_model.train()    
    batch_cnt = 0

    for xy_pair in tqdm(total_dataloader["train"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()

        noise_x = utils.contaminate_all(x.cpu().numpy(), train_noise_sample.cpu().numpy(), augment_snr, is_torch=True)
        noise_x = noise_x.cuda(non_blocking=True).float()

        batch_len = x.shape[0]
        neg_idxs = utils.generate_negative_idxs(y, margin=args.margin)
        with autocast(enabled=True):
            with torch.no_grad():
                T_emb = T_ssl_model(x, attention_mask=mask).last_hidden_state
                T_h = torch.mean(T_emb, dim=1)
            S_emb = ssl_model(noise_x, attention_mask=mask).last_hidden_state
            S_h = torch.mean(S_emb, dim=1)
            emo_pred = ser_model(S_h)
            
            ccc = utils.CCC_loss(emo_pred, y)
            rmse = torch.sqrt(F.mse_loss(T_h.detach(), S_h))

            infoNCE = []
            for cur_idx in range(batch_len):
                neg_idx = neg_idxs[cur_idx]
                if len(neg_idx) == 0:
                    continue
                pos = T_h[cur_idx].unsqueeze(0)
                query = S_h[cur_idx].unsqueeze(0)
                negs = S_h[neg_idx]

                cur_loss = InfoNCEloss(query, pos, negs)
                infoNCE.append(cur_loss)
            if len(infoNCE) != 0:
                infoNCE = torch.stack(infoNCE).mean()
            else:
                infoNCE = 0.0

            total_loss = torch.sum(1.0-ccc) + args.tl_coef*rmse + args.cl_coef*infoNCE
            total_loss /= ACCUMULATION_STEP
        scaler.scale(total_loss).backward()
        if (batch_cnt+1) % ACCUMULATION_STEP == 0 or (batch_cnt+1) == len(total_dataloader["train"]):
            scaler.step(ssl_opt)
            scaler.step(ser_opt)
            scaler.update()
            ssl_opt.zero_grad(set_to_none=True)
            ser_opt.zero_grad(set_to_none=True)
        batch_cnt += 1

        # Logging
        lm.add_torch_stat("train_aro", ccc[0])
        lm.add_torch_stat("train_dom", ccc[1])
        lm.add_torch_stat("train_val", ccc[2])  
        lm.add_torch_stat("train_TL", rmse)  
        lm.add_torch_stat("train_CL", infoNCE)   

    ssl_sch.step()
    ser_sch.step()
    ssl_model.eval()
    ser_model.eval() 
    total_pred = [] 
    total_y = []
    for xy_pair in tqdm(total_dataloader["dev"]):
        x = xy_pair[0]; x=x.cuda(non_blocking=True).float()
        y = xy_pair[1]; y=y.cuda(non_blocking=True).float()
        mask = xy_pair[2]; mask=mask.cuda(non_blocking=True).float()

        noise_x = utils.contaminate_all(x.cpu().numpy(), dev_noise_sample.cpu().numpy(), augment_snr, is_torch=True)
        noise_x = noise_x.cuda(non_blocking=True).float()
        
        with torch.no_grad():
            ssl = ssl_model(noise_x, attention_mask=mask).last_hidden_state
            ssl = torch.mean(ssl, dim=1)
            emo_pred = ser_model(ssl)

            total_pred.append(emo_pred)
            total_y.append(y)

    # CCC calculation
    total_pred = torch.cat(total_pred, 0)
    total_y = torch.cat(total_y, 0)
    ccc = utils.CCC_loss(total_pred, total_y)
    # Logging
    lm.add_torch_stat("dev_aro", ccc[0])
    lm.add_torch_stat("dev_dom", ccc[1])
    lm.add_torch_stat("dev_val", ccc[2])

    # Save model
    lm.print_stat()
    torch.save(ser_model.state_dict(), \
        os.path.join(MODEL_PATH, "param", str(epoch)+"_ser.pt"))
    torch.save(ssl_model.state_dict(), \
        os.path.join(MODEL_PATH, "param", str(epoch)+"_ssl.pt"))

        
    dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
    if min_loss > dev_loss:
        min_epoch = epoch
        min_loss = dev_loss

        print("Save",min_epoch)
        print("Loss",3.0-min_loss)
        for mtype in ["ser", "ssl"]:
            os.system("cp "+os.path.join(MODEL_PATH, "param", "{}_{}.pt".format(min_epoch, mtype)) + \
            " "+os.path.join(MODEL_PATH, "final_{}.pt".format(mtype)))
