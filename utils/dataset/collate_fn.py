import torch
import torch.nn as nn
import numpy as np

def collate_fn_wav_lab_mask(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    for wav_data in batch:
        wav, dur = wav_data[0]   
        lab = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(lab)
        total_dur.append(dur)

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    
    total_lab = torch.Tensor(np.array(total_lab))
    max_dur = np.max(total_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    ## compute mask
    return total_wav, total_lab, attention_mask

def collate_fn_wav_txt_masks(batch):
    total_wav = []
    total_lab = []
    total_dur = []
    total_text_dur = []
    for wav_data in batch:
        wav, dur = wav_data[0]   
        text, text_dur = wav_data[1]
        total_wav.append(torch.Tensor(wav))
        total_lab.append(torch.Tensor(text))
        total_dur.append(dur)
        total_text_dur.append(text_dur)

    total_wav = nn.utils.rnn.pad_sequence(total_wav, batch_first=True)
    total_text = nn.utils.rnn.pad_sequence(total_lab, batch_first=True)
    
    max_dur = np.max(total_dur)
    max_text_dur = np.max(total_text_dur)
    attention_mask = torch.zeros(total_wav.shape[0], max_dur)
    text_attention_mask = torch.zeros(total_wav.shape[0], max_text_dur)
    for data_idx, dur in enumerate(total_dur):
        attention_mask[data_idx,:dur] = 1
    text_mask = torch.Tensor(np.array(total_text_dur)).long()
    ## compute mask
    return total_wav, total_text, attention_mask, text_mask
