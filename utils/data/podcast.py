import os
import numpy as np
import pandas as pd
SPLIT_MAP = {
    "train": "Train",
    "dev": "Development",
    "test": "Test1"
}

# Load label
def load_utts(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    
    return cur_utts

def load_adv_emo_label(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[label_df["Split_Set"] == SPLIT_MAP[dtype]]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_labs = cur_df[["EmoAct", "EmoDom", "EmoVal"]].to_numpy()

    return cur_utts, cur_labs

def load_spk_id(label_path, dtype):
    label_df = pd.read_csv(label_path, sep=",")
    cur_df = label_df[(label_df["Split_Set"] == SPLIT_MAP[dtype])]
    cur_df = cur_df[(cur_df["SpkrID"] != "Unknown")]
    cur_utts = cur_df["FileName"].to_numpy()
    cur_spk_ids = cur_df["SpkrID"].to_numpy().astype(np.int)
    # Cleanining speaker id
    uniq_spk_id = list(set(cur_spk_ids))
    uniq_spk_id.sort()
    for new_id, old_id in enumerate(uniq_spk_id):
        cur_spk_ids[cur_spk_ids == old_id] = new_id
    total_spk_num = len(uniq_spk_id)

    return cur_utts, cur_spk_ids, total_spk_num

import json
from tqdm import tqdm
def load_char_label(trs_dir, utt_list, vocab_dict):    
    stop_chars = [".", ",", "?", "!", "\"", ":", ";", "-", "{", "}", "`", "–", "…", "_", "#", "=", "*", "\\", "/", "—", "~"]
    total_trs_list = []
    valid_utts = []
    for utt_id in tqdm(utt_list):
        utt_id = utt_id.split(".")[0]
        trs_path = trs_dir+"/"+utt_id+".txt"
        if not os.path.exists(trs_path):
            continue
        with open(trs_path, 'r') as f:
            text = f.readline()
        text = text.lstrip().rstrip()
        if "speaker 1:" in text:
            text = text.replace("speaker 1:", "")
        if "speaker 2:" in text:
            text = text.replace("speaker 2:", "")
        if "’" in text:
            text = text.replace("’", "'")

        # Replace from [ to ] into <unk>
        # Ignore from ( to )
        
        tokens = []
        result_tokens = []
        result_tokens.append(vocab_dict["<s>"])
        pre_verbal = False
        pre_square = False
        pre_round = False

        for c in text:
            if c == "[":
                pre_square = True
                continue
            elif c == "]" and pre_square:
                pre_square = False
                tokens.append(vocab_dict["<unk>"])
                continue
            elif c == "(":
                pre_round = True
                continue
            elif c == ")" and pre_round:
                pre_round = False
                continue
            elif pre_square or pre_round:
                continue
            elif c == " ":
                if pre_verbal:
                    tokens.append(vocab_dict["|"])
                pre_verbal=False
                continue
            elif c in stop_chars:
                pre_verbal=True
                continue
            else:
                c = c.upper()
                cur_token = vocab_dict.get(c, 3)
                tokens.append(cur_token)
                pre_verbal=True
        if len(tokens) > 0:
            valid_utts.append(utt_id+".wav")
            result_tokens.extend(tokens)
            result_tokens.append(vocab_dict["</s>"])
            total_trs_list.append(result_tokens)
    return total_trs_list, valid_utts
