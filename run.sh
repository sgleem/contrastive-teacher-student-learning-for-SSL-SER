#!/bin/bash

seed=0
# Fine-tune SER model with clean speech set
ssl_type=wav2vec2-large-robust # Change ssl model type here
clean_data_path=/media/kyunster/ssd1/corpus/MSP_Podcast_1.10
clean_model_path=model/w2v-large-robust-clean
python train_ssl_ser.py \
    --data_path=${clean_data_path} \
    --seed=${seed} \
    --ssl_type=${ssl_type} \
    --batch_size=32 \
    --accumulation_steps=4 \
    --lr=1e-4 \
    --epochs=10 \
    --model_path=${clean_model_path} || exit 0;

# Apply contrastive teacher-student learning SER model with noisy speech set
noise_path=/media/kyunster/ssd1/corpus/noise-sample
noisy_model_path=model/w2v-large-robust-0db
snr=0
margin=0.5
tl_coef=100.0
cl_coef=10.0
python train_ssl_ser_tl_cl.py \
    --data_path=${clean_data_path} \
    --noise_path=${noise_path} \
    --snr=${snr} \
    --seed=${seed} \
    --ssl_type=${ssl_type} \
    --batch_size=32 \
    --accumulation_steps=4 \
    --lr=1e-4 \
    --epochs=10 \
    --margin=${margin} \
    --tl_coef=${tl_coef} \
    --cl_coef=${cl_coef} \
    --original_model_path=${clean_model_path} \
    --model_path=${noisy_model_path} || exit 0;

# Evaluate SER model
python eval_ssl_ser.py \
    --ssl_type=${ssl_type} \
    --model_path=${noisy_model_path} || exit 0;
