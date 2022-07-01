#!/usr/bin/env bash

lan=$1

seed=$2

MODEL_NAME=bert_${lan}
python -u order_train.py \
        --seed $seed \
        --train \
        --train_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll.withrechunk \
        --test_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll\
        --batch_size 80 \
        --accum_count 1 \
        --lr 0.001 \
        --optimizer reduce \
        --num_state 12 \
        --hidden_units 128 \
        --pos_embedding_dim 200 \
        --char_embedding_dim 100 \
        --model_name_or_path bert-base-multilingual-cased \
        --model_name $MODEL_NAME \
        --vocab_name bert_${lan} \
        --vocab_path ./output/bert_${lan}.vocab \
        --overwrite_cache \
        --proj_name pos_emnlp \
        --encoder bert \
        --use_bert \
        --bert_layer avg \
        --decoder lstm \
        --order_loss word \
        --show_acc \
        --max_seq_length 250 \
        --pred_from_vocab \
        --patience 40 \
        --valid_nepoch 5 \
        --ud_format \
        --chara_model rnn \
        --kl_reg_weight 0 \
        --entropy_reg_weight 0 \
        --use_gumbel \
        --mean_loss \
        --gumbel_temp 2.0



MODEL_NAME=bert_${lan}_continue
python -u order_train.py \
        --seed $seed \
        --train \
        --train_file /playpen-ssd/home/xzh/datasets/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll.withrechunk \
        --test_file /playpen-ssd/home/xzh/datasets/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll\
        --batch_size 80 \
        --accum_count 1 \
        --lr 0.001 \
        --optimizer reduce \
        --num_state 12 \
        --hidden_units 128 \
        --pos_embedding_dim 200 \
        --char_embedding_dim 100 \
        --model_name_or_path bert-base-multilingual-cased \
        --model_name $MODEL_NAME \
        --vocab_name bert_${lan} \
        --vocab_path ./output/bert_${lan}.vocab \
        --proj_name pos_emnlp \
        --encoder bert \
        --use_bert \
        --bert_layer avg \
        --decoder lstm \
        --order_loss marginal \
        --show_acc \
        --max_seq_length 250 \
        --pred_from_vocab \
        --patience 40 \
        --valid_nepoch 5 \
        --ud_format \
        --chara_model rnn \
        --kl_reg_weight 0 \
        --entropy_reg_weight 0 \
        --load_model_path ./dump_models/order/${MODEL_NAME::-9}/pos_avg_${seed}.pt \
        --use_gumbel \
        --mean_loss \
        --gumbel_temp 2.0

