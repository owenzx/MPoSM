#!/usr/bin/env bash


seed=$1
MODEL_NAME=run_en
python -u order_train.py \
        --seed $seed \
        --train \
        --train_file ./data/pos_wsj_full.txt.withrechunk \
        --test_file ./data/pos_wsj_full.txt \
        --batch_size  80 \
        --lr 0.001 \
        --optimizer reduce \
        --num_state 45 \
        --hidden_units 128 \
        --pos_embedding_dim 200 \
        --char_embedding_dim 100 \
        --model_name $MODEL_NAME \
        --vocab_name en \
        --vocab_path ./output/en.vocab \
        --overwrite_cache \
        --proj_name pos_emnlp \
        --encoder local \
        --decoder lstm \
        --order_loss marginal \
        --show_acc \
        --word_vec ./sample_data/wsj_word_vec.pkl \
        --max_seq_length 250 \
        --pred_from_vocab \
        --patience 200 \
        --valid_nepoch 5 \
        --chara_model rnn \
        --kl_reg_weight 0 \
        --entropy_reg_weight 0 \
        --mean_loss \
        --use_gumbel \
        --gumbel_temp 2.0