#!/usr/bin/env bash


lan=$1
seed=$2

MODEL_NAME=${lan}_pretrain
python -u order_train.py \
        --seed $seed \
        --train \
        --train_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll.withrechunk \
        --test_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll\
        --batch_size 80 \
        --lr 0.001 \
        --optimizer reduce \
        --num_state 12 \
        --hidden_units 128 \
        --pos_embedding_dim 200 \
        --enc_layer_num 0 \
        --char_embedding_dim 100 \
        --proj_name pos_emnlp \
        --model_name $MODEL_NAME \
        --vocab_name ut_${lan} \
        --vocab_path ./output/ut_${lan}.vocab \
        --overwrite_cache \
        --encoder local \
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
        --max_epoch 200 \
        --bind_xz \
        --use_gumbel \
        --gumbel_temp 2.0

MODEL_NAME=${lan}_pretrain_continue
python -u order_train.py \
        --seed $seed \
        --train \
        --train_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll.withrechunk \
        --test_file ./data/universal_treebanks_v2.1/std/${lan}/${lan}-universal-full.conll\
        --batch_size 80 \
        --lr 0.001 \
        --optimizer reduce \
        --num_state 12 \
        --hidden_units 128 \
        --pos_embedding_dim 200 \
        --enc_layer_num 0 \
        --char_embedding_dim 100 \
        --proj_name pos_emnlp \
        --model_name $MODEL_NAME \
        --vocab_name ut_${lan} \
        --vocab_path ./output/ut_${lan}.vocab \
        --encoder local \
        --decoder lstm \
        --order_loss marginal \
        --show_acc \
        --max_seq_length 250 \
        --pred_from_vocab \
        --patience 200 \
        --valid_nepoch 5 \
        --ud_format \
        --chara_model rnn \
        --kl_reg_weight 0 \
        --entropy_reg_weight 0 \
        --load_model_path ./dump_models/order/${MODEL_NAME::-9}/pos_${seed}.pt \
        --bind_xz \
        --use_gumbel \
        --gumbel_temp 2.0

