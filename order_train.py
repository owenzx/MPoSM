from __future__ import print_function

import pickle
import argparse
import sys
import time
import os
import json
import copy

import torch
import numpy as np
import random
import warnings

from modules import read_conll, \
    intra_word_tokenize, \
    build_vocab, \
    build_vocab_char, \
    ConllSent, \
    expand_word_vocab, \
    set_vocab_freq, \
    partial_load, \
    update_oracle_dict

from modules import OrderPOS
from sklearn.metrics.cluster import v_measure_score
from collections import Counter
from tqdm import trange, tqdm
from modules import write_conll
import wandb
from pprint import pprint

from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def set_seed(seed, args):
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed % 4294967296)
    random.seed(seed)


def init_config():
    parser = argparse.ArgumentParser(description='POS tagging')

    parser.add_argument('--test', action='store_true', default=False,
                        help='set if want to test, also add show_acc parameter if want to see full results')
    parser.add_argument('--train', action='store_true', default=False, help='set if training')

    # bert arguments
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    parser.add_argument(
        "--max_seq_length",
        default=250,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_word_length",
        default=60,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )

    # train and test data
    parser.add_argument('--word_vec', type=str,
                        help='the word vector file (cPickle saved file)')
    parser.add_argument('--train_file', type=str, help='train data')
    parser.add_argument('--test_file', default='', type=str, help='test data')
    parser.add_argument('--vocab_file', default=None, type=str, help='vocab data')
    parser.add_argument('--vocab_path', default=None, type=str, help='vocab path')
    parser.add_argument('--vocab_name', type=str, default=None, help='vocab name')
    parser.add_argument('--ud_format', action='store_true', default=False,
                        help='handles the difference in format for ud files')

    # loss options
    parser.add_argument('--order_loss', type=str, default='marginal',
                        help='the type of order loss, select from [marginal, word, none]')
    parser.add_argument('--pred_from_vocab', action='store_true', default=False,
                        help='if set, the word is predict from whole vocab instead of only the current sentence')
    parser.add_argument('--mean_loss', action='store_true', default=False,
                        help='if set, optimize mean instead of sum of loss')
    parser.add_argument('--entropy_reg_weight', type=float, default=0.0,
                        help='the weight for entropy regularization term, default is 0 (not used)')

    # optimization parameters
    parser.add_argument('--mask_rate', default=0.15, type=float, help='mask rate')
    parser.add_argument('--batch_size', default=512, type=int, help='batch_size')
    parser.add_argument('--epochs', default=9999, type=int, help='number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', help='choose optimizer')
    parser.add_argument('--dropout', default=0.5, type=float, help='dropout rate')
    parser.add_argument('--gumbel_temp', default=1.0, type=float, help='temperature of gumbel softmax')

    # model config
    parser.add_argument('--chara_model', type=str, default='None', help='select from [None, RNN, CNN])')

    parser.add_argument('--use_bert', action='store_true', default=False,
                        help='set if use bert (including tokenizers, etc.)')
    parser.add_argument('--encoder', default=None, type=str, help='encoder architecture')
    parser.add_argument('--decoder', default=None, type=str, help='encoder architecture')
    parser.add_argument('--self_attention', action='store_true', default=False,
                        help='if set, add self-attention after lstm')
    parser.add_argument('--bert_ff_layer_num', default=4, type=int, help='bert ff layer num')
    parser.add_argument('--bert_ff_hidden_dim', default=50, type=int, help='bert ff hidden dim')
    parser.add_argument('--bert_layer', default='12', type=str,
                        help='use the hidden states from which layer')
    parser.add_argument('--n_heads', default=4, type=int,
                        help='decoder heads')
    parser.add_argument('--num_inds', default=32, type=int,
                        help='num_inds')
    parser.add_argument('--no_position', action='store_true', default=False,
                        help='if use no position embedding from BERT')
    parser.add_argument('--num_state', default=45, type=int,
                        help='number of hidden states of z')
    parser.add_argument('--hidden_units', default=128, type=int, help='hidden units in ReLU Net')
    parser.add_argument('--encoder_hidden_dim', default=100, type=int, help='hidden units in encoder')
    parser.add_argument('--enc_layer_num', default=1, type=int, help='layer num in encoder')
    parser.add_argument('--dec_layer_num', default=1, type=int, help='layer num in decoder')
    parser.add_argument('--word_embedding_dim', default=100, type=int, help='dimension of the word embedding')
    parser.add_argument('--char_embedding_dim', default=8, type=int, help='dimension of the char embedding')
    parser.add_argument('--pos_embedding_dim', default=200, type=int, help='dimension of the pos embedding')
    parser.add_argument('--freeze_embeddings', action='store_true', default=False,
                        help='if set, freeze all the embeddings')
    parser.add_argument('--nonmask_loss', action='store_true', default=False,
                        help='if set, also calculate loss at the non-mask positions')
    parser.add_argument('--kl_reg_weight', type=float, default=0.5, help='the weight for kl regularization')
    parser.add_argument('--use_gumbel', action='store_true', default=False,
                        help='if set, use gumbel softmax at the bottom')

    parser.add_argument('--bind_xz', action='store_true', default=False,
                        help='if set, bind p(x|z) and p(z|x) using bayes rule')

    # pretrained model options
    parser.add_argument('--proj_name', default='uns-pos', type=str,
                        help='project name for wandb')
    parser.add_argument('--model_name', default='', type=str,
                        help='model path to save/load, feed directory for train and specific ckpt for test')
    parser.add_argument('--load_model_path', default=None, type=str,
                        help='model path to load pretrained models')

    # log parameters
    parser.add_argument('--valid_nepoch', default=5, type=int, help='valid_nepoch')
    parser.add_argument('--valid_nsteps', default=None, type=int, help='valid_nsteps')
    parser.add_argument('--show_acc', action='store_true', default=False,
                        help='not showing acc by default for purely unsupervised learning')
    parser.add_argument('--select_on_train', action='store_true', default=False, help='select with train loss')
    parser.add_argument('--patience', default=100, type=int,
                        help='how many epochs to wait before after the loss stop decreasing')
    parser.add_argument('--max_epoch', default=9999999, type=int, help='stop immediately when reach this epoch num')

    # Others
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--val_seed', default=2039, type=int,
                        help='always set to this seed before validation to reduce val metric variance')

    parser.add_argument('--accum_count', default=1, type=int, help='accum count for gradient update')
    parser.add_argument('--warmup_steps', default=0, type=float, help='number of warmup steps')
    parser.add_argument('--total_steps', default=-1, type=int, help='number of total training steps')
    parser.add_argument('--total_epochs', default=-1, type=int, help='number of total training epochs')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    assert (args.order_loss != 'none')

    args.save_dir = save_dir = os.path.join("dump_models/order", args.model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    args.tag_dir = tag_dir = os.path.join("output/pred_tags", args.model_name)
    if not os.path.exists(tag_dir):
        os.makedirs(tag_dir)

    if args.chara_model != 'None':
        args.use_chara = True
    else:
        args.use_chara = False

    if args.vocab_file == None:
        args.vocab_file = args.train_file

    seed = args.seed
    if seed is None:
        args.seed = seed = random.randrange(sys.maxsize)

    set_seed(seed, args)

    if args.use_bert:
        id_ = "pos_%s_%d" % (args.bert_layer, seed)
    else:
        id_ = "pos_%d" % (seed)
    model_ckpt_path = os.path.join(save_dir, id_ + '.pt')
    args.save_path = args.load_path = model_ckpt_path
    args.result_save_path = os.path.join(save_dir, id_ + '.result.json')
    args.full_result_save_path = os.path.join(save_dir, id_ + '.full_result.json')
    print("model save/load path: ", model_ckpt_path)

    args.tag_path = tag_path = os.path.join(tag_dir, id_)

    args_dict = vars(args)
    args_json_save_path = os.path.join(save_dir, id_ + '.args.json')
    with open(args_json_save_path, 'w') as fw:
        json.dump(args_dict, fw)

    wandb.init(name=args.model_name + '___' + id_, project=args.proj_name, config=args_dict)
    return args


class CoNLLFeatures(object):
    """
    A single set of features of data.
    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        label: Label corresponding to the input
    """

    def __init__(self, input_ids, attention_mask, token_type_ids=None, offsets=None, input_tags=None, charas=None,
                 chara_lengths=None, word_level_ids=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.offsets = offsets
        self.input_tags = input_tags
        self.charas = charas
        self.chara_lengths = chara_lengths
        self.word_level_ids = word_level_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features_nonbert(
        examples,
        vocab,
        max_length=512,
        max_char_length=60,
        lowercase=False,
        tag_vocab=None,
        char_vocab=None,
        use_chara=False,
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: HANS
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        if type(example) is ConllSent:
            text = example["word"]
        else:
            text = example
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        UNK_IDX = len(vocab.keys()) + 1  # include one pad token
        input_ids = [vocab[w] if w in vocab else UNK_IDX for w in text]
        if use_chara:
            CHAR_UNK_IDX = len(char_vocab.keys()) + 1
            chara_lengths = [min(len(w), max_char_length) for w in text]
            charas = []
            for w in text:
                charas.append([])
                for c in w:
                    charas[-1].append(char_vocab[c] if c in char_vocab else CHAR_UNK_IDX)
                ch_padding_length = max_char_length - len(charas[-1])
                if ch_padding_length > 0:
                    charas[-1] = charas[-1] + ([0] * ch_padding_length)
                else:
                    charas[-1] = charas[-1][:max_char_length]
        else:
            charas = None
            chara_lengths = None

        if tag_vocab is not None:
            tag = example['tag']
            input_tags = [tag_vocab[t] for t in tag]
        else:
            input_tags = None

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)
            if tag_vocab is not None:
                input_tags = input_tags + ([0] * padding_length)
            if use_chara:
                chara_lengths = chara_lengths + ([0] * padding_length)
                charas = charas + [[0] * max_char_length] * padding_length
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            if tag_vocab is not None:
                input_tags = input_tags[:max_length]
            if use_chara:
                chara_lengths = chara_lengths[:max_length]
                charas = charas[:max_length]

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )

        if ex_index < 10:
            print("*** Example ***")
            print("text_a: %s" % (text))
            print("ex_index: %s" % (ex_index))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))

        features.append(
            CoNLLFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=None,
                offsets=None,
                input_tags=input_tags,
                charas=charas,
                chara_lengths=chara_lengths
            )
        )
    return features


def update_cnt_stats(cnt_stats, tags, index, eval_tags, pred_lens, batch_idx, model_vm, gold_vm, args):
    tgs = tags[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
    eval_tags += tgs
    for (seq_gold_tags, seq_model_tags, pred_len) in zip(tgs, index, pred_lens):
        seq_model_tags = seq_model_tags[:pred_len]
        if len(seq_gold_tags) != len(seq_model_tags):
            print(seq_model_tags)
            print(len(seq_model_tags))
            print(seq_gold_tags)
            print(len(seq_gold_tags))
            warnings.warn('LENGTH MISMATCH!!!')
            seq_gold_tags = seq_gold_tags[:len(seq_model_tags)]
            assert (len(seq_gold_tags) == len(seq_model_tags))
        # print(seq_gold_tags)
        # print(seq_model_tags)
        # exit()
        for (gold_tag, model_tag) in zip(seq_gold_tags, seq_model_tags):
            model_tag = model_tag.item()
            gold_vm += [gold_tag]
            model_vm += [model_tag]
            if model_tag not in cnt_stats:
                cnt_stats[model_tag] = Counter()
            cnt_stats[model_tag][gold_tag] += 1


def get_cluster_metrics(cnt_stats, eval_tags, index_all, gold_vm, model_vm, all_nwords, print_path, sentences,
                        null_index):
    match_dict = {}
    correct = 0.0
    # match
    for tag in cnt_stats:
        # print(tag, cnt_stats[tag])
        match_dict[tag] = cnt_stats[tag].most_common(1)[0][0]
    # exit()

    # eval many2one
    for (seq_gold_tags, seq_model_tags) in zip(eval_tags, index_all):
        for (gold_tag, model_tag) in zip(seq_gold_tags, seq_model_tags):
            model_tag = model_tag.item()
            if match_dict[model_tag] == gold_tag:
                correct += 1
    accuracy = correct / all_nwords
    vm = v_measure_score(gold_vm, model_vm)
    if print_path is not None:
        print(print_path)
        write_conll(print_path, sentences, index_all, null_index)
    return accuracy, vm


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        lowercase=False,
        use_chara=False,
        word_vocab=None
):
    """
    Loads a data file into a list of ``InputFeatures``
    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: HANS
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)
    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        text = example["word"]
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        UNK_IDX = len(word_vocab.keys()) + 1  # include one pad token
        word_level_ids = [word_vocab[w] if w in word_vocab else UNK_IDX for w in text]
        padding_length = max_length - len(word_level_ids)
        if padding_length > 0:
            word_level_ids = word_level_ids + ([0] * padding_length)
        else:
            word_level_ids = word_level_ids[:max_length]

        wordpieces, offsets = intra_word_tokenize(tokenizer, text, add_special_tokens=True)
        token_offsets = [x if x is not None else (max_length - 2, max_length - 1) for x in offsets]
        input_ids = [t[1] for t in wordpieces]
        token_type_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        offset_padding_length = max_length - len(token_offsets)
        if padding_length > 0:
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        else:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]

        if offset_padding_length > 0:
            if pad_on_left:
                token_offsets = ([(-1, -1)] * offset_padding_length) + token_offsets
            else:
                token_offsets = token_offsets + ([(-1, -1)] * offset_padding_length)
        else:
            token_offsets = token_offsets[:max_length]

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )
        assert len(token_offsets) == max_length, "Error with input length {} vs {}".format(
            len(token_offsets), max_length
        )

        if ex_index < 10:
            print("*** Example ***")
            print("text_a: %s" % (text))
            print("ex_index: %s" % (ex_index))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            print("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))

        features.append(
            CoNLLFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                offsets=token_offsets,
                word_level_ids=word_level_ids
            )
        )
    return features


def get_all_vocab_char_input(args, w2clist):
    real_max_len = max([len(l) for l in w2clist.values()])
    vocab_len = len(w2clist.keys())
    max_len = min(args.max_word_length, real_max_len)
    all_vocab_char_list = []
    all_vocab_char_length_list = []
    # 0 is PAD, vocab_size + 1 is UNK
    for i in range(vocab_len + 2):
        if i not in w2clist:
            assert (i == 0 or i == (vocab_len + 1))
            all_vocab_char_list.append([0] * max_len)
            all_vocab_char_length_list.append(1)
            continue

        pad_length = max_len - len(w2clist[i])
        if pad_length < 0:
            w2clist[i] = w2clist[i][:max_len]
            pad_length = 0
        all_vocab_char_list.append(w2clist[i] + [0] * pad_length)
        all_vocab_char_length_list.append(len(w2clist[i]))

    return torch.tensor(all_vocab_char_list, dtype=torch.long), torch.tensor(all_vocab_char_length_list,
                                                                             dtype=torch.long)


def load_and_cache_examples_nonbert(args, examples, vocab, cache_prefix=None, lowercase=False, tag_vocab=None,
                                    char_vocab=None, use_chara=False):
    cached_features_file = "{}_cached_{}_{}".format(
        cache_prefix,
        args.vocab_name,
        str(args.max_seq_length),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file for %s", cache_prefix)
        features = convert_examples_to_features_nonbert(
            examples,
            vocab,
            max_length=args.max_seq_length,
            max_char_length=args.max_word_length,
            lowercase=lowercase,
            tag_vocab=tag_vocab,
            char_vocab=char_vocab,
            use_chara=use_chara
        )
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    if tag_vocab is not None:
        all_input_tags = torch.tensor([f.input_tags for f in features], dtype=torch.long)
        if use_chara:
            all_charas = torch.tensor([f.charas for f in features], dtype=torch.long)
            all_chara_lengths = torch.tensor([f.chara_lengths for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_input_tags, all_charas, all_chara_lengths)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_input_tags)
    else:
        if use_chara:
            all_charas = torch.tensor([f.charas for f in features], dtype=torch.long)
            all_chara_lengths = torch.tensor([f.chara_lengths for f in features], dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_attention_mask, all_charas, all_chara_lengths)
        else:
            dataset = TensorDataset(all_input_ids, all_attention_mask)
    return dataset


def load_and_cache_examples(args, examples, tokenizer, cache_prefix=None, lowercase=False, use_chara=False,
                            word_vocab=None):
    cached_features_file = "{}_cached_{}_{}".format(
        cache_prefix,
        list(filter(None, args.model_name_or_path.split("/"))).pop(),
        str(args.max_seq_length),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file for %s", cache_prefix)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id,
            lowercase=lowercase,
            use_chara=use_chara,
            word_vocab=word_vocab,
        )

        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # print(features[0])

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_offsets = torch.tensor([f.offsets for f in features], dtype=torch.long)
    # if use_chara:
    all_word_level_ids = torch.tensor([f.word_level_ids for f in features], dtype=torch.long)

    print(all_input_ids.shape)
    print(all_attention_mask.shape)
    print(all_token_type_ids.shape)
    print(all_offsets.shape)
    # if use_chara:
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_offsets, all_word_level_ids)
    # else:
    #     dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_offsets)
    return dataset


def eval_model_for_loss(model, data, tags, args, sentences=None, print_path=None, null_index=None):
    """This function is used for calculating the loss with masks"""
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.batch_size)
    model.eval()
    set_seed(args.val_seed, args)
    with torch.no_grad():
        all_loss, all_nwords, all_nbatches = 0, 0, 0
        all_pure_loss = 0
        all_reg_loss = 0
        index_all, eval_tags = [], []
        top_index_all = []
        bottom_index_all = []
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(args.device) for t in batch)
            if args.use_bert:
                inputs = {"input_ids": batch[0], "masks": batch[1], "offsets": batch[3], "greedy_gumbel": False}
                # if args.use_chara:
                inputs['word_level_ids'] = batch[-1]
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            else:
                inputs = {"input_ids": batch[0], "masks": batch[1], "greedy_gumbel": False}

            if args.use_chara:
                inputs['chars'] = batch[-2]
                inputs['char_lengths'] = batch[-1].cpu()

            sents = sentences[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            nwords = sum(len(sent) for sent in sents)
            all_nwords += nwords
            all_nbatches += 1
            output_dict = model(**inputs)
            index = output_dict['pred_tags']
            index_all += list(index)
            top_index = output_dict['top_pred_tags']
            top_index_all += list(top_index)
            bottom_index = output_dict['bottom_pred_tags']
            bottom_index_all += list(bottom_index)
            loss = output_dict['loss']
            pure_loss = output_dict['pure_loss']
            reg_loss = output_dict['reg_loss']
            # num_words = output_dict['word_count']
            all_loss += loss
            all_pure_loss += pure_loss
            all_reg_loss += reg_loss

        model.train()

        if args.mean_loss:
            eval_result_dict = {
                'loss': (all_loss / all_nbatches).item() if (
                                                                        all_loss / all_nbatches) is not float else all_loss / all_nbatches,
                'pure_loss': (all_pure_loss / all_nbatches).item() if (
                                                                                  all_pure_loss / all_nbatches) is not float else all_pure_loss / all_nbatches,
                'reg_loss': (all_reg_loss / all_nbatches).item() if (
                                                                                all_reg_loss / all_nbatches) is not float else all_reg_loss / all_nbatches,
            }
        else:
            eval_result_dict = {
                'loss': (all_loss / all_nwords).item() if type(
                    all_loss / all_nwords) is not float else all_loss / all_nwords,
                'opt_loss': (all_loss / all_nbatches).item() if (
                                                                            all_loss / all_nbatches) is not float else all_loss / all_nbatches,
                'pure_loss': (all_pure_loss / all_nwords).item() if (
                                                                                all_pure_loss / all_nwords) is not float else all_pure_loss / all_nwords,
                'reg_loss': (all_reg_loss / all_nbatches).item() if (
                                                                                all_reg_loss / all_nbatches) is not float else all_reg_loss / all_nbatches,
            }

    return eval_result_dict


def eval_model(model, data, tags, args, sentences=None, print_path=None, null_index=None):
    eval_sampler = SequentialSampler(data)
    eval_dataloader = DataLoader(data, sampler=eval_sampler, batch_size=args.batch_size)
    model.eval()
    set_seed(args.val_seed, args)
    with torch.no_grad():
        all_loss, all_nwords, all_nbatches = 0, 0, 0
        all_pure_loss = 0
        all_reg_loss = 0
        correct = 0.0
        index_all, eval_tags = [], []
        top_index_all = []
        bottom_index_all = []
        top_eval_tags, bottom_eval_tags = [], []
        gold_vm, model_vm = [], []
        top_gold_vm, top_model_vm = [], []
        bottom_gold_vm, bottom_model_vm = [], []
        cnt_stats = {}
        top_cnt_stats = {}
        bottom_cnt_stats = {}
        match_dict = {}
        for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
            batch = tuple(t.to(args.device) for t in batch)
            if args.use_bert:
                inputs = {"input_ids": batch[0], "masks": batch[1], "offsets": batch[3], "greedy_gumbel": True}
                # if args.use_chara:
                inputs['word_level_ids'] = batch[-1]
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
            else:
                inputs = {"input_ids": batch[0], "masks": batch[1], "greedy_gumbel": True}

            if args.use_chara:
                inputs['chars'] = batch[-2]
                inputs['char_lengths'] = batch[-1].cpu()

            sents = sentences[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            nwords = sum(len(sent) for sent in sents)
            all_nwords += nwords
            all_nbatches += 1

            output_dict = model(**inputs)
            index = output_dict['pred_tags']
            top_index = output_dict['top_pred_tags']
            bottom_index = output_dict['bottom_pred_tags']
            pred_lens = output_dict['pred_len']
            index_all += list(index)
            top_index_all += list(top_index)
            bottom_index_all += list(bottom_index)
            # num_words = output_dict['word_count']
            if tags is not None and args.show_acc:
                update_cnt_stats(cnt_stats, tags, index, eval_tags, pred_lens, batch_idx, model_vm, gold_vm, args)
                update_cnt_stats(top_cnt_stats, tags, top_index, top_eval_tags, pred_lens, batch_idx, top_model_vm,
                                 top_gold_vm, args)
                update_cnt_stats(bottom_cnt_stats, tags, bottom_index, bottom_eval_tags, pred_lens, batch_idx,
                                 bottom_model_vm, bottom_gold_vm, args)

        if tags is not None and args.show_acc:
            accuracy, vm = get_cluster_metrics(cnt_stats, eval_tags, index_all, gold_vm, model_vm, all_nwords,
                                               print_path, sentences, null_index)
            top_accuracy, top_vm = get_cluster_metrics(top_cnt_stats, eval_tags, top_index_all, top_gold_vm,
                                                       top_model_vm, all_nwords, print_path, sentences, null_index)
            bottom_accuracy, bottom_vm = get_cluster_metrics(bottom_cnt_stats, eval_tags, bottom_index_all,
                                                             bottom_gold_vm, bottom_model_vm, all_nwords, print_path,
                                                             sentences, null_index)

        model.train()

        eval_result_dict = {}

        if tags is not None and args.show_acc:
            eval_result_dict['accuracy'] = accuracy
            eval_result_dict['vm'] = vm
            eval_result_dict['top_accuracy'] = top_accuracy
            eval_result_dict['top_vm'] = top_vm
            eval_result_dict['bottom_accuracy'] = bottom_accuracy
            eval_result_dict['bottom_vm'] = bottom_vm

    return eval_result_dict


def main(args):
    word_vec = None
    # intialize tokenizer
    if args.word_vec:
        word_vec = pickle.load(open(args.word_vec, 'rb'))
        print('complete loading word vectors')

    if args.use_bert:
        assert (args.bert_layer in ['avg'] + [str(i) for i in range(13)])

        bert_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       cache_dir=args.cache_dir if args.cache_dir else None)
        bert_config = AutoConfig.from_pretrained(args.model_name_or_path,
                                                 cache_dir=args.cache_dir if args.cache_dir else None)

        bert_config.update({"output_hidden_states": True})

    train_text, null_index = read_conll(args.train_file, ud_format=args.ud_format)
    if args.test_file != '':
        test_text, null_index = read_conll(args.test_file, ud_format=args.ud_format)
    else:
        test_text = train_text

    if os.path.exists(args.vocab_path):
        vocab = pickle.load(open(args.vocab_path, 'rb'))
        if args.use_chara:
            char_vocab = pickle.load(open(args.vocab_path + '.char', 'rb'))
        else:
            char_vocab = None
        vocab_freq = pickle.load(open(args.vocab_path + '.freq', 'rb'))
    else:
        vocab_text, _ = read_conll(args.vocab_file, ud_format=args.ud_format)
        vocab = build_vocab(vocab_text)
        pickle.dump(vocab, open(args.vocab_path, 'wb'))
        if args.use_chara:
            char_vocab = build_vocab_char(vocab_text)
            pickle.dump(char_vocab, open(args.vocab_path + '.char', 'wb'))
        else:
            char_vocab = None
        vocab_freq = set_vocab_freq(vocab_text, vocab)
        pickle.dump(vocab_freq, open(args.vocab_path + '.freq', 'wb'))

    if args.use_chara:
        w2c_list = expand_word_vocab(vocab, char_vocab)
        all_vocab_char_tensor, all_vocab_char_length_tensor = get_all_vocab_char_input(args, w2c_list)

    train_tags = [sent['tag'] for sent in train_text]
    test_tags = [sent["tag"] for sent in test_text]

    if args.use_bert:
        train_data = load_and_cache_examples(args, train_text, bert_tokenizer, cache_prefix=args.train_file,
                                             lowercase=args.do_lower_case, use_chara=args.use_chara, word_vocab=vocab)
        test_data = load_and_cache_examples(args, test_text, bert_tokenizer, cache_prefix=args.test_file,
                                            lowercase=args.do_lower_case, use_chara=args.use_chara, word_vocab=vocab)
    else:
        train_data = load_and_cache_examples_nonbert(args, train_text, vocab, cache_prefix=args.train_file,
                                                     lowercase=args.do_lower_case, tag_vocab=None,
                                                     char_vocab=char_vocab, use_chara=args.use_chara)
        test_data = load_and_cache_examples_nonbert(args, test_text, vocab, cache_prefix=args.test_file,
                                                    lowercase=args.do_lower_case, tag_vocab=None, char_vocab=char_vocab,
                                                    use_chara=args.use_chara)

    num_dims = len(train_data[0][0])
    print('complete reading data')

    print('#training sentences: %d' % len(train_data))
    print('#testing sentences: %d' % len(test_data))

    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device

    if args.use_bert:
        model = OrderPOS(args, bert_config=bert_config, vocab=vocab, char_vocab=char_vocab).to(device)
    else:
        model = OrderPOS(args, pretrained_embeddings=word_vec, vocab=vocab, char_vocab=char_vocab).to(device)

    if args.use_chara:
        model.set_all_vocab_char_input(all_vocab_char_tensor, all_vocab_char_length_tensor, device)

    model.set_word_freq(vocab_freq, device)

    full_eval_result_dicts = []

    wb_oracle_dict = None

    # be compatable with earlier approach
    if args.test and args.load_model_path is None:
        args.load_model_path = args.model_name

    if args.load_model_path is not None:
        partial_load(model, torch.load(args.load_model_path))

    if args.test:
        eval_result_dict = eval_model_for_loss(model, test_data, test_tags, args, sentences=test_text,
                                               print_path=args.tag_path, null_index=null_index)
        eval_result_dict2 = eval_model(model, test_data, test_tags, args, sentences=test_text, print_path=args.tag_path,
                                       null_index=null_index)
        for k in eval_result_dict2.keys():
            if 'loss' not in k:
                eval_result_dict[k] = eval_result_dict2[k]
        pprint(eval_result_dict)

    if args.train:
        if args.total_epochs > 0:
            args.total_steps = len(train_data) // args.batch_size * args.total_epochs
        if (args.warmup_steps > 0) and (args.warmup_steps < 1):
            args.warmup_steps = int(args.total_steps * args.warmup_steps)
        else:
            args.warmup_steps = int(args.warmup_steps)
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'reduce':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        else:
            raise NotImplementedError

        begin_time = time.time()
        print('begin training')

        set_seed(args.seed, args)

        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)
        model.train()
        best_epoch, best_loss, best_accuracy, best_vm = -1, 99999999999999, 0., 0.
        best_eval_dict = {}
        optimizer.zero_grad()
        total_train_iter = 0
        for epoch in range(args.epochs):
            set_seed(args.seed + epoch, args)
            # model.print_params()
            report_loss = 0
            report_ll = report_num_words = report_batches = 0
            report_word_acc = 0
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for train_iter, batch in enumerate(epoch_iterator):
                total_train_iter += 1
                batch = tuple(t.to(args.device) for t in batch)
                if args.use_bert:
                    inputs = {"input_ids": batch[0], "masks": batch[1], "offsets": batch[3]}
                    inputs['word_level_ids'] = batch[-1]
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet"] else None
                    )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                else:
                    inputs = {"input_ids": batch[0], "masks": batch[1]}
                if args.use_chara:
                    inputs['chars'] = batch[-2]
                    inputs['char_lengths'] = batch[-1].cpu()

                output_dict = model(**inputs)
                loss = output_dict['loss']
                num_words = output_dict['word_count']

                report_loss += loss.detach().cpu().numpy()
                report_num_words += num_words
                report_batches += 1

                loss.backward()

                if total_train_iter % args.accum_count == 0:
                    optimizer.step()

                    optimizer.zero_grad()

                if (args.total_steps > 0) and (total_train_iter > args.total_steps):
                    break

            if epoch % args.valid_nepoch == 0:
                train_log_dict = {'train_epoch': epoch,
                                  'train_loss': report_loss / report_batches,
                                  'total_time': time.time() - begin_time}
                wandb.log(train_log_dict)
                train_loss = report_loss / report_num_words

                print('epoch %d, loss %.4f, elapsed %.4f sec' % (
                epoch, report_loss / report_num_words, time.time() - begin_time))
                print('word acc %.4f' % (report_word_acc / report_batches))

            if epoch % args.valid_nepoch == 0:
                model.eval()
                tag_path = args.tag_path + '.epoch%d' % epoch if args.tag_path is not None else None
                eval_result_dict = eval_model_for_loss(model, test_data, test_tags, args, sentences=test_text,
                                                       print_path=tag_path, null_index=null_index)
                eval_result_dict2 = eval_model(model, test_data, test_tags, args, sentences=test_text,
                                               print_path=tag_path, null_index=null_index)
                for k in eval_result_dict2.keys():
                    if 'loss' not in k:
                        eval_result_dict[k] = eval_result_dict2[k]
                full_eval_result_dicts.append(eval_result_dict)
                test_loss = eval_result_dict['loss']
                accuracy = eval_result_dict['accuracy'] if 'accuracy' in eval_result_dict.keys() else None
                vm = eval_result_dict['vm'] if 'accuracy' in eval_result_dict.keys() else None
                if args.select_on_train:
                    select_loss = train_loss
                else:
                    select_loss = test_loss
                if select_loss < best_loss:
                    best_loss = select_loss
                    best_epoch, best_accuracy, best_vm = epoch, accuracy, vm
                    best_eval_dict = eval_result_dict
                    if args.use_bert:
                        bert_tokenizer.save_pretrained(args.save_dir)
                    torch.save(model.state_dict(), args.save_path)

                print('\n*****epoch %d*****\n' % (epoch))
                pprint(eval_result_dict)
                wb_eval_dict = {'dev_' + k: v for k, v in eval_result_dict.items()}
                wb_oracle_dict = update_oracle_dict(wb_eval_dict, wb_oracle_dict)
                wandb.log(wb_eval_dict)
                wandb.log(wb_oracle_dict)
                model.train()

                if total_train_iter > (args.warmup_steps + args.patience * (len(train_data) // args.batch_size)):
                    if best_epoch + args.patience < epoch:
                        break

                    if epoch >= args.max_epoch:
                        break

                if args.optimizer == 'reduce':
                    scheduler.step(eval_result_dict['pure_loss'])

            if (args.total_steps > 0) and (total_train_iter > args.total_steps):
                break
        result_dict = {'dev': best_eval_dict}

        if best_accuracy is not None:
            print('\n complete training, best epoch %f, loss %f, accuracy %f, vm %f\n' % (
            best_epoch, best_loss, best_accuracy, best_vm))
        else:
            print('\n complete training, best epoch %f, loss %f\n' % (best_epoch, best_loss))

        with open(args.result_save_path, 'w') as fw:
            json.dump(result_dict, fw)

        with open(args.full_result_save_path, 'w') as fw:
            for d in full_eval_result_dicts:
                json.dump(d, fw)
                fw.write('\n')


if __name__ == '__main__':
    wandb.login()
    parse_args = init_config()
    main(parse_args)
