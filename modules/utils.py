import torch
from collections import defaultdict
import math
import numpy as np
from math import log
from transformers import BertTokenizer, BertModel, BertConfig
from tqdm import tqdm
from allennlp.nn.util import batched_span_select

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Optional, Tuple, Union
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer


def update_oracle_dict(dev_dict, oracle_dict=None):
    if oracle_dict == None:
        oracle_dict = {}
        for k, v in dev_dict.items():
            if ('loss' in k) or ('vm' in k) or ('accuracy' in k):
                assert (k[:4] == 'dev_')
                oracle_dict['oracle_' + k[4:]] = v
    else:
        for k, v in dev_dict.items():
            if ('loss' in k) or ('vm' in k) or ('accuracy' in k):
                assert (k[:4] == 'dev_')
                oracle_k = 'oracle_' + k[4:]
                if 'loss' in k:
                    oracle_dict[oracle_k] = min(oracle_dict[oracle_k], v)
                elif ('vm' in k) or ('accuracy' in k):
                    oracle_dict[oracle_k] = max(oracle_dict[oracle_k], v)
                else:
                    raise ValueError

    return oracle_dict


def partial_load(model, load_dict):
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in load_dict.items() if k in model_dict}
    model_dict.update(filtered_dict)
    model.load_state_dict(model_dict)


def get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`int`, `optional`, defaults to 1):
            The number of hard restarts to use.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_accuracy_with_mask(pred: torch.Tensor, label: torch.Tensor, mask: torch.Tensor):
    total = mask.sum()
    corr = ((pred == label) & mask.bool()).sum()
    return torch.true_divide(corr, total)


def set_mask_tokens(inputs, set_mask, mask_id):
    labels = inputs.clone()

    masked_indices = set_mask.bool()
    inputs[masked_indices] = mask_id

    return inputs, set_mask, labels


def get_token_mask_only(input_mask: torch.Tensor, mask_prob=0.15) -> torch.Tensor:
    device = input_mask.device

    probability_matrix = torch.full(input_mask.shape, mask_prob).to(device)

    probability_matrix.masked_fill_(~(input_mask.bool()), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    return masked_indices.int()


def mask_tokens_one_per_example(inputs: torch.Tensor, input_mask: torch.Tensor, mask_id: Union[torch.Tensor, int]) -> \
Tuple[torch.Tensor, torch.Tensor]:
    device = inputs.device

    labels = inputs.clone()

    batch_size = inputs.shape[0]
    example_len = inputs.shape[1]
    example_real_lens = input_mask.sum(-1)

    mask_loc = torch.randint(low=0, high=example_len, size=(batch_size,)).to(device) % example_real_lens
    mask_loc = mask_loc.view(batch_size, 1)

    masked_indices = torch.zeros((batch_size, example_len)).to(device)

    masked_indices = masked_indices.scatter_(1, mask_loc, 1.0)

    masked_indices = masked_indices.bool()

    labels[~masked_indices] = -100
    inputs[masked_indices] = mask_id

    return inputs, masked_indices.int(), labels


def mask_tokens(inputs: torch.Tensor, input_mask: torch.Tensor, mask_id: Union[torch.Tensor, int], mask_prob=0.15) -> \
Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """

    device = inputs.device

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(input_mask.shape, mask_prob).to(device)
    probability_matrix.masked_fill_(~(input_mask.bool()), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    inputs[masked_indices] = mask_id

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, masked_indices.int(), labels


class PositionalEmbeddings(torch.nn.Module):
    """Implement the Positional Encoding function."""

    def __init__(self, input_dim: int, max_len: int = 5000) -> None:
        super().__init__()

        # Compute the positional encodings once in log space.
        positional_encoding = torch.zeros(max_len, input_dim, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, input_dim, 2).float() * -(math.log(10000.0) / input_dim)
        )
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding[:, : x.size(1)].repeat(x.size(0), 1, 1)


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


def greedy_pred(logits):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = logits
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return y_hard


def sanitize_wordpiece(wordpiece: str) -> str:
    """
    Sanitizes wordpieces from BERT, RoBERTa or ALBERT tokenizers.
    """
    if wordpiece.startswith("##"):
        return wordpiece[2:]
    elif wordpiece.startswith("Ġ"):
        return wordpiece[1:]
    elif wordpiece.startswith("▁"):
        return wordpiece[1:]
    else:
        return wordpiece


def intra_word_tokenize(tokenizer, string_tokens: List[str], add_special_tokens: bool
                        ) -> Tuple[List[Tuple], List[Optional[Tuple[int, int]]]]:
    tokens: List[Tuple] = []
    offsets: List[Optional[Tuple[int, int]]] = []
    if add_special_tokens:
        tokens.append((tokenizer.cls_token, tokenizer.convert_tokens_to_ids(tokenizer.cls_token)))
        # Do not include offsets since we do not need this token
        # offsets.append((0,1))
    for token_string in string_tokens:
        wordpieces = tokenizer.encode_plus(
            token_string,
            add_special_tokens=False,
            return_tensors=None,
            return_offsets_mapping=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        wp_ids = wordpieces["input_ids"]

        if len(wp_ids) > 0:
            offsets.append((len(tokens), len(tokens) + len(wp_ids) - 1))
            tokens.extend(
                (wp_text, wp_id)
                for wp_id, wp_text in zip(wp_ids, tokenizer.convert_ids_to_tokens(wp_ids))
            )
        else:
            offsets.append(None)
    if add_special_tokens:
        # offsets.append((len(tokens), len(tokens) + 1))
        tokens.append((tokenizer.sep_token, tokenizer.convert_tokens_to_ids(tokenizer.sep_token)))
    return tokens, offsets


def estimate_character_indices(tokenizer, text, token_ids, lowercase
                               ) -> List[Optional[Tuple[int, int]]]:
    """
    The huggingface tokenizers produce tokens that may or may not be slices from the
    original text.  Differences arise from lowercasing, Unicode normalization, and other
    kinds of normalization, as well as special characters that are included to denote
    various situations, such as "##" in BERT for word pieces from the middle of a word, or
    "Ġ" in RoBERTa for the beginning of words not at the start of a sentence.
    This code attempts to calculate character offsets while being tolerant to these
    differences. It scans through the text and the tokens in parallel, trying to match up
    positions in both. If it gets out of sync, it backs off to not adding any token
    indices, and attempts to catch back up afterwards. This procedure is approximate.
    Don't rely on precise results, especially in non-English languages that are far more
    affected by Unicode normalization.
    """

    token_texts = [
        sanitize_wordpiece(t) for t in tokenizer.convert_ids_to_tokens(token_ids)
    ]
    token_offsets: List[Optional[Tuple[int, int]]] = [None] * len(token_ids)
    if lowercase:
        text = text.lower()
        token_texts = [t.lower() for t in token_texts]

    min_allowed_skipped_whitespace = 3
    allowed_skipped_whitespace = min_allowed_skipped_whitespace

    text_index = 0
    token_index = 0
    while text_index < len(text) and token_index < len(token_ids):
        token_text = token_texts[token_index]
        token_start_index = text.find(token_text, text_index)

        # Did we not find it at all?
        if token_start_index < 0:
            token_index += 1
            # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
            allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
            continue

        # Did we jump too far?
        non_whitespace_chars_skipped = sum(
            1 for c in text[text_index:token_start_index] if not c.isspace()
        )
        if non_whitespace_chars_skipped > allowed_skipped_whitespace:
            # Too many skipped characters. Something is wrong. Ignore this token.
            token_index += 1
            # When we skip a token, we increase our tolerance, so we have a chance of catching back up.
            allowed_skipped_whitespace += 1 + min_allowed_skipped_whitespace
            continue
        allowed_skipped_whitespace = min_allowed_skipped_whitespace

        token_offsets[token_index] = (
            token_start_index,
            token_start_index + len(token_text),
        )
        text_index = token_start_index + len(token_text)
        token_index += 1
    return token_offsets


class ConllSent(object):
    """docstring for ConllSent"""

    def __init__(self, key_list=["word", "tag", "head"]):
        super(ConllSent, self).__init__()
        self.sent_dict = {}
        self.keys = key_list
        for key in key_list:
            self.sent_dict[key] = []

    def __getitem__(self, key):
        return self.sent_dict[key]

    def __setitem__(self, key, item):
        self.sent_dict[key] = item

    def __len__(self):
        return len(self.sent_dict["word"])


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def cast_to_int(s):
    try:
        return int(s)
    except ValueError:
        return s


def word2id(sentences):
    """map words to word ids

    Args:
        sentences: a nested list of sentences

    """
    ids = defaultdict(lambda: len(ids))
    id_sents = [[ids[word] for word in sent] for sent in sentences]
    return id_sents, ids


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)


def set_vocab_freq(sentences, vocab):
    vocab_size = len(vocab.keys())
    freqs = [0 for _ in range(vocab_size + 2)]
    for sent in sentences:
        for w in sent["word"]:
            if w in vocab:
                w_idx = vocab[w]
            else:
                w_idx = vocab_size
            freqs[w_idx] += 1
    return freqs


def build_vocab(sentences):
    vocab_dict = {}
    w_idx = 0
    for sent in sentences:
        for w in sent["word"]:
            if w not in vocab_dict:
                w_idx += 1
                vocab_dict[w] = w_idx

    return vocab_dict


def build_vocab_char(sentences):
    vocab_dict = {}
    c_idx = 0
    for sent in sentences:
        for w in sent["word"]:
            for c in w:
                if c not in vocab_dict:
                    c_idx += 1
                    vocab_dict[c] = c_idx

    return vocab_dict


def expand_word_vocab(word_vocab, char_vocab):
    w2clist = {}
    for w in word_vocab:
        w_idx = word_vocab[w]
        w2clist[w_idx] = []
        for c in w:
            w2clist[w_idx].append(char_vocab[c])
    return w2clist


def sents_to_vec(vec_dict, sentences):
    """read data, produce training data and labels.

    Args:
        vec_dict: a dict mapping words to vectors.
        sentences: A list of ConllSent objects

    Returns:
        embeddings: a list of tensors
        tags: a nested list of gold tags
    """
    vector_length = len(vec_dict[','])
    embeddings = []
    for sent in sentences:
        sample = [vec_dict[word] if word in vec_dict else [0.] * vector_length for word in sent["word"]]
        embeddings.append(sample)

    return embeddings


def sents_to_vec_bert(sentences, layer=12):
    """read data, produce training data and labels.

    Args:
        vec_dict: a dict mapping words to vectors.
        sentences: A list of ConllSent objects

    Returns:
        embeddings: a list of tensors
        tags: a nested list of gold tags
    """
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', cache_dir="dump_models/cache")
    model = BertModel.from_pretrained('bert-base-cased', cache_dir="dump_models/cache").cuda()
    embeddings = []
    for sent in tqdm(sentences):
        input_ids = torch.tensor([[tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(sent["word"]) +
                                  [tokenizer.sep_token_id]], dtype=torch.long).cuda()
        attention_mask = torch.tensor([[1] * len(input_ids)], dtype=torch.long).cuda()
        token_type_ids = torch.tensor([[0] * len(input_ids)], dtype=torch.long).cuda()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=True)
        embedding = outputs[2][layer].detach().cpu().numpy()[0][1:-1]
        embeddings.append(embedding)
    return embeddings


def sents_to_vec_bert_match(sentences, layer=12, no_position=False):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', cache_dir="dump_models/cache")
    config = BertConfig.from_pretrained('bert-base-cased', cache_dir="dump_models/cache")
    config.update({"output_hidden_states": True})
    model = BertModel.from_pretrained('bert-base-cased', config=config,
                                      cache_dir="dump_models/cache").cuda()
    model.eval()
    all_embeddings = []
    for sent in tqdm(sentences):
        words = [tokenizer.cls_token] + sent["word"] + [tokenizer.sep_token]
        new_words = []
        offsets = []
        for word in words:
            start = len(new_words)
            new_words.extend(tokenizer.tokenize(word))
            end = len(new_words) - 1
            offsets.append([start, end])
        input_ids = tokenizer.convert_tokens_to_ids(new_words)
        input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
        offsets = torch.tensor([offsets], dtype=torch.long).cuda()
        attention_mask = torch.tensor([[1] * len(new_words)], dtype=torch.long).cuda()
        token_type_ids = torch.tensor([[0] * len(new_words)], dtype=torch.long).cuda()
        if no_position:
            position_ids = torch.zeros(input_ids.size(), dtype=torch.long).cuda()
            token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long).cuda()
            embeddings = model.embeddings.LayerNorm(model.get_input_embeddings()(input_ids) +
                                                    model.embeddings.position_embeddings(position_ids) +
                                                    model.embeddings.token_type_embeddings(token_type_ids))
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            embeddings = outputs[2][layer]
        span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), offsets)
        span_mask = span_mask.unsqueeze(-1)
        span_embeddings *= span_mask  # zero out paddings
        span_embeddings_sum = span_embeddings.sum(2)
        span_embeddings_len = span_mask.sum(2)
        # Shape: (batch_size, num_orig_tokens, embedding_size)
        orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
        # All the places where the span length is zero, write in zeros.
        orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0
        orig_embeddings = orig_embeddings.detach().cpu().numpy()[0][1:-1]
        all_embeddings.append(orig_embeddings)
    return all_embeddings


def sents_to_tagid(sentences):
    """transform tagged sents to tagids,
    also return the look up table
    """
    ids = defaultdict(lambda: len(ids))
    id_sents = [[ids[tag] for tag in sent["tag"]] for sent in sentences]
    return id_sents, ids


def read_conll(fname, max_len=1e3, rm_null=True, prc_num=True, ud_format=False):
    sentences = []
    sent = ConllSent()

    null_total = []
    null_sent = []
    loc = 0
    with open(fname) as fin:
        for line in fin:
            if line != '\n':
                line = line.strip().split('\t')
                sent["head"].append((int(line[0]),
                                     cast_to_int(line[3])))
                if rm_null and line[2] == '-NONE-':
                    null_sent.append(loc)
                else:
                    if ud_format:
                        sent["tag"].append(line[3])
                    else:
                        sent["tag"].append(line[2])
                    if prc_num and is_number(line[1]):
                        sent["word"].append('0')
                    else:
                        sent["word"].append(line[1])

                loc += 1
            else:
                loc = 0
                if len(sent) > 0 and len(sent) <= max_len:
                    sentences.append(sent)
                    null_total.append(null_sent)

                null_sent = []
                sent = ConllSent()

    return sentences, null_total


def read_conll_from_lines(lines, max_len=1e3, rm_null=True, prc_num=True, ud_format=False):
    sentences = []
    sent = ConllSent()

    null_total = []
    null_sent = []
    loc = 0
    for line in lines:
        if line != '\n':
            line = line.strip().split('\t')
            sent["head"].append((int(line[0]),
                                 cast_to_int(line[3])))
            if rm_null and line[2] == '-NONE-':
                null_sent.append(loc)
            else:
                if ud_format:
                    sent["tag"].append(line[3])
                else:
                    sent["tag"].append(line[2])
                if prc_num and is_number(line[1]):
                    sent["word"].append('0')
                else:
                    sent["word"].append(line[1])

            loc += 1
        else:
            loc = 0
            if len(sent) > 0 and len(sent) <= max_len:
                sentences.append(sent)
                null_total.append(null_sent)

            null_sent = []
            sent = ConllSent()

    return sentences, null_total


def write_conll(fname, sentences, pred_tags, null_total):
    with open(fname, 'w') as fout:
        for (pred, null_sent, sent) in zip(pred_tags, null_total, sentences):
            word_list = sent["word"]
            head_list = sent["head"]
            length = len(sent) + len(null_sent)
            assert (length == len(head_list))
            pred_tag_list = [str(k.item()) for k in pred]
            for null in null_sent:
                pred_tag_list.insert(null, '-NONE-')
                word_list.insert(null, '-NONE-')
            for i in range(min(len(pred_tag_list), length)):
                fout.write("{}\t{}\t{}\t{}\n".format(
                    i + 1, word_list[i], pred_tag_list[i],
                    head_list[i][1]))
            fout.write('\n')


def input_transpose(sents, pad):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in range(max_len):
        sents_t.append([sent[i] if len(sent) > i else pad for sent in sents])
        masks.append([1 if len(sent) > i else 0 for sent in sents])

    return sents_t, masks


def to_input_tensor(sents, pad, device):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    sents, masks = input_transpose(sents, pad)

    sents_t = torch.tensor(sents, dtype=torch.float32, requires_grad=False, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, requires_grad=False, device=device)

    return sents_t, masks_t


def data_iter(data, batch_size, is_test=False, shuffle=True):
    index_arr = np.arange(len(data))
    # in_place operation

    if shuffle:
        np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        batch_ids = index_arr[i * batch_size: (i + 1) * batch_size]
        batch_data = [data[index] for index in batch_ids]

        if is_test:
            # batch_data.sort(key=lambda e: -len(e[0]))
            test_data = [data_tuple[0] for data_tuple in batch_data]
            tags = [data_tuple[1] for data_tuple in batch_data]

            yield test_data, tags

        else:
            # batch_data.sort(key=lambda e: -len(e))
            yield batch_data


def generate_seed(data, size, shuffle=True):
    index_arr = np.arange(len(data))
    # in_place operation

    if shuffle:
        np.random.shuffle(index_arr)

    seed = [data[index] for index in index_arr[:size]]

    return seed


def get_tag_set(tag_list):
    tag_set = set()
    tag_set.update([x for s in tag_list for x in s])
    return tag_set


def stable_math_log(val, default_val=-1e20):
    if val == 0:
        return default_val

    return math.log(val)


def unravel_index(input, size):
    """Unravel the index of tensor given size
    Args:
        input: LongTensor
        size: a tuple of integers

    Outputs: output,
        - **output**: the unraveled new tensor

    Examples::
        <<< value = torch.LongTensor(4,5,7,9)
        <<< max_val, flat_index = torch.max(value.view(4, 5, -1), dim=-1)
        <<< index = unravel_index(flat_index, (7, 9))
        <<< # output is a tensor with size (4, 5, 2)

    """
    idx = []
    for adim in size[::-1]:
        idx.append((input % adim).unsqueeze(dim=-1))
        input = input / adim
    idx = idx[::-1]
    return torch.cat(idx, -1)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreas:es linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


"""Noam Scheduler."""

from torch.optim import lr_scheduler


class NoamLR(lr_scheduler._LRScheduler):
    r"""Noam Learning rate schedule.
    Increases the learning rate linearly for the first `warmup_steps` training steps, then decreases it proportional to
    the inverse square root of the step number.
              ^
             / \
            /   `
           /     `
          /         `
         /               `
        /                       `
       /                                   `
      /                                                    `
     /                                                                              `
    /                                                                                                                  `
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimiser instance to modify the learning rate of.
    warmup_steps : int
        The number of steps to linearly increase the learning rate.
    Notes
    -----
    If step <= warmup_steps,
        scale = step / warmup_steps
    If step > warmup_steps,
        scale = (warmup_steps ^ 0.5) / (step ^ 0.5)
    """

    def __init__(self, optimizer, model_size, warmup_steps=4000):
        self.warmup_steps = warmup_steps
        self.model_size = model_size
        super(NoamLR, self).__init__(optimizer)

    def scale(self, step):
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup_steps ** (-1.5))

    def get_lr(self):
        scale = self.scale(max(1, self._step_count))
        return [base_lr * scale for base_lr in self.base_lrs]
