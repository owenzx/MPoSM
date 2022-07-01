from __future__ import print_function

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.nn import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from tqdm import tqdm
from allennlp.nn.util import batched_span_select

from .utils import gumbel_softmax_sample, gumbel_softmax, PositionalEmbeddings, mask_tokens, get_accuracy_with_mask, \
    greedy_pred, set_mask_tokens, get_token_mask_only, mask_tokens_one_per_example

from transformers import AutoModel
from allennlp.modules.matrix_attention import LinearMatrixAttention, BilinearMatrixAttention
from allennlp.nn.util import masked_softmax

eps = 1e-7


def get_bert_word_rep(model, input_ids, attention_mask, token_type_ids, offsets, layer=12, no_position=False):
    if no_position:
        position_ids = torch.zeros(input_ids.size(), dtype=torch.long).cuda()
        token_type_ids = torch.zeros(input_ids.size(), dtype=torch.long).cuda()
        embeddings = model.embeddings.LayerNorm(model.get_input_embeddings()(input_ids) +
                                                model.embeddings.position_embeddings(position_ids) +
                                                model.embeddings.token_type_embeddings(token_type_ids))
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if layer == 'avg':
            all_hidden = torch.stack(outputs[2], dim=0)
            embeddings = torch.mean(all_hidden, dim=0)
        else:
            layer = int(layer)
            embeddings = outputs[2][layer]
    # concat mask with embedding
    embeddings = torch.cat((embeddings, attention_mask.unsqueeze(-1)), -1)

    span_embeddings, span_mask = batched_span_select(embeddings.contiguous(), offsets)
    span_mask = span_mask.unsqueeze(-1)
    span_embeddings *= span_mask  # zero out paddings
    span_embeddings_sum = span_embeddings.sum(2)
    span_embeddings_len = span_mask.sum(2)
    # Shape: (batch_size, num_orig_tokens, embedding_size)
    orig_embeddings = span_embeddings_sum / torch.clamp_min(span_embeddings_len, 1)
    # All the places where the span length is zero, write in zeros.
    orig_embeddings[(span_embeddings_len == 0).expand(orig_embeddings.shape)] = 0

    orig_masks = orig_embeddings[:, :, -1:].squeeze(-1)
    orig_embeddings = orig_embeddings[:, :, :-1]
    return orig_embeddings, orig_masks


class OrderPOS(nn.Module):
    def __init__(self, args, bert_config=None, pretrained_embeddings=None, vocab=None, char_vocab=None):
        super(OrderPOS, self).__init__()

        self.args = args
        self.device = args.device

        self.mask_rate = args.mask_rate
        self.hidden_units = args.hidden_units
        self.pos_embedding_dim = args.pos_embedding_dim
        self.n_heads = args.n_heads
        self.num_inds = args.num_inds
        self.dropout = args.dropout
        self.position_dim = self.hidden_units
        self.max_len = args.max_seq_length
        self.order_loss_type = args.order_loss
        self._gumbel_temp = args.gumbel_temp
        self._freeze_embeddings = args.freeze_embeddings
        self._pred_from_vocab = args.pred_from_vocab


        self.bind_xz = args.bind_xz

        if self.order_loss_type in ['marginal', 'word']:
            # one more state for [MASK] (the last state)
            self.num_state = args.num_state + 1
            self._mask_state = args.num_state
            self.num_real_state = args.num_state
        else:
            self.num_state = args.num_state
            self.num_real_state = args.num_state


        self.entropy_reg_weight = args.entropy_reg_weight
        if self.entropy_reg_weight == 0:
            self.entropy_reg = False
        else:
            self.entropy_reg = True

        self.kl_reg_weight = args.kl_reg_weight

        self.mean_loss = args.mean_loss

        self.use_gumbel = args.use_gumbel

        self.word_embedding_dim = 100
        self.enc_hidden_dim = args.encoder_hidden_dim
        self.enc_layer_num = args.enc_layer_num
        self.dec_layer_num = args.dec_layer_num
        self.vocab_size = len(vocab.keys()) + 2
        self._load_pretrained_vec = (pretrained_embeddings is not None)
        self._nonmask_loss = args.nonmask_loss
        self._use_chara = args.use_chara
        self._chara_model = args.chara_model
        if self._use_chara:
            self.char_vocab_size = len(char_vocab.keys()) + 2
            self.char_emb_dim = args.char_embedding_dim
            self.cemb = nn.Embedding(num_embeddings=self.char_vocab_size, embedding_dim=self.char_emb_dim,
                                     padding_idx=0)
            if args.chara_model == 'rnn':
                self.char_lstm = nn.LSTM(self.cemb.embedding_dim, self.cemb.embedding_dim, 1, bidirectional=True)
            elif args.chara_model == 'cnn':
                self.char_channel_size = 100
                self.char_channel_width = 5
                self.char_conv = nn.Sequential(
                    nn.Conv2d(1, self.char_channel_size, (self.char_emb_dim, self.char_channel_width)), nn.ReLU())
            else:
                raise NotImplementedError

        self.use_self_attention = args.self_attention
        if self.use_self_attention:
            assert (args.decoder == 'lstm')
            self._self_attention = LinearMatrixAttention(self.hidden_units, self.hidden_units, combination='x,y,x*y')
            self._post_attention_nn = torch.nn.Sequential(nn.Linear(self.hidden_units, self.hidden_units), nn.ReLU(),
                                                          nn.Dropout(self.dropout))

        self._encoder_type = args.encoder
        if self._encoder_type == 'bert':
            self._bert_layer = args.bert_layer
            self.encoder_model = AutoModel.from_pretrained(args.model_name_or_path,
                                                           cache_dir=args.cache_dir if args.cache_dir else None,
                                                           config=bert_config)

            self.bert_ff_layer_num = args.bert_ff_layer_num
            self.bert_ff_hidden_dim = args.bert_ff_hidden_dim

            _bert_ff = [nn.Linear(bert_config.hidden_size, self.bert_ff_hidden_dim), nn.ReLU(),
                        nn.Dropout(self.dropout)]
            for i in range(self.bert_ff_layer_num - 1):
                _bert_ff.append(nn.Linear(self.bert_ff_hidden_dim, self.bert_ff_hidden_dim))
                _bert_ff.append(nn.ReLU())
                _bert_ff.append(nn.Dropout(self.dropout))
            self._bert_ff = nn.Sequential(*_bert_ff)

            self._encoder_output_dim = self.bert_ff_hidden_dim

            # raise NotImplementedError

        elif self._encoder_type == 'local':
            if self._load_pretrained_vec:
                self._load_vec(pretrained_embeddings, vocab)
            else:
                self._word_embeddings = torch.nn.Embedding(num_embeddings=self.vocab_size,
                                                           embedding_dim=self.word_embedding_dim, padding_idx=0)
            if self.enc_layer_num == 0:
                self.encoder_model = lambda x: x
                self._encoder_output_dim = self.word_embedding_dim
            else:
                layers = [nn.Linear(self.word_embedding_dim, self.enc_hidden_dim), nn.ReLU(), nn.Dropout(self.dropout)]
                for _ in range(self.enc_layer_num - 1):
                    layers.append(nn.Linear(self.enc_hidden_dim, self.enc_hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(self.dropout))
                self.encoder_model = nn.Sequential(*layers)
                self._encoder_output_dim = self.enc_hidden_dim

        else:
            raise NotImplementedError

        if self._use_chara:
            self._pos_classification_layer = torch.nn.Linear(self._encoder_output_dim + 2 * self.char_emb_dim,
                                                                self.num_real_state)
        else:
            self._pos_classification_layer = torch.nn.Linear(self._encoder_output_dim, self.num_real_state)
        # 0 is the mask state
        self._pos_embeddings = torch.nn.Embedding(num_embeddings=self.num_state, embedding_dim=self.pos_embedding_dim)

        self._decoder_type = args.decoder
        if self._decoder_type == 'lstm':
            self._decoder = torch.nn.LSTM(input_size=self.pos_embedding_dim, hidden_size=self.hidden_units // 2,
                                          num_layers=self.dec_layer_num, batch_first=True, dropout=self.dropout,
                                          bidirectional=True)

        if self.order_loss_type in ['marginal', 'word']:
            self.output_mlp = nn.Sequential(
                nn.Linear(self.hidden_units, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_units, self.num_real_state)
            )

            self.output_mlp_word = nn.Sequential(
                nn.Linear(self.hidden_units, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_units, self.vocab_size)
            )

        if self.order_loss_type in ['marginal', 'word']:
            if self._pred_from_vocab:
                self._vocab_fc = nn.Sequential(
                    nn.Linear(self.pos_embedding_dim, self.word_embedding_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.word_embedding_dim, self.vocab_size)
                )
                if self._encoder_type != 'bert':
                    self._vocab_fc[-1].weight = self._word_embeddings.weight  # share embeddings

    def set_all_vocab_char_input(self, all_vocab_char_tensor, all_vocab_char_length_tensor, device):
        self._all_vocab_char_tensor = all_vocab_char_tensor.to(device)
        self._all_vocab_char_length_tensor = all_vocab_char_length_tensor.to(device)

    def set_word_freq(self, vocab_freq, device):
        assert (len(vocab_freq) == self.vocab_size)
        vocab_freq = torch.FloatTensor(vocab_freq).to(device)
        vocab_freq += 1
        self.vocab_prob = vocab_freq / vocab_freq.sum()

    def _load_vec(self, emb_dict, vocab):
        word_embeddings = np.zeros((self.vocab_size, self.word_embedding_dim))
        for word, word_idx in vocab.items():
            if word in emb_dict:
                word_embeddings[word_idx] = emb_dict[word]
            else:
                word_embeddings[word_idx] = np.random.normal(scale=0.6, size=(self.word_embedding_dim,))
        # UNK
        word_embeddings[len(vocab.keys()) + 1] = np.random.normal(scale=0.6, size=(self.word_embedding_dim,))

        self._word_embeddings = torch.nn.Embedding.from_pretrained(torch.FloatTensor(word_embeddings))

    def _forward_char_model(self, real_chars, real_char_lengths):
        real_char_num = real_chars.size(0)
        if self._chara_model == 'rnn':
            packed = pack_padded_sequence(self.cemb(real_chars), real_char_lengths.cpu(), batch_first=True,
                                          enforce_sorted=False)
            output, (final_h, final_c) = self.char_lstm(packed)
            final_h = final_h.view(self.char_lstm.num_layers, 2, -1,
                                   self.char_lstm.hidden_size)[-1]  # 2 x B x d_c
            # cembs = final_h.transpose(0, 1).contiguous().view(batch_size, word_len, -1)
            cembs = final_h.transpose(0, 1).contiguous().view(real_char_num, -1)

        elif self._chara_model == 'cnn':
            x = self.cemb(real_chars)  # num_word * word_len * char_dim
            x = x.transpose(1, 2)  # num_word * char_dim * word_len
            x = x.unsqueeze(1)  # num_word * 1 * char_dim * word_len
            x = self.char_conv(x).squeeze()  # num_word * char_channel_size * conv_len
            x = F.max_pool1d(x, x.size(2)).squeeze()  # num_word * char_channel_size
            cembs = x

        else:
            raise NotImplementedError

        return cembs

    def _get_embeddings_from_weight(self, input_ids, emb_weights):
        return F.embedding(input_ids, emb_weights, padding_idx=0)

    def _self_attention_module(self, input, mask):
        # input b x len x dim, mask b x len
        mask = mask.bool()
        attention_mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        attention_scores = self._self_attention(input, input)
        attention_distribution = masked_softmax(attention_scores, attention_mask, dim=-1)  # b x len x len
        sum_vec = torch.bmm(attention_distribution, input)
        attend_vec = self._post_attention_nn(sum_vec)
        return input + attend_vec

    def forward(self, input_ids, masks, chars=None, char_lengths=None, token_type_ids=None, offsets=None,
                gold_tags=None, greedy_gumbel=False, word_level_ids=None):

        batch_size = input_ids.size(0)
        # Step 1 encode
        if self._encoder_type == 'bert':
            if self._freeze_embeddings:
                with torch.no_grad():
                    bert_embeddings, encoded_mask = get_bert_word_rep(self.encoder_model, input_ids, masks,
                                                                      token_type_ids, offsets, layer=self._bert_layer)
            else:
                bert_embeddings, encoded_mask = get_bert_word_rep(self.encoder_model, input_ids, masks, token_type_ids,
                                                                  offsets, layer=self._bert_layer)

            bert_embeddings = self._bert_ff(bert_embeddings)

            encoded = bert_embeddings

        elif self._encoder_type == 'local':
            encoded_mask = masks
            input_embeddings = self._word_embeddings(input_ids)
            if self._freeze_embeddings:
                input_embeddings = input_embeddings.detach()
            encoded = self.encoder_model(input_embeddings)

        else:
            raise NotImplementedError
        batch_max_len = encoded.size(1)
        encoded_mask = encoded_mask.long()
        lengths = encoded_mask.sum(-1)

        if self._use_chara:
            # forward full_vocab to get character embeddings
            full_vocab_cembs = self._forward_char_model(self._all_vocab_char_tensor, self._all_vocab_char_length_tensor)
            if self._encoder_type == 'bert':
                char_embeddings = self._get_embeddings_from_weight(word_level_ids, full_vocab_cembs)
            else:
                char_embeddings = self._get_embeddings_from_weight(input_ids, full_vocab_cembs)
            encoded = torch.cat((encoded, char_embeddings), -1)

        # Step 2 gumble softmax to one-hot representation
        pos_logits = self._pos_classification_layer(encoded)
        bottom_log_probs = F.log_softmax(pos_logits, dim=-1)
        bottom_probs = pos_prob = F.softmax(pos_logits, dim=-1)
        zero_mask_probs = torch.zeros(*pos_prob.shape[:-1], 1, device=pos_prob.device)
        pos_prob_with_maskstate = torch.cat((zero_mask_probs, pos_prob), -1)

        if self.bind_xz:

            # Compute pz
            px = self.vocab_prob.unsqueeze(-1)
            logpx = torch.log(px)
            all_vocab_idxs = torch.arange(self.vocab_size, device=input_ids.device)
            all_vocab_embddings = self._word_embeddings(all_vocab_idxs)
            all_encoded = self.encoder_model(all_vocab_embddings)
            if self._use_chara:
                if self._encoder_type == 'bert':
                    raise NotImplementedError
                else:
                    all_char_embeddings = self._get_embeddings_from_weight(all_vocab_idxs, full_vocab_cembs)
                all_encoded = torch.cat((all_encoded, all_char_embeddings), -1)
            log_pzxs = F.log_softmax(self._pos_classification_layer(all_encoded), dim=-1)
            log_pzs = torch.logsumexp(log_pzxs + logpx, dim=0)

        if self.use_gumbel:
            if greedy_gumbel:
                pos_onehot = greedy_pred(F.log_softmax(pos_logits, dim=-1))
            else:
                pos_onehot = gumbel_softmax(F.log_softmax(pos_logits, dim=-1), self._gumbel_temp)

            pos_embeddings = torch.mm(pos_onehot.view(-1, pos_prob.size(-1)), self._pos_embeddings.weight[:-1, :])
            pos_embeddings = pos_embeddings.view(batch_size, -1, pos_embeddings.size(-1))

        else:
            pos_embeddings = torch.mm(pos_prob_with_maskstate.view(-1, pos_prob_with_maskstate.size(-1)),
                                      self._pos_embeddings.weight)
            pos_embeddings = pos_embeddings.view(batch_size, -1, pos_embeddings.size(-1))

        bottom_pred_tags = pos_prob.argmax(-1)

        # Step 3 mask reconstruction
        if self.order_loss_type == 'marginal':
            if greedy_gumbel:
                combined_input = pos_embeddings
                pos_masks = torch.zeros_like(bottom_pred_tags, device=bottom_pred_tags.device)
            else:
                mask_pos_embeddings = self._pos_embeddings(
                    torch.zeros(1, device=pos_embeddings.device, dtype=torch.long))
                new_embeddings, pos_masks, masked_labels = mask_tokens(pos_embeddings.clone(), encoded_mask,
                                                                       mask_pos_embeddings, mask_prob=self.mask_rate)

                pos_masks = pos_masks.unsqueeze(-1)
                combined_input = new_embeddings
            pos_masks = pos_masks.squeeze(-1)
            if self._decoder_type == 'lstm':
                packed = pack_padded_sequence(combined_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                # decoder_output, (fn, cn) = self._decoder(combined_input)
                decoder_output_packed, (fn, cn) = self._decoder(packed)
                decoder_output, _ = pad_packed_sequence(decoder_output_packed, batch_first=True,
                                                        total_length=combined_input.shape[1])
                if self.use_self_attention:
                    decoder_output = self._self_attention_module(decoder_output, encoded_mask)


            else:
                raise NotImplementedError
            logits = self.output_mlp(decoder_output)
            top_log_probs = F.log_softmax(logits, dim=-1)
            top_progs = F.softmax(logits, dim=-1)
            pred_tags = logits.argmax(-1)

            # useful mask here is pos_masks
            pos_masks = pos_masks.bool()

            if self._nonmask_loss:
                loss_masks = encoded_mask.bool()
            else:
                loss_masks = pos_masks.bool()

            selected_logits = logits[loss_masks]
            selected_logpz = F.log_softmax(selected_logits, dim=-1)

            all_pos_tags = torch.arange(self.num_real_state).to(self.device)
            all_pos_tags = all_pos_tags + 1  # avoid mask tag

            if self._pred_from_vocab:
                if self.bind_xz:
                    if self._encoder_type != 'bert':
                        selected_labels = input_ids[loss_masks]
                    else:
                        selected_labels = word_level_ids[loss_masks]
                    log_cond_pxz = (log_pzxs[selected_labels] + logpx[selected_labels]) - log_pzs.unsqueeze(0)
                    selected_logpxz = log_cond_pxz
                else:

                    all_pos_embeddings = self._pos_embeddings(all_pos_tags)
                    # pos_word_logits = self._vocab_fc(all_pos_embeddings)

                    pos_word_logits = self._vocab_fc(all_pos_embeddings)

                    pos_word_log_probs = F.log_softmax(pos_word_logits, dim=-1)

                    if self._encoder_type != 'bert':
                        selected_labels = input_ids[loss_masks]
                    else:
                        selected_labels = word_level_ids[loss_masks]
                    selected_logpxz = pos_word_log_probs.permute(1, 0)[selected_labels, :]

            else:
                raise NotImplementedError

            log_selected_px = torch.logsumexp(selected_logpz + selected_logpxz, dim=-1)
            # print(torch.exp(selected_px[:10]))

            order_loss = -log_selected_px

        elif self.order_loss_type == 'word':
            if greedy_gumbel:
                combined_input = pos_embeddings
                pos_masks = torch.zeros_like(bottom_pred_tags, device=bottom_pred_tags.device)
            else:
                mask_pos_embeddings = self._pos_embeddings(
                    torch.zeros(1, device=pos_embeddings.device, dtype=torch.long))
                new_embeddings, pos_masks, masked_labels = mask_tokens(pos_embeddings.clone(), encoded_mask,
                                                                       mask_pos_embeddings, mask_prob=self.mask_rate)

                pos_masks = pos_masks.unsqueeze(-1)
                combined_input = new_embeddings
            pos_masks = pos_masks.squeeze(-1)
            if self._decoder_type == 'lstm':
                packed = pack_padded_sequence(combined_input, lengths.cpu(), batch_first=True, enforce_sorted=False)
                decoder_output_packed, (fn, cn) = self._decoder(packed)
                decoder_output, _ = pad_packed_sequence(decoder_output_packed, batch_first=True,
                                                        total_length=combined_input.shape[1])
                if self.use_self_attention:
                    decoder_output = self._self_attention_module(decoder_output, encoded_mask)


            else:
                raise NotImplementedError
            word_logits = self.output_mlp_word(decoder_output)
            pred_words = word_logits.argmax(-1)
            if self._encoder_type == 'bert':
                all_labels = word_level_ids
            else:
                all_labels = input_ids
            acc = (pred_words == all_labels)[pos_masks.bool()].float().mean()

            # useful mask here is pos_masks
            pos_masks = pos_masks.bool()

            if self._nonmask_loss:
                loss_masks = encoded_mask.bool()
            else:
                loss_masks = pos_masks.bool()

            selected_logits = word_logits[loss_masks]
            selected_logpx = F.log_softmax(selected_logits, dim=-1)

            if self._encoder_type == 'bert':
                selected_labels = word_level_ids[loss_masks]
            else:
                selected_labels = input_ids[loss_masks]
            order_loss = F.cross_entropy(selected_logits, selected_labels)

            pred_tags = bottom_probs.argmax(-1)



        else:
            raise NotImplementedError

        if self.mean_loss:
            pure_loss = order_loss.mean()
            loss = order_loss.mean()
        else:
            pure_loss = order_loss.sum()
            loss = order_loss.sum()

        if self.kl_reg_weight > 0:
            if self.mean_loss:
                kl_div = F.kl_div(bottom_log_probs[encoded_mask], top_log_probs[encoded_mask], log_target=True,
                                  reduction='mean')
            else:
                kl_div = F.kl_div(bottom_log_probs[encoded_mask], top_log_probs[encoded_mask], log_target=True,
                                  reduction='sum')
            loss += (self.kl_reg_weight * kl_div)

        # print(pred_tags)
        if self.entropy_reg:
            ent_z = -(log_pzs * torch.exp(log_pzs)).sum()
            # print(pzs)
            loss = loss - self.entropy_reg_weight * ent_z

        output_dict = {'word_count': encoded_mask.sum().detach().cpu().item(),
                       'top_pred_tags': pred_tags.detach().cpu().numpy(),
                       'pred_len': encoded_mask.sum(-1).detach().cpu().numpy()}
        output_dict['bottom_pred_tags'] = bottom_pred_tags
        output_dict['pred_tags'] = output_dict['top_pred_tags']
        output_dict['pure_loss'] = pure_loss
        output_dict['loss'] = loss
        if self.order_loss_type == 'word':
            output_dict['reg_loss'] = acc
        else:
            output_dict['reg_loss'] = loss - pure_loss
        return output_dict
