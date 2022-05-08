import torch
from torch.autograd import Variable
import torch.nn as nn
from transformers import BertModel, BertTokenizer
# from torchcrf import CRF
import torch
import torch.nn as nn
from crf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import torch.nn.functional as F
from transformers.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertLayerNorm, BertSelfAttention

class Highway(nn.Module):
    def __init__(self, input_dim, num_layers=1):
        super(Highway, self).__init__()

        self._layers = nn.ModuleList([nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)])
        for layer in self._layers:
            # Bias the highway layer to just carry its input forward.
            # Set the bias on B(x) to be positive, then g will be biased to be high
            # The bias on B(x) is the second half of the bias vector in each linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs):
        current_inputs = inputs
        for layer in self._layers:
            linear_part = current_inputs
            projected_inputs = layer(current_inputs)

            nonlinear_part, gate = projected_inputs.chunk(2, dim=-1)
            nonlinear_part = torch.relu(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_inputs = gate * linear_part + (1 - gate) * nonlinear_part
        return current_inputs


class BERT_LISM_CRF_teacher(nn.Module):

    def __init__(self, args,tagset_size):
        super(BERT_LISM_CRF_teacher, self).__init__()
        self.args=args
        if args.use_bert==True:
            self.encoder = BertModel.from_pretrained(args.bert)
            # self.encoder.requires_grad_(False)
            # self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            # self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            # self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False
            # unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False
                for ele in self.args.unfreeze_layers:
                    if ele in name:
                        param.requires_grad = True
                        break
            self.config = self.encoder.config
            self.attention = BertAttention(self.config)
            self.crossattention = BertAttention(self.config)
            self.intermediate = BertIntermediate(self.config)
            self.output = BertOutput(self.config)
        else:
            self.encoder=nn.Embedding(args.vocab_size,args.embedding)

        self.dropout = nn.Dropout(args.dropout)
        # self.need_birnn = need_birnn
        self.highway=Highway(args.rnn_dim * 2, 1)
        self.lstm = nn.GRU(args.bert_embedding,args.rnn_dim, num_layers=args.rnn_layer, batch_first=True,
                            bidirectional=True)
        if self.args.use_idcnn:
            self.idcnn = IDCNN(input_size=args.embedding, filters=64)
        out_dim = args.rnn_dim * 2
        self.crf = CRF(tagset_size, self.args.use_cuda)
        if self.args.use_lstm:
            self.hidden2tag = nn.Linear(out_dim, tagset_size + 2)
        else:
            if self.args.use_bert:
                self.hidden2tag = nn.Linear(args.bert_embedding, tagset_size+2)
            else:
                self.hidden2tag = nn.Linear(64, tagset_size + 2)
        # self.hidden2tag = nn.Linear(args.bert_embedding, tagset_size + 2)


    def forward(self, input_ids, input_mask, targets):
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)

        if self.args.use_bert==True:
            outputs, _ = self.encoder(input_ids, attention_mask=input_mask)
            input_mask.requires_grad = False
            outputs = outputs * (input_mask.unsqueeze(-1).float())
            outputs = self.highway(outputs)
            sequence_output = outputs
        else:
            sequence_output=self.encoder(input_ids)

        if self.args.use_lstm==False:
            sequence_output = self.idcnn(sequence_output, seq_length)
        else:
            sequence_output, _ = self.birnn(sequence_output)

        sequence_output = self.dropout(sequence_output)
        output = self.hidden2tag(sequence_output)
        maskk = input_mask.ge(1)
        total_loss = self.crf.neg_log_likelihood_loss(output, maskk, targets)
        scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
        return total_loss / outputs.size(0), tag_seq

    def tag_outputs(self, input_ids, input_mask, targets,sen_len):
        seq_length = input_ids.size(1)
        if self.args.use_bert==True:
            outputs, _ = self.encoder(input_ids, attention_mask=input_mask)
            input_mask.requires_grad = False
            outputs = outputs * (input_mask.unsqueeze(-1).float())
        else:
            outputs = self.encoder(input_ids)

        batch_size = outputs.size(0)
        # output, _=self.lstm(outputs)
        if self.args.use_lstm:
            sort_len, perm_idx = sen_len.sort(0, descending=True)
            _, un_idx = torch.sort(perm_idx, dim=0)
            x_input = outputs[perm_idx]
            packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)
            packed_out, _ = self.lstm(packed_input)
            output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
            # sequence_output = torch.index_select(output.cuda(), 0, un_idx.cuda())
            if self.args.use_cuda:
                sequence_output = torch.index_select(output.cuda(), 0, un_idx.cuda())
            else:
                sequence_output = torch.index_select(output, 0, un_idx)
            outputs = self.dropout(sequence_output)
        else:
            if self.args.use_idcnn:
                outputs = self.idcnn(outputs, seq_length)

        if self.args.use_self_attention:
            self_attention_outputs = self.attention(outputs)
            attention_output = self_attention_outputs[0]
            intermediate_output = self.intermediate(attention_output)  # 增大空间表示  增强模型表示能力   最后一维扩大为config.hidden_act 并激活
            outputs = self.output(intermediate_output, attention_output)  # bert输出 一个全连接层 增加LayerNorm加快模型收敛速度
        output = self.hidden2tag(outputs)
        maskk = input_mask.ge(1)
        total_loss = self.crf.neg_log_likelihood_loss(output, maskk, targets)
        scores, tag_seq = self.crf._viterbi_decode(output, input_mask)
        return total_loss / batch_size, tag_seq,output

        # sort_len, perm_idx = sen_len.sort(0, descending=True)
        # _, un_idx = torch.sort(perm_idx, dim=0)
        # x_input = outputs[perm_idx]
        # packed_input = pack_padded_sequence(x_input, sort_len, batch_first=True)
        # packed_out, _ = self.birnn(packed_input)
        # output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        # sequence_output = torch.index_select(output, 0, un_idx)
        # # sequence_output = torch.index_select(output.cuda(), 0, un_idx.cuda())
        # sequence_output = torch.index_select(output, 0, un_idx)
        # sequence_output = self.dropout(sequence_output)
        # output = self.hidden2tag(sequence_output)
        # loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        # active_loss = input_mask.view(-1) == 1
        # active_logits = output.view(-1, self.num_tags)[active_loss]
        # active_labels = targets.view(-1)[active_loss]
        # loss = loss_fct(active_logits, active_labels)
        # output=torch.max(output,-1)
        # return loss, output


    def batchify(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        # sent_len_all=[ i.cpu() for i in sent_ids]
        max_sent_len = self.args.max_seq
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        target_ids=torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        for idx, (seq, seqlen,target) in enumerate(zip(sent_ids, sent_lens,targets)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
            target_ids[idx, :seqlen]=torch.LongTensor(target)
        if self.args.use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = target_ids.cuda()
            sent_lens = torch.tensor(sent_lens).cpu()
        else:
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            targets = target_ids.cpu()
            sent_lens=torch.tensor(sent_lens).cpu()
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info

    def batchify_create(self, batch_list):
        batch_size = len(batch_list)
        sent_idx = [ele[0] for ele in batch_list]
        sent_ids = [ele[1] for ele in batch_list]
        targets = [ele[2] for ele in batch_list]
        x_y=[ele[3] for ele in batch_list]
        sent_lens = list(map(len, sent_ids))
        # sent_len_all=[ i.cpu() for i in sent_ids]
        max_sent_len = self.args.max_seq
        input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
        attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
        target_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()

        for idx, (seq, seqlen, target) in enumerate(zip(sent_ids, sent_lens, targets)):
            input_ids[idx, :seqlen] = torch.LongTensor(seq)
            attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
            target_ids[idx, :seqlen] = torch.LongTensor(target)
        if self.args.use_cuda:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            targets = target_ids.cuda()
            sent_lens = torch.tensor(sent_lens).cpu()
        else:
            input_ids = input_ids.cpu()
            attention_mask = attention_mask.cpu()
            targets = target_ids.cpu()
            sent_lens = torch.tensor(sent_lens).cpu()
        info = {"seq_len": sent_lens, "sent_idx": sent_idx}
        return input_ids, attention_mask, targets, info,x_y

# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class IDCNN(nn.Module):
    """
      (idcnns): ModuleList(
    (0): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (1): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (2): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
    (3): Sequential(
      (layer0): Conv1d(10, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer1): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(1,))
      (layer2): Conv1d(1, 1, kernel_size=(3,), stride=(1,), padding=(2,), dilation=(2,))
    )
  )
)
    """
    def __init__(self, input_size, filters, kernel_size=3, num_block=4):
        super(IDCNN, self).__init__()
        self.layers = [
            {"dilation": 1},
            {"dilation": 1},
            {"dilation": 2}]
        net = nn.Sequential()
        norms_1 = nn.ModuleList([LayerNorm(400) for _ in range(len(self.layers))])
        norms_2 = nn.ModuleList([LayerNorm(400) for _ in range(num_block)])
        for i in range(len(self.layers)):
            dilation = self.layers[i]["dilation"]
            single_block = nn.Conv1d(in_channels=filters,
                                     out_channels=filters,
                                     kernel_size=kernel_size,
                                     dilation=dilation,
                                     padding=kernel_size // 2 + dilation - 1)
            net.add_module("layer%d"%i, single_block)
            net.add_module("relu", nn.ReLU())
            net.add_module("layernorm", norms_1[i])

        self.linear = nn.Linear(input_size, filters)
        self.idcnn = nn.Sequential()


        for i in range(num_block):
            self.idcnn.add_module("block%i" % i, net)
            self.idcnn.add_module("relu", nn.ReLU())
            self.idcnn.add_module("layernorm", norms_2[i])

    def forward(self, embeddings, length):
        embeddings = self.linear(embeddings)
        embeddings = embeddings.permute(0, 2, 1)
        output = self.idcnn(embeddings).permute(0, 2, 1)
        return output


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2





