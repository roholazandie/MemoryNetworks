import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_var
import copy
import math


class MemNN(nn.Module):
    def __init__(self, vocab_size, embd_size, ans_size, max_story_len, hops=3, dropout=0.2, te=True, pe=True):
        super(MemNN, self).__init__()
        self.hops = hops
        self.embd_size = embd_size
        self.temporal_encoding = te
        self.position_encoding = pe

        init_rng = 0.1
        self.dropout = nn.Dropout(p=dropout)
        self.M = nn.ModuleList([nn.Embedding(vocab_size, embd_size) for _ in range(hops + 1)])
        for i in range(len(self.M)):
            self.M[i].weight.data.normal_(0, init_rng)
            self.M[i].weight.data[0] = 0  # for padding index
        self.B = self.M[0]  # query encoder

        # Temporal Encoding: see 4.1
        if self.temporal_encoding:
            self.TA = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))
            self.TC = nn.Parameter(torch.Tensor(1, max_story_len, embd_size).normal_(0, 0.1))

    def get_position_embedding(self, story_sent_len, story_len, batch_size):
        J = story_sent_len
        d = self.embd_size
        pe = to_var(torch.zeros(J, d))  # (story_sent_len, embd_size)
        for j in range(1, J + 1):
            for k in range(1, d + 1):
                l_kj = (1 - j / J) - (k / d) * (1 - 2 * j / J)
                pe[j - 1][k - 1] = l_kj
        pe = pe.unsqueeze(0).unsqueeze(0)  # (1, 1, story_sent_len, embd_size)
        pe = pe.repeat(batch_size, story_len, 1, 1)  # (bs, story_len, story_sent_len, embd_size)

        return pe


    def forward(self, x, question):
        # x (bs, story_len, story_sent_len)
        # q (bs, q_sent_len)

        batch_size = x.size(0)
        story_len = x.size(1)
        story_sent_len = x.size(2)

        # Position Encoding
        if self.position_encoding:
            pe = self.get_position_embedding(story_sent_len, story_len, batch_size)

        x = x.view(batch_size * story_len, -1)  # (bs*story_sent_len, story_sent_len)

        query_embeds = self.dropout(self.B(question))  # (bs, q_sent_len, embd_size)
        query_embeds = torch.sum(query_embeds, 1)  # (bs, embd_size)

        # Adjacent weight tying
        for i in range(self.hops):
            in_memory_embeds = self.dropout(self.M[i](x))  # (bs*story_len, story_sent_len, embd_size)
            in_memory_embeds = in_memory_embeds.view(batch_size, story_len, story_sent_len, -1)  # (bs, story_len, story_sent_len, embd_size)
            if self.position_encoding:
                in_memory_embeds *= pe  # (bs, story_len, story_sent_len, embd_size)
            in_memory_embeds = torch.sum(in_memory_embeds, 2)  # (bs, story_len, embd_size) bag of words(vectors): this shows we use one vector for all of the tokens in sentence
            if self.temporal_encoding:
                in_memory_embeds += self.TA.repeat(batch_size, 1, 1)[:, :story_len, :]

            out_memory_embeds = self.dropout(self.M[i + 1](x))  # (bs*story_len, story_sent_len, embd_size)
            out_memory_embeds = out_memory_embeds.view(batch_size, story_len, story_sent_len, -1)  # (bs, story_len, story_sent_len, embd_size)
            out_memory_embeds = torch.sum(out_memory_embeds, 2)  # (bs, story_len, embd_size)
            if self.temporal_encoding:
                out_memory_embeds += self.TC.repeat(batch_size, 1, 1)[:, :story_len, :]  # (bs, story_len, embd_size)

            attention = torch.bmm(in_memory_embeds, query_embeds.unsqueeze(2)).squeeze()  # (bs, story_len)
            probs = F.softmax(attention, -1).unsqueeze(1)  # (bs, 1, story_len)
            memory_output = torch.bmm(probs, out_memory_embeds).squeeze(1)  # use in_memory_embeds as out_memory_embeds, (bs, embd_size)
            query_embeds = memory_output + query_embeds  # (bs, embd_size)

        W = torch.t(self.M[-1].weight)  # (embd_size, vocab_size)
        out = torch.bmm(query_embeds.unsqueeze(1), W.unsqueeze(0).repeat(batch_size, 1, 1)).squeeze()  # (bs, ans_size)

        return F.log_softmax(out, -1)
