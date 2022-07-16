import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        print('attn_drop out:',attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))


        if mask is not None:
            #print(attn.shape)
            #print(mask.shape)
            attn = attn.masked_fill(mask == 1, -1e9)


        attn = self.dropout(F.softmax(attn, dim=-1))
        att_ft = torch.matmul(attn, v)

        #print(attn[0,0])

        return att_ft, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_att, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_att = d_att

        self.w_qs = nn.Linear(d_model, n_head * d_att, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_att, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_att, bias=False)
        self.fc = nn.Linear(n_head * d_att, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_att ** 0.5)

        self.dropout = nn.Dropout(dropout, inplace=True)
        #self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        #input shape: batch_size, len, dim

        d_att, n_head = self.d_att, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        #pre-norm
        #map to key, query, value
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_att)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_att)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_att)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        att_ft, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        att_ft = att_ft.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        att_ft = self.dropout(self.fc(att_ft))

        return att_ft, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        #self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout, inplace=True)

    def forward(self, x):

        x = self.w_2(F.gelu(self.w_1(x)))
        x = self.dropout(x)

        return x

class TransformerEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_att, dropout):
        super(TransformerEncoderLayer, self).__init__()
        #n_head, d_att, d_model, dropout
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.slf_attn = MultiHeadAttention(n_head=n_head, d_att=d_att, d_model=d_model,dropout=dropout)
        #d_in, d_hid, dropout
        self.pff_norm = nn.LayerNorm(d_model)
        self.pos_ffn = PositionwiseFeedForward(d_in=d_model, d_hid=d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        #remeber residual
        residual = enc_input
        #pre_norm
        enc_input = self.self_attn_norm(enc_input)
        #fee to self-attention
        enc_output, enc_slf_attn = self.slf_attn(
            q=enc_input, k=enc_input, v=enc_input, mask=slf_attn_mask)
        # add residual
        enc_output = residual + enc_output

        #remeber residual
        residual = enc_output
        #pre-norm
        enc_output = self.pff_norm(enc_output)
        #feed to ffn
        enc_output = self.pos_ffn(enc_output)
        #add residual
        enc_output = enc_output + residual


        return enc_output, enc_slf_attn

class TransformerEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, encoder_layer, layer_num):

        super(TransformerEncoder, self).__init__()

        self.layer_stack = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(layer_num)])


    def forward(self, input, mask=None, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = input
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_input=enc_output, slf_attn_mask=mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        else:
            return enc_output