import torch
import torch.nn as nn
import math
import torch.nn.functional as F

try:
    import ipdb
except ImportError:
    pass
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, linear_pre,linear_q,linear_v, mask=None):
        attnp = torch.bmm(linear_pre, linear_q.transpose(linear_pre, linear_q)) # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch
        attne = torch.bmm(linear_q, linear_v.transpose(linear_q, linear_v))
        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attnp = attne / math.sqrt(linear_pre)
        attne = attne / math.sqrt(linear_q)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attnp = torch.exp(attnp)
        attne = torch.exp(attne)
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attnp = attnp / attnp.sum(-1, keepdim=True)
        attnp = self.dropout(attnp)
        precompute = torch.bmm(attnp, linear_q)
        attne = attne / attne.sum(-1, keepdim=True)
        attne = self.dropout(attne)
        energy = torch.bmm(attne, linear_v) # (Batch, Seq, Feature)
        return energy, precompute

class ScoreAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim,dropout=0.1):
        super(ScoreAttention, self).__init__()
        #self.attend_dim = attend_dim
        #self.query_dim = query_dim
        #self.att_dim = att_dim

        self.linear_pre = nn.Linear(attend_dim, att_dim, bias=True)
        # self.linear_2 = nn.Linear(att_dim, att_dim, bias=True)
        self.linear_q = nn.Linear(query_dim, att_dim, bias=True)
        self.linear_v = nn.Linear(att_dim, 1, bias=True)
        if torch.__version__[:6] == '0.1.12':
            self.sm = nn.Softmax()
        else:
            self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context, precompute=None):
        """
        input: batch x dim
        context: batch x sourceL x dim
        """
        if precompute is None:
            precompute00 = self.linear_pre(context.contiguous().view(-1, context.size(2)))
            precompute = precompute00.view(context.size(0), context.size(1), -1)  # batch x sourceL x att_dim
        targetT = self.linear_q(input).unsqueeze(1)  # batch x 1 x att_dim

        tmp10 = precompute + targetT.expand_as(precompute)  # batch x sourceL x att_dim
        tmp20 = F.tanh(tmp10)
        energy = self.linear_v(tmp20.view(-1, tmp20.size(2))).view(tmp20.size(0), tmp20.size(1))  # batch x sourceL

        if self.mask is not None:
            energy = energy * (1 - self.mask) + self.mask * (-1e8)
        energy = F.softmax(energy, dim=1)

        return energy, precompute


class ScoreMultiAttention(nn.Module):
    def __init__(self, attend_dim, query_dim, att_dim):
        super(ScoreAttention, self).__init__()
        assert linear_q == linear_v
        self.attend_dim = attend_dim//3      #here 3 is the number of heads
        self.query_dim = query_dim//3
        self.att_dim = att_dim//3
       # self.attn_heads = nn.ModuleList([
        #    ScoreAttention(attend_dim, query_dim, att_dim,dropout)
        self.attention = ScoreAttention()
       #self.projection = nn.Linear(attend_dim, query_dim, att_dim)

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, linear_pre,linear_q,linear_v):
        
        xe , xp =[attn(linear_pre,linear_q,linear_v)
            for i,attn in enumerate(self.attn_heads) ]


        xe,xp = self.attention(attend_dim, query_dim, att_dim,dropout)

        energy = torch.cat(xe,dim = linear_q)
        precompute = torch.cat(xp,dim = linear_v)
       # x= self.projection(x)

        return energy, precompute

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.att_dim) + ' * ' + '(' \
               + str(self.attend_dim) + '->' + str(self.att_dim) + ' + ' \
               + str(self.query_dim) + '->' + str(self.att_dim) + ')' + ')'