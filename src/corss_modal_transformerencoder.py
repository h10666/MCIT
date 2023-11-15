from torch import nn
from torch.nn import Module, LayerNorm, Dropout
import torch.nn.functional as F
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
import math
from typing import Optional
import torch
from torch import Tensor




#pisition-encoder
#PE为2维矩阵，大小跟输入的embedding的维度一样，行表示词语，列表示词向量；
#pos表示词语在句子中的位置，dmodel表示词向量的维度，i表示词向量的位置。
#所以可以表示在每个词语的词向量的偶数位置添加sin变量，奇数位置添加cos变量。以此来填满整个PE矩阵
#然后在添加到input embedding中去，这样就完成了位置编码的引入了。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len,dropout=0.1 ):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #CLS(开始)+Seq(序列)+Sqe(结尾)
        #定义一个PE矩阵并零初始化，shape(max_len,d_model)
        pe = torch.zeros(max_len, d_model)
        #定义一个position矩阵，shape是(max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #定义一个div_term,这是一个一维向量，长度是dmodel/2
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        #pe[:,0::2]，表示所有行取奇数列。
        pe[:, 0::2] = torch.sin(position * div_term)
        #pe[:,1::2]，表示取所有行偶数列
        pe[:, 1::2] = torch.cos(position * div_term)
        #这里对每一行的每一维度赋值，即做位置编码。
        self.pe = pe.unsqueeze(0).transpose(0, 1).cuda()
        # self.register_buffer('pe', pe)

    def forward(self, x,cap_len):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[ :x.size(0), : ]
        return self.dropout(x)


class cross_modal_fusion(nn.Module):
    def __init__(self,hidden_dim,heads,dim_feedforward,dropout):
        super(cross_modal_fusion, self).__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=heads,dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.activation = nn.GELU()
        self.norm1 = LayerNorm(hidden_dim)
        self.norm2 = LayerNorm(hidden_dim)

    def forward(self,emd1,emd2, key_padding_mask=None):
        emd1_emd2_att = self.cross_att(emd1,emd2,emd2,key_padding_mask=key_padding_mask)[0]
        emd1_emd2_att = emd1 + self.dropout1(emd1_emd2_att)
        emd1_emd2_att = self.norm1(emd1_emd2_att)
        emd1_emd2_att_out = self.linear2(self.dropout2(self.activation(self.linear1(emd1_emd2_att))))
        emd1_emd2_att_out = emd1_emd2_att + emd1_emd2_att_out
        emd1_emd2_att_out = self.norm2(emd1_emd2_att_out)
        return emd1_emd2_att_out


class multiencoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(multiencoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor,src_att:Tensor,key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        embed1 = src
        embed2 = src_att

        for mod in self.layers:
            output = mod(embed1,embed2, key_padding_mask=key_padding_mask)

        if self.norm is not None:
            output = self.norm(embed1)

        return output

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def cosine_similarity(x1, x2, dim=2, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    #torch.squeeze() 这个函数主要对数据的维度进行压缩，去掉维数为1的的维度

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    im = l2norm(im)
    s = l2norm(s)
    # print("im:",im)
    # print("s:",s)
    return im.mm(s.t())
class Contrastive(nn.Module):
    #Contrastive_loss，大致方法就是，讲s_n与s_p嵌入到相似对中，并寻求最小的(s_n-s_p)
    #与此同时,在(s_n-s_p)中增大s_p,就是减少了s_n
    #1.该方法缺少训练的灵活性，s_n和s_p的惩罚强度被限制为相等。即在给定指定的损失函数情况下，关于s_n和s_p的梯度具有相同的幅度。
    #2.该方法收敛状态不明确，优化(s_n-s_p)通常导致判定边界为s_p-s_n=m(m是边际)，这里m就是margin.
    #该判决边界允许用于收敛的模糊性,比如(s_n0,s_p0)=(0.2,0.5),(s_n1,s_p1)=(0.4,0.7)模糊边界都是0.3，但是s_n1与s_p0很相近
    #在余弦相似度度量下，我们期望s_p->1,s_n->0

    def __init__(self, margin=0,  max_violation=False):
        super(Contrastive, self).__init__()
        self.margin = margin

        self.max_violation = max_violation
    def compute_contrastive_loss(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)
        d1 = diagonal.expand_as(scores)
        cost_s = abs((self.margin + scores - d1))
        mask = torch.eye(scores.size(0)) > .5
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
        return cost_s.sum()

    def compute_contrastive_loss1(self, scores):
        diagonal = scores.diag().view(scores.size(0), 1)

        d1 = diagonal.expand_as(scores)    #将diagonal变成score一样size()的tensor
        d2 = diagonal.t().expand_as(scores)    #diagonal转置后,变成与score一样size()的tensor

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = abs((self.margin + scores - d1))

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        # print(mask)
        I = mask
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)
        # print("cost_s:", cost_s)
        # print("cost_im:", cost_im)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        # print(" cost_s.sum()", cost_s.sum())
        # print("cost_im.sum():",cost_im.sum())

        return (cost_s.sum() + cost_im.sum())

def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score
def dot_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())
class ContrastiveLoss(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    # def forward(self, im, s):
    def forward(self, scores):
        # compute image-sentence score matrix
        # print("scores:",scores)
        return self.compute_contrastive_loss(scores)

class ContrastiveLoss1(Contrastive):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, max_violation=False):
        super(ContrastiveLoss1, self).__init__()
        self.margin = margin
        self.sim = cosine_sim
        self.max_violation = max_violation

    # def forward(self, im, s):
    def forward(self, scores):
        # compute image-sentence score matrix
        # print("scores:",scores)
        return self.compute_contrastive_loss1(scores)
#Circle_Loss:圆形损失，是对Contrastive_Loss的一种改进
#需要设置两个超参数:都是经验设置两个超参数
#m:margin,是松弛因子(relax factor)，主要控制圆形决策边界的半径。
#gamma:是scale factor

class Circle_Loss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(Circle_Loss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, scores:Tensor) -> Tensor:
        a_0 = scores.size(0)
        a_1 = scores.size(1)
        sp = torch.zeros(a_0)
        sn = torch.zeros((a_1-1)*a_0)
        i_n = 0
        for i_0 in range(a_0):
            for i_1 in range(a_1):
                if i_0 == i_1:
                    sp[i_0] = scores[i_0][i_1]
                else:
                    sn[i_n] = scores[i_0][i_1]
                    i_n += 1

        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = torch.log(1 + torch.clamp_max(torch.exp(logit_n).sum() * torch.exp(logit_p).sum(), max=1e38))

        return loss


class Info_NCE(torch.nn.Module):
    def __init__(self,temp):
        super(Info_NCE, self).__init__()
        self.temp = temp
    def forward(self,similiary_martix):
        sim_len = similiary_martix.shape[0]
        d1 = similiary_martix.diag().view(-1,1)
        i_loss = 0
        for i in range(sim_len):
            i_nums = similiary_martix[i]
            i_sum = 0
            for i_0 in i_nums:
                i_sum = i_sum + torch.exp(i_0/self.temp)
            i_loss = i_loss - torch.log((torch.exp(d1[i]/self.temp))/i_sum)
        loss = i_loss
        return loss







# class TransformerEncoderLayer_1(Module):
#     r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
#     This standard encoder layer is based on the paper "Attention Is All You Need".
#     Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
#     Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
#     Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
#     in a different way during application.
#
#     Args:
#         d_model: the number of expected features in the input (required).
#         nhead: the number of heads in the multiheadattention models (required).
#         dim_feedforward: the dimension of the feedforward network model (default=2048).
#         dropout: the dropout value (default=0.1).
#         activation: the activation function of intermediate layer, relu or gelu (default=relu).
#
#     Examples::
#         # >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
#         # >>> src = torch.rand(10, 32, 512)
#         # >>> out = encoder_layer(src)
#     """
#
#     def __init__(self, d_model, nhead, dim_feedforward=2048,k_dim=None,v_dim=None,dropout=0.1, activation="relu"):
#         super(TransformerEncoderLayer_1, self).__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,kdim=k_dim,vdim=v_dim)
#         # Implementation of Feedforward model
#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)
#
#         self.norm1 = LayerNorm(d_model)
#         self.norm2 = LayerNorm(d_model)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#
#         self.activation = _get_activation_fn(activation)
#
#     def __setstate__(self, state):
#         if 'activation' not in state:
#             state['activation'] = F.relu
#         super(TransformerEncoderLayer_1, self).__setstate__(state)
#
#     def forward(self, Q: Tensor, K: Tensor,V: Tensor,src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#
#         Shape:
#             see the docs in Transformer class.
#         """
#
#         src2 = self.self_attn(Q, K, V, attn_mask=src_mask,
#                               key_padding_mask=src_key_padding_mask)[0]
#         src = Q + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src
#
#     # def get_params(self):
#     #
#     #
#     #     params = list(self.self_attn.parameters())
#     #     params += list(self.linear1.parameters())
#     #     params += list(self.linear2.parameters())
#     #
#     #
#     #
#     #     return params