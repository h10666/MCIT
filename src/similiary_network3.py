from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules.transformer import _get_clones
from transformers import BertConfig

from src.xbert import BertModel

config = BertConfig.from_json_file('/home/10501002/BERT/CNTR/src/config_bert.json')


class image_network(nn.Module):  # 把transformer与跨模态transformer结合在一起训练
    def __init__(self, image_dim, hidden_dim):
        super(image_network, self).__init__()
        self.linear = nn.Linear(image_dim, hidden_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, object_input):
        img_trans_input_1 = self.relu(self.dropout(self.linear(object_input)))
        img_trans_input = self.linear_out(img_trans_input_1)
        return img_trans_input

class text_network(nn.Module):  # 把自动编码器与transformer在一起训练,(S,B,D)
    def __init__(self, word_dim, hidden_dim):
        super(text_network, self).__init__()
        self.cap_encoder = nn.Sequential(
            nn.Linear(word_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, word_input):
        x_encoder = self.cap_encoder(word_input)
        return x_encoder

class cross_modal_fusion(nn.Module):
    def __init__(self,hidden_dim,heads,dim_feedforward,dropout):
        super(cross_modal_fusion, self).__init__()
        self.cross_att = nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=heads,dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self,emd1,emd2, key_padding_mask=None):
        emd1_emd2_att = self.cross_att(emd1,emd2,emd2,key_padding_mask=key_padding_mask)[0]
        emd1_emd2_att = emd1 + self.dropout1(emd1_emd2_att)
        emd1_emd2_att = self.norm1(emd1_emd2_att)
        emd1_emd2_att_out = self.linear2(self.dropout2(self.activation(self.linear1(emd1_emd2_att))))
        emd1_emd2_att_out = emd1_emd2_att + emd1_emd2_att_out
        emd1_emd2_att_out = self.norm2(emd1_emd2_att_out)
        return emd1_emd2_att_out
class multiencoder(nn.Module):
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

    def forward(self, src: Tensor,tag:Tensor,key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        embed1 = src
        embed2 = tag

        for mod in self.layers:
            output = mod(embed1,embed2, key_padding_mask=key_padding_mask)

        if self.norm is not None:
            output = self.norm(embed1)

        return output
def Info_NCE(temp,similiary_martix,postive_idx,):
    sim_len = similiary_martix.shape[0]
    loss = 0
    for i in range(sim_len):
        i_postive = 0
        i_all = 0
        for j in range(sim_len):
            if(int(postive_idx[i][j]) == 1):
                i_postive = i_postive + torch.exp(similiary_martix[i][j]/temp)
            i_all = i_all + torch.exp(similiary_martix[i][j]/temp)
        loss = loss - torch.log(i_postive/i_all)
    return loss

class classifier_network(nn.Module):
    def __init__(self, text_encoder, image_dim, cap_emd, hidden_dim):
        super(classifier_network, self).__init__()
        # self.attention_trans_encoder = TransformerEncoderLayer_1(d_model=word_dim,nhead=4,dim_feedforward=300,dropout=0.1)
        self.hidden_dim = hidden_dim
        self.image_net = image_network(image_dim, hidden_dim)
        self.text_net = text_network(cap_emd, hidden_dim)
        self.text_encoder =  BertModel.from_pretrained('bert-base-uncased', config=config, add_pooling_layer=False)
        cap_img_enco = cross_modal_fusion(hidden_dim=hidden_dim, heads=4, dim_feedforward=1024, dropout=0.1)
        self.multi_encoder = multiencoder(cap_img_enco,6)
        self.image_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.image_layer = nn.Linear(hidden_dim, hidden_dim)

        self.cap_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.cap_layer = nn.Linear(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim,2)

    def forward(self,img_inputs,text_inputs_all,ids):
        #
        len = img_inputs.shape[ 0 ]
        key_padding_mask = text_inputs_all.attention_mask
        key_padding_mask_bool = ~(key_padding_mask.bool())
        #
        idx = ids.view(-1, 1)  # 展开成n行1列   ,idx.t() ----> (1,n)
        pos_idx = torch.eq(idx, idx.t()).float()
        #
        image_inputs = F.normalize(self.image_net(img_inputs))  #8,36,256  ,128,36,256
        img_fusion_1 = F.softmax(self.image_mlp(image_inputs[ :, 1:, : ]))
        img_fusion_2 = self.image_layer(image_inputs[ :, 1:, : ])
        img_fusion = torch.bmm(img_fusion_1.permute(0, 2, 1), img_fusion_2).squeeze(1)
        img_out = img_fusion + image_inputs[ :, 0, : ]

        text_inputs = self.text_encoder(text_inputs_all.input_ids,key_padding_mask,return_dict=True, mode='text')
        text_inputs = text_inputs.last_hidden_state
        text_inputs = F.normalize(self.text_net(text_inputs))
        cap_fusion_1 = F.softmax(self.cap_mlp(text_inputs[ :, 1:, : ]), dim=1)
        cap_fusion_2 = self.cap_layer(text_inputs[ :, 1:, : ])
        cap_fusion = torch.bmm(cap_fusion_1.permute(0, 2, 1), cap_fusion_2).squeeze(1)
        cap_out = cap_fusion + text_inputs[ :, 0, : ]

        score_bf = torch.zeros(len, len)
        for i in range(len):
            score_bf[ i ] = F.cosine_similarity(cap_out[ i ].reshape(1, -1), img_out)
        score_bf = score_bf.cuda()
        loss_bf = Info_NCE(0.05,score_bf,pos_idx) + Info_NCE(0.05,score_bf.t(),pos_idx)

        scores_itm = torch.zeros(len,len)
        scores_itm_diag = torch.zeros(len,2)
        for i in range(len):
            image_inp = image_inputs[i].repeat(len,1,1)
            img_txt_pos_out = self.multi_encoder(image_inp.transpose(0,1),text_inputs.transpose(0,1),
                                                key_padding_mask_bool)
            img_txt_pos_cls_out = img_txt_pos_out.transpose(0,1)[ :, 0, : ]
            i_score = self.linear(img_txt_pos_cls_out)
            scores_itm[i] = i_score[:,1]
            scores_itm_diag[i] = i_score[i,:]
        loss_contrastive = Info_NCE(0.05,scores_itm,pos_idx)+ Info_NCE(0.05,scores_itm.t(),pos_idx)
        label = torch.ones(len,dtype=torch.long)
        loss_itm = F.cross_entropy(scores_itm_diag, label)


        loss =  loss_bf + loss_itm + loss_contrastive
        print("loss_bf",loss_bf)
        print("loss_itm",loss_itm)
        print("loss_contrastive",loss_contrastive)
        print("loss:",loss)
        return loss


    def forward_test(self, image_input, text_inputs,key_padding_mask,token_type_ids):
        len_i = image_input.shape[ 0 ]
        len_c = text_inputs.shape[ 0 ]
        key_padding_mask_bool = ~(key_padding_mask.bool())

        image_inputs = F.normalize(self.image_net(image_input))  # 8,36,256  ,128,36,256
        text_inputs = self.text_encoder(text_inputs, key_padding_mask, return_dict=True, mode='text')
        text_inputs = text_inputs.last_hidden_state
        text_inputs = F.normalize(self.text_net(text_inputs))

        scores_itm = torch.zeros(len_c, len_i)
        for i in range(len_i):
            image_inp = image_inputs[ i ].repeat(len_c, 1, 1)
            img_txt_pos_out = self.multi_encoder(image_inp.transpose(0, 1), text_inputs.transpose(0, 1),
                                                 key_padding_mask_bool)
            img_txt_pos_cls_out = img_txt_pos_out.transpose(0, 1)[ :, 0, : ]
            scores_itm[ i ] = self.linear(img_txt_pos_cls_out)[ :, 1 ]

        return scores_itm,scores_itm.t()


