import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

from xbert import BertConfig,BertModel

config = BertConfig.from_json_file('/home/10501002/BERT/CNTR/src/config_bert.json')

class image_network(nn.Module):   #把transformer与跨模态transformer结合在一起训练
    def __init__(self,image_dim,hidden_dim):
        super(image_network, self).__init__()
        self.norm1 = nn.LayerNorm(image_dim)
        self.linear = nn.Linear(image_dim,hidden_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.linear_out = nn.Linear(hidden_dim,hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward (self,object_input):
        object_input = self.norm1(object_input)
        img_trans_input_1 = self.relu(self.dropout(self.linear(object_input)))
        img_trans_input = self.norm2(self.linear_out(img_trans_input_1))
        return img_trans_input



class text_network(nn.Module):   #把自动编码器与transformer在一起训练,(S,B,D)
    def __init__(self,word_dim,hidden_dim):
        super(text_network, self).__init__()
        self.cap_encoder = nn.Sequential(
            nn.LayerNorm(word_dim),
            nn.Linear(word_dim,word_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(word_dim,hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    def forward (self,word_input):
        x_encoder = self.cap_encoder(word_input)
        return x_encoder

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
        # print(i_postive)
        # print(i_all)
        loss = loss - torch.log(i_postive/i_all)
    return loss
class Pooler(nn.Module):
    def __init__(self,hidden_size):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size,hidden_size)
        self.activation = nn.Tanh()
    def forward(self,hidden_states):
        first_token_tensor = hidden_states[:,0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
class classifier_network5(nn.Module):
    def __init__(self, text_encoder,image_dim, cap_emd, hidden_dim):
        super(classifier_network5, self).__init__()
        # self.attention_trans_encoder = TransformerEncoderLayer_1(d_model=word_dim,nhead=4,dim_feedforward=300,dropout=0.1)

        self.image_net = image_network(image_dim, hidden_dim)
        self.text_net = text_network(cap_emd, hidden_dim)


        self.bert_model = BertModel.from_pretrained('bert-base-uncased', config=config, add_pooling_layer=False)
        img_mod = nn.TransformerEncoderLayer(hidden_dim,4,image_dim,0.1)
        self.img_model = nn.TransformerEncoder(img_mod,8)

        # self.multi_encoder = BertModel.from_pretrained('bert-base-uncased', config=config, add_pooling_layer=False)
        self.pooler = Pooler(hidden_dim)
        self.token_type_embeddings = nn.Embedding(3, hidden_dim)

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

        self.itm_head = nn.Linear(hidden_dim, 2)


    def forward(self, image_input, text_inputs, idx):
        len = image_input.shape[0]
        key_padding_mask = text_inputs.attention_mask
        idx = idx.view(-1, 1)  # 展开成n行1列   ,idx.t() ----> (1,n)
        pos_idx = torch.eq(idx, idx.t()).float()

        image_input = F.normalize(self.image_net(image_input))
        img_IMG = torch.ones(len,1,image_input.shape[2]).cuda()
        image_input = torch.cat((img_IMG,image_input),dim=1).transpose(0,1)
        image_input = self.img_model(image_input).transpose(0,1)
        img_fusion_1 = F.softmax(self.image_mlp(image_input[ :, 1:, : ]))
        img_fusion_2 = self.image_layer(image_input[ :, 1:, : ])
        img_fusion = torch.bmm(img_fusion_1.permute(0, 2, 1), img_fusion_2).squeeze(1)
        img_out = img_fusion + image_input[ :, 0, : ]

        cap_input = self.bert_model(text_inputs.input_ids,attention_mask=key_padding_mask,return_dict=True,mode='text').last_hidden_state
        cap_input = F.normalize(self.text_net(cap_input))
        cap_fusion_1 = F.softmax(self.cap_mlp(cap_input[ :, 1:, : ]), dim=1)
        cap_fusion_2 = self.cap_layer(cap_input[ :, 1:, : ])
        cap_fusion = torch.bmm(cap_fusion_1.permute(0, 2, 1), cap_fusion_2).squeeze(1)
        cap_out = cap_fusion + cap_input[ :, 0, : ]

        score_bf = torch.zeros(len, len)
        for i in range(len):
            i_cap_out = cap_out[ i ].reshape(1, -1)
            score_bf[ i ] = F.cosine_similarity(i_cap_out, img_out)
        score_bf = score_bf.cuda()
        loss_bf = Info_NCE(0.05, score_bf, pos_idx) + Info_NCE(0.05, score_bf.t(), pos_idx.t())


         # get a (img,txt)negative pair match
        img_embeds_neg = [ ]
        text_embeds_neg = [ ]
        score_bf = torch.clamp(score_bf, 0, 1)
        for i in range(len):
            for j in range(len):
                if (int(pos_idx[ i ][ j ]) == 1):
                    score_bf[ i ][ j ] = -1
        max_img_inx = torch.max(score_bf, dim=1)[ 1 ]
        max_txt_inx = torch.max(score_bf, dim=0)[ 1 ]
        text_neg_mask = [ ]
        for i in range(len):
            img_embeds_neg.append(image_input[ max_img_inx[ i ] ])
            text_embeds_neg.append(cap_input[ max_txt_inx[ i ] ])
            text_neg_mask.append(key_padding_mask[ max_txt_inx[ i ] ])

        text_neg_mask = torch.stack(text_neg_mask, dim=0)
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        img_embeds_neg = torch.stack(img_embeds_neg, dim=0)  # 4, 37, 768

        # double stream
        #postive
        image_atts = torch.ones(image_input.size()[ :-1 ], dtype=torch.long).cuda()
        img_txt_pos_out = self.bert_model(encoder_embeds=cap_input,
                                          attention_mask=key_padding_mask,
                                          encoder_hidden_states=image_input,
                                          encoder_attention_mask=image_atts,
                                          return_dict=True,
                                          mode='fusion'
                                          )
        img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]
        img_txt_pos_cls_cap,_ = (img_txt_pos_out.last_hidden_state[:,:cap_input.shape[1]],img_txt_pos_out.last_hidden_state[:,cap_input.shape[1]:])

        #negative
        text_embeds_neg_all = torch.cat([ cap_input, text_embeds_neg ], dim=0)
        img_embeds_neg_all = torch.cat([ img_embeds_neg, image_input ], dim=0)
        text_attention_neg_all = torch.cat([ key_padding_mask, text_neg_mask ], dim=0)
        img_attention_neg_all = torch.cat([ image_atts, image_atts ], dim=0)
        img_txt_neg_out_1 = self.bert_model(encoder_embeds=text_embeds_neg_all,
                                            attention_mask=text_attention_neg_all,
                                            encoder_hidden_states=img_embeds_neg_all,
                                            encoder_attention_mask=img_attention_neg_all,
                                            return_dict=True,
                                            mode='fusion'
                                            )
        img_txt_neg_cls_out_1 = img_txt_neg_out_1.last_hidden_state[ :, 0, : ]
        img_txt_neg_cls_cap,_ = (img_txt_neg_out_1.last_hidden_state[:,:text_embeds_neg_all.shape[1]],img_txt_neg_out_1.last_hidden_state[:,text_embeds_neg_all.shape[1]:])
        img_txt_cls = torch.cat((img_txt_pos_cls_out, img_txt_neg_cls_out_1), dim=0)
        img_txt_score = self.itm_head(img_txt_cls)
        img_txt_label = torch.cat([ torch.ones(len, dtype=torch.long), torch.zeros(2 * len, dtype=torch.long) ],
                                  dim=0).cuda()

        #singal stream
        #postive
        text_embeds, image_embeds = (
            cap_input + self.token_type_embeddings(torch.zeros_like(key_padding_mask)),
            image_input
            + self.token_type_embeddings(
                torch.full_like(image_atts, 1)
            ),
        )
        co_inp = torch.cat([ text_embeds, image_embeds ], dim=1)
        co_masks = torch.cat([key_padding_mask,image_atts],dim=1)
        co_out_all = self.bert_model(encoder_embeds=co_inp,
                                attention_mask=co_masks,
                                encoder_hidden_states=co_inp,
                                encoder_attention_mask=co_masks,
                                return_dict=True,
                                mode='fusion')
        # co_out = self.bert_model(co_inp,src_key_padding_mask=co_masks)
        co_out = co_out_all.last_hidden_state
        _,co_img = (co_out[:,:text_embeds.shape[1]],co_out[:,text_embeds.shape[1]:])
        cls_feats = self.pooler(co_out)
        #negative
        text_neg_embeds, image_neg_embeds = (
            text_embeds_neg_all + self.token_type_embeddings(torch.zeros_like(text_attention_neg_all)),
            img_embeds_neg_all
            + self.token_type_embeddings(
                torch.full_like(img_attention_neg_all, 1)
            ),
        )
        co_neg_inp = torch.cat([text_neg_embeds,image_neg_embeds],dim=1)
        co_neg_mask = torch.cat([text_attention_neg_all,img_attention_neg_all],dim=1)
        # co_neg_out = self.encoder(co_neg_inp,src_key_padding_mask=co_neg_mask).transpose(0,1)
        co_neg_out_all = self.bert_model(encoder_embeds=co_neg_inp,
                                attention_mask=co_neg_mask,
                                encoder_hidden_states=co_neg_inp,
                                encoder_attention_mask=co_neg_mask,
                                return_dict=True,
                                mode='fusion')
        co_neg_out = co_neg_out_all.last_hidden_state
        _,co_neg_img = (co_neg_out[:,:text_neg_embeds.shape[1]],co_neg_out[:,text_neg_embeds.shape[1]:])
        cls_neg_feats = self.pooler(co_neg_out)
        co_neg_all_cls = torch.cat((cls_feats, cls_neg_feats), dim=0)
        co_all_score = self.itm_head(co_neg_all_cls)
        co_all_label = torch.cat([ torch.ones(len, dtype=torch.long), torch.zeros(2 * len, dtype=torch.long) ],
                                  dim=0).cuda()

        #co stream
        #postive
        img_txt_co_pos_out = self.bert_model(encoder_embeds=img_txt_pos_cls_cap,
                                          attention_mask=key_padding_mask,
                                          encoder_hidden_states=co_img,
                                          encoder_attention_mask=image_atts,
                                          return_dict=True,
                                          mode='fusion'
                                          )
        img_txt_co_pos_cls_out = img_txt_co_pos_out.last_hidden_state[ :, 0, : ]
        #negative
        img_txt_co_neg_out = self.bert_model(encoder_embeds=img_txt_neg_cls_cap,
                                          attention_mask=text_attention_neg_all,
                                          encoder_hidden_states=co_neg_img,
                                          encoder_attention_mask=img_attention_neg_all,
                                          return_dict=True,
                                          mode='fusion'
                                          )
        img_txt_co_pos_neg_out = img_txt_co_neg_out.last_hidden_state[ :, 0, : ]
        co_neg_cls = torch.cat((img_txt_co_pos_cls_out, img_txt_co_pos_neg_out), dim=0)
        co_score = self.itm_head(co_neg_cls)
        co_label = torch.cat([ torch.ones(len, dtype=torch.long), torch.zeros(2 * len, dtype=torch.long) ],
                                  dim=0).cuda()


        loss_itm = F.cross_entropy(img_txt_score, img_txt_label) + F.cross_entropy(co_all_score,co_all_label) + F.cross_entropy(co_score,co_label)
        loss_num = loss_itm + loss_bf
        print("loss_bf:",loss_bf)
        print("loss_itm:",loss_itm)
        print("loss_num", loss_num)
        return loss_num

    def forward_testic(self, image_input, text_inputs,key_padding_mask,token_type_ids):
        len_i = image_input.shape[0]
        len_c = text_inputs.shape[0]
        image_input = F.normalize(self.image_net(image_input))
        img_IMG = torch.ones(len_i,1,image_input.shape[2]).cuda()
        image_input = torch.cat((img_IMG,image_input),dim=1).transpose(0,1)
        image_input = self.img_model(image_input).transpose(0,1)
        cap_input = self.bert_model(text_inputs,attention_mask=key_padding_mask,return_dict=True,mode='text').last_hidden_state
        cap_input = F.normalize(self.text_net(cap_input))

        img_txt_score = torch.zeros(len_i,len_c)
        for i in range(len_i):
            img_inp = image_input[i].repeat(len_c,1,1)
            image_atts = torch.ones(img_inp.size()[ :-1 ], dtype=torch.long).cuda()
            # get the postive pair match
            img_txt_pos_out = self.bert_model(encoder_embeds=cap_input,
                                              attention_mask=key_padding_mask,
                                              encoder_hidden_states=img_inp,
                                              encoder_attention_mask=image_atts,
                                              return_dict=True,
                                              mode='fusion'
                                              )
            img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]

            img_txt_score[i] = self.itm_head(img_txt_pos_cls_out)[ :, 1 ]
        # print("img_txt_score:",img_txt_score)

        return img_txt_score

    def forward_testci(self, image_input, text_inputs,key_padding_mask,token_type_ids):
        len_i = image_input.shape[0]
        len_c = text_inputs.shape[0]
        image_input = F.normalize(self.image_net(image_input))
        img_IMG = torch.ones(len_i,1,image_input.shape[2]).cuda()
        image_input = torch.cat((img_IMG,image_input),dim=1).transpose(0,1)
        image_input = self.img_model(image_input).transpose(0,1)
        cap_input = self.bert_model(text_inputs,attention_mask=key_padding_mask,return_dict=True,mode='text').last_hidden_state
        cap_input = F.normalize(self.text_net(cap_input))

        img_txt_score = torch.zeros(len_i,len_c)
        for i in range(len_c):
            cap_inp = cap_input[i].repeat(len_i,1,1)
            key_padding_mask_inp = key_padding_mask[i].repeat(len_i,1)
            image_atts = torch.ones(image_input.size()[ :-1 ], dtype=torch.long).cuda()
            # get the postive pair match
            img_txt_pos_out = self.bert_model(encoder_embeds=cap_inp,
                                              attention_mask=key_padding_mask_inp,
                                              encoder_hidden_states=image_input,
                                              encoder_attention_mask=image_atts,
                                              return_dict=True,
                                              mode='fusion'
                                              )
            img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]

            img_txt_score[i] = self.itm_head(img_txt_pos_cls_out)[ :, 1 ]
        # print("img_txt_score:",img_txt_score)

        return img_txt_score














