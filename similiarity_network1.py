import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from xbert_change1 import BertConfig, BertModel, BertForMaskedLM

config = BertConfig.from_json_file('/home/10501002/BERT/CNTR/src/config_bert.json')


class image_network(nn.Module):  # 把transformer与跨模态transformer结合在一起训练
    def __init__(self, image_dim, hidden_dim):
        super(image_network, self).__init__()
        self.norm1 = nn.LayerNorm(image_dim)
        self.linear = nn.Linear(image_dim, hidden_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        o_n = nn.TransformerEncoderLayer(hidden_dim, nhead=8, dim_feedforward=image_dim, dropout=0.1)
        self.transformer = nn.TransformerEncoder(o_n, num_layers=4)
        self.linear_out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, object_input):
        object_input = self.norm1(object_input)
        img_trans_input = self.relu(self.dropout(self.linear(object_input)))
        img_trans_input = self.linear_out(img_trans_input).transpose(0, 1)
        img_trans_out = self.norm2(self.transformer(img_trans_input).transpose(0, 1))
        return img_trans_out


class text_network(nn.Module):  # 把自动编码器与transformer在一起训练,(S,B,D)
    def __init__(self, word_dim, hidden_dim):
        super(text_network, self).__init__()
        self.cap_encoder = nn.Sequential(
            nn.LayerNorm(word_dim),
            nn.Linear(word_dim, word_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(word_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, word_input):
        x_encoder = self.cap_encoder(word_input)
        return x_encoder


def Info_NCE(temp, similiary_martix, postive_idx, ):
    sim_len = similiary_martix.shape[ 0 ]
    loss = 0
    for i in range(sim_len):
        i_postive = 0
        i_all = 0
        for j in range(sim_len):
            if (int(postive_idx[ i ][ j ]) == 1):
                i_postive = i_postive + torch.exp(similiary_martix[ i ][ j ] / temp)
            i_all = i_all + torch.exp(similiary_martix[ i ][ j ] / temp)
        # print(i_postive)
        # print(i_all)
        loss = loss - torch.log(i_postive / i_all)
    return loss


class classifier_network5(nn.Module):
    def __init__(self, tokenizer, image_dim, cap_emd, hidden_dim):
        super(classifier_network5, self).__init__()
        # self.attention_trans_encoder = TransformerEncoderLayer_1(d_model=word_dim,nhead=4,dim_feedforward=300,dropout=0.1)
        self.tokenizer = tokenizer
        self.image_net = image_network(image_dim, hidden_dim)
        self.text_net = text_network(cap_emd, hidden_dim)

        # self.text_encoder = text_encoder

        # self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased', config=config)

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

    def forward(self, image_input, text_inputs, idx, img_delta, txt_delta):
        len = image_input.shape[ 0 ]
        key_padding_mask = text_inputs.attention_mask
        IMG = torch.ones(len, 1, image_input.shape[ 2 ]).cuda()

        img_inputs_noisy = image_input + img_delta
        img_inputs_noisy = torch.cat((IMG, img_inputs_noisy), dim=1)  # B,N_I+[IMG],dim
        print("img_inputs_noisy", img_inputs_noisy.shape)

        image_input = torch.cat((IMG, image_input), dim=1)  # B,N_I+[IMG],dim
        image_input = F.normalize(self.image_net(image_input))
        img_fusion_1 = F.softmax(self.image_mlp(image_input[ :, 1:, : ]))
        img_fusion_2 = self.image_layer(image_input[ :, 1:, : ])
        img_fusion = torch.bmm(img_fusion_1.permute(0, 2, 1), img_fusion_2).squeeze(1)
        img_out = img_fusion + image_input[ :, 0, : ]

        # cap_input = self.text_encoder(text_inputs.input_ids,key_padding_mask,text_inputs.token_type_ids)[ 'last_hidden_state' ]
        cap_input = self.bert_model.bert(text_inputs.input_ids, attention_mask=key_padding_mask, return_dict=True,
                                         mode='text').last_hidden_state

        txt_inputs_noisy = cap_input + txt_delta
        print("txt_inputs_noisy", txt_inputs_noisy.shape)

        cap_input = F.normalize(self.text_net(cap_input))
        cap_fusion_1 = F.softmax(self.cap_mlp(cap_input[ :, 1:, : ]), dim=1)
        cap_fusion_2 = self.cap_layer(cap_input[ :, 1:, : ])
        cap_fusion = torch.bmm(cap_fusion_1.permute(0, 2, 1), cap_fusion_2).squeeze(1)
        cap_out = cap_fusion + cap_input[ :, 0, : ]

        idx = idx.view(-1, 1)  # 展开成n行1列   ,idx.t() ----> (1,n)
        pos_idx = torch.eq(idx, idx.t()).float()

        img_inputs_noisy = F.normalize(self.image_net(img_inputs_noisy))
        txt_inputs_noisy = F.softmax(self.text_net(txt_inputs_noisy))

        score_bf = torch.zeros(len, len)
        for i in range(len):
            i_cap_out = cap_out[ i ].reshape(1, -1)
            score_bf[ i ] = F.cosine_similarity(i_cap_out, img_out)
        score_bf = score_bf.cuda()

        loss_bf = Info_NCE(0.05, score_bf, pos_idx) + Info_NCE(0.05, score_bf.t(), pos_idx.t())

        # Noisy data add
        # image noisy
        image_atts = torch.ones(image_input.size()[ :-1 ], dtype=torch.long).cuda()
        img_txt_pos_img_noisy = self.bert_model.bert(encoder_embeds=cap_input,
                                                     attention_mask=key_padding_mask,
                                                     encoder_hidden_states=img_inputs_noisy,
                                                     encoder_attention_mask=image_atts,
                                                     return_dict=True,
                                                     mode='fusion'
                                                     )
        img_txt_pos_img_noisy_out = img_txt_pos_img_noisy.last_hidden_state[ :, 0, : ]
        # Text noisy
        img_txt_pos_txt_noisy = self.bert_model.bert(encoder_embeds=txt_inputs_noisy,
                                                     attention_mask=key_padding_mask,
                                                     encoder_hidden_states=image_input,
                                                     encoder_attention_mask=image_atts,
                                                     return_dict=True,
                                                     mode='fusion'
                                                     )
        img_txt_pos_txt_noisy_out = img_txt_pos_txt_noisy.last_hidden_state[ :, 0, : ]

        # get the postive pair match
        img_txt_pos_out = self.bert_model.bert(encoder_embeds=cap_input,
                                               attention_mask=key_padding_mask,
                                               encoder_hidden_states=image_input,
                                               encoder_attention_mask=image_atts,
                                               return_dict=True,
                                               mode='fusion'
                                               )
        img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]

        img_txt_pos_cls_out_all = torch.cat((img_txt_pos_cls_out, img_txt_pos_img_noisy_out, img_txt_pos_txt_noisy_out),
                                            dim=0)

        # compute kl_div
        # fine-grained noramlize term
        img_txt_pos_cls_out_prob = F.softmax(img_txt_pos_cls_out, dim=1)
        img_txt_pos_cls_out_logprob = F.log_softmax(img_txt_pos_cls_out, dim=1)
        img_txt_pos_img_noisy_out_prob = F.softmax(img_txt_pos_img_noisy_out, dim=1)
        img_txt_pos_img_noisy_out_logprob = F.log_softmax(img_txt_pos_img_noisy_out, dim=1)
        img_txt_pos_txt_noisy_out_prob = F.softmax(img_txt_pos_txt_noisy_out, dim=1)
        img_txt_pos_txt_noisy_out_logprob = F.log_softmax(img_txt_pos_txt_noisy_out, dim=1)
        img_kl_loss = F.kl_div(img_txt_pos_cls_out_logprob, img_txt_pos_img_noisy_out_prob) + \
                      F.kl_div(img_txt_pos_img_noisy_out_logprob, img_txt_pos_cls_out_prob) + \
                      F.kl_div(img_txt_pos_cls_out_logprob, img_txt_pos_txt_noisy_out_prob) + \
                      F.kl_div(img_txt_pos_txt_noisy_out_logprob, img_txt_pos_cls_out_prob)

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
        text_embeds_neg_all = torch.cat([ cap_input, text_embeds_neg ], dim=0)
        # img_embeds_neg = torch.stack(img_embeds_neg, dim=0)  # 4, 37, 768
        img_embeds_neg = torch.stack(img_embeds_neg, dim=0)  # 4, 37, 768
        img_embeds_neg_all = torch.cat([ img_embeds_neg, image_input ], dim=0)

        text_attention_neg_all = torch.cat([ key_padding_mask, text_neg_mask ], dim=0)
        img_attention_neg_all = torch.cat([ image_atts, image_atts ], dim=0)

        img_txt_neg_out_1 = self.bert_model.bert(encoder_embeds=text_embeds_neg_all,
                                                 attention_mask=text_attention_neg_all,
                                                 encoder_hidden_states=img_embeds_neg_all,
                                                 encoder_attention_mask=img_attention_neg_all,
                                                 return_dict=True,
                                                 mode='fusion'
                                                 )
        img_txt_neg_cls_out_1 = img_txt_neg_out_1.last_hidden_state[ :, 0, : ]

        img_txt_cls = torch.cat((img_txt_pos_cls_out_all, img_txt_neg_cls_out_1), dim=0)
        img_txt_score = self.itm_head(img_txt_cls)
        img_txt_label = torch.cat([ torch.ones(3 * len, dtype=torch.long), torch.zeros(2 * len, dtype=torch.long) ],
                                  dim=0).cuda()

        loss_itm = F.cross_entropy(img_txt_score, img_txt_label)

        ### MlM(mask language modeling)
        input_ids = text_inputs.input_ids.clone()
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, 0.15)

        input_ids, labels = self.mask(input_ids, self.bert_model.config.vocab_size, image_input.device, targets=labels,
                                      probability_matrix=probability_matrix)
        mlm_output = self.bert_model(input_ids,
                                     attention_mask=key_padding_mask,
                                     encoder_hidden_states=image_input,
                                     encoder_attention_mask=image_atts,
                                     return_dict=True,
                                     labels=labels,
                                     alpha=0
                                     )
        loss_mlm = mlm_output.loss

        print("loss_itm", loss_itm)
        print("loss_bf:", loss_bf)
        print("img_kl_loss:", img_kl_loss)
        print("loss_mlm", loss_mlm)
        loss_num = loss_itm + 0.1 * loss_bf + img_kl_loss + loss_mlm
        print("loss_num", loss_num)
        return loss_num

    def forward_testic(self, image_input, text_inputs, key_padding_mask, mode):
        len_i = image_input.shape[ 0 ]
        len_c = text_inputs.shape[ 0 ]
        IMG = torch.ones(len_i, 1, image_input.shape[ 2 ]).cuda()
        image_input = torch.cat((IMG, image_input), dim=1)  # B,N_I+[IMG],dim

        image_input = F.normalize(self.image_net(image_input))
        cap_input = self.bert_model.bert(text_inputs, attention_mask=key_padding_mask, return_dict=True,
                                         mode='text').last_hidden_state
        cap_input = F.normalize(self.text_net(cap_input))

        if mode == "cons":
            img_fusion_1 = F.softmax(self.image_mlp(image_input[ :, 1:, : ]))
            img_fusion_2 = self.image_layer(image_input[ :, 1:, : ])
            img_fusion = torch.bmm(img_fusion_1.permute(0, 2, 1), img_fusion_2).squeeze(1)
            img_out = img_fusion + image_input[ :, 0, : ]
            cap_fusion_1 = F.softmax(self.cap_mlp(cap_input[ :, 1:, : ]), dim=1)
            cap_fusion_2 = self.cap_layer(cap_input[ :, 1:, : ])
            cap_fusion = torch.bmm(cap_fusion_1.permute(0, 2, 1), cap_fusion_2).squeeze(1)
            cap_out = cap_fusion + cap_input[ :, 0, : ]
            score_bf = torch.zeros(len_i, len_c)
            for i in range(len_i):
                i_cap_out = cap_out[ i ].reshape(1, -1)
                score_bf[ i ] = F.cosine_similarity(i_cap_out, img_out)
            score_bf = score_bf.cuda()
            return score_bf.t()
        if mode == "fusion":
            img_inp = image_input
            image_atts = torch.ones(img_inp.size()[ :-1 ], dtype=torch.long).cuda()
            # get the postive pair match
            img_txt_pos_out = self.bert_model.bert(encoder_embeds=cap_input,
                                                   attention_mask=key_padding_mask,
                                                   encoder_hidden_states=img_inp,
                                                   encoder_attention_mask=image_atts,
                                                   return_dict=True,
                                                   mode='fusion'
                                                   )
            img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]

            img_txt_score = self.itm_head(img_txt_pos_cls_out)[ :, 1 ]
            # print("img_txt_score:",img_txt_score)
            return img_txt_score

    # def forward_testic(self, image_input, text_inputs, key_padding_mask, token_type_ids):
    #     len_i = image_input.shape[ 0 ]
    #     len_c = text_inputs.shape[ 0 ]
    #     IMG = torch.ones(len_i, 1, image_input.shape[ 2 ]).cuda()
    #     image_input = torch.cat((IMG, image_input), dim=1)  # B,N_I+[IMG],dim
    #
    #     image_input = F.normalize(self.image_net(image_input))
    #     # cap_input = self.text_encoder(text_inputs, key_padding_mask, token_type_ids)[ 'last_hidden_state' ]
    #     cap_input = self.bert_model.bert(text_inputs, attention_mask=key_padding_mask, return_dict=True,
    #                                      mode='text').last_hidden_state
    #     cap_input = F.normalize(self.text_net(cap_input))
    #
    #     img_txt_score = torch.zeros(len_i, len_c)
    #     for i in range(len_i):
    #         img_inp = image_input[ i ].repeat(len_c, 1, 1)
    #         image_atts = torch.ones(img_inp.size()[ :-1 ], dtype=torch.long).cuda()
    #         # get the postive pair match
    #         img_txt_pos_out = self.bert_model.bert(encoder_embeds=cap_input,
    #                                                attention_mask=key_padding_mask,
    #                                                encoder_hidden_states=img_inp,
    #                                                encoder_attention_mask=image_atts,
    #                                                return_dict=True,
    #                                                mode='fusion'
    #                                                )
    #         img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]
    #
    #         img_txt_score[ i ] = self.itm_head(img_txt_pos_cls_out)[ :, 1 ]
    #     # print("img_txt_score:",img_txt_score)
    #
    #     return img_txt_score
    #
    # def forward_testci(self, image_input, text_inputs, key_padding_mask, token_type_ids):
    #     len_i = image_input.shape[ 0 ]
    #     len_c = text_inputs.shape[ 0 ]
    #     IMG = torch.ones(len_i, 1, image_input.shape[ 2 ]).cuda()
    #     image_input = torch.cat((IMG, image_input), dim=1)  # B,N_I+[IMG],dim
    #     image_input = F.normalize(self.image_net(image_input))
    #     # cap_input = self.text_encoder(text_inputs, key_padding_mask, token_type_ids)[ 'last_hidden_state' ]
    #     cap_input = self.bert_model.bert(text_inputs, attention_mask=key_padding_mask, return_dict=True,
    #                                      mode='text').last_hidden_state
    #     cap_input = F.normalize(self.text_net(cap_input))
    #
    #     img_txt_score = torch.zeros(len_i, len_c)
    #     for i in range(len_c):
    #         cap_inp = cap_input[ i ].repeat(len_i, 1, 1)
    #         key_padding_mask_inp = key_padding_mask[ i ].repeat(len_i, 1)
    #         image_atts = torch.ones(image_input.size()[ :-1 ], dtype=torch.long).cuda()
    #         # get the postive pair match
    #         img_txt_pos_out = self.bert_model.bert(encoder_embeds=cap_inp,
    #                                                attention_mask=key_padding_mask_inp,
    #                                                encoder_hidden_states=image_input,
    #                                                encoder_attention_mask=image_atts,
    #                                                return_dict=True,
    #                                                mode='fusion'
    #                                                )
    #         img_txt_pos_cls_out = img_txt_pos_out.last_hidden_state[ :, 0, : ]
    #
    #         img_txt_score[ i ] = self.itm_head(img_txt_pos_cls_out)[ :, 1 ]
    #     # print("img_txt_score:",img_txt_score)
    #
    #     return img_txt_score

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[ input_ids == self.tokenizer.pad_token_id ] = False
        masked_indices[ input_ids == self.tokenizer.cls_token_id ] = False

        if targets is not None:
            targets[ ~masked_indices ] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[ indices_replaced ] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[ indices_random ] = random_words[ indices_random ]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids












