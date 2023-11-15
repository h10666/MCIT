import argparse
import datetime
import json
import sys
import time
import os

from timm.scheduler import CosineLRScheduler

from dataloader import train_dataset,test_dataset
from transformers import BertTokenizer,BertModel
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from similiarity_network2_rs import classifier_network5
import ruamel_yaml as yaml

log_dir = './train_log/train_log_dir'
writer = SummaryWriter(log_dir=log_dir, comment='_scalars', filename_suffix='12345678')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def match_paircoco_i2c5K(scores):
    #5000,25000
    npts = scores.shape[ 0 ]  # image_shape
    print("npts",npts)
    ranks = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(scores[ index ])[ ::-1 ]
        # Score
        rank = 1e20
        print("index:",index)
        for i in range(5*index, 5*index + 5, 1):
            tmp = np.where(inds == i)[ 0 ][ 0 ]
            # 最大相似度的索引值
            if tmp < rank:
                rank = tmp
        print("rank // 5",rank // 5)
        ranks[ index ] = rank // 5
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[ 0 ]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[ 0 ]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[ 0 ]) / len(ranks)
    # medr = np.floor(np.median(ranks)) + 1
    # meanr = ranks.mean() + 1
    return r1, r5, r10
def match_paircoco_c2i5K(scores):
    #25000,5000
    npts = scores.shape[ 0 ]  # image_shape
    ranks = np.zeros(npts)

    for index in range(0,npts,1):
        inds = np.argsort(scores[ index ])[ ::-1 ]
        # Score
        rank = 1e20
        index_c = index // 5
        tmp = np.where(inds == index_c)[ 0 ][ 0 ]
        # 最大相似度的索引值
        if tmp < rank:
            rank = tmp
        ranks[ index ] = rank
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[ 0 ]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[ 0 ]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[ 0 ]) / len(ranks)
    # medr = np.floor(np.median(ranks)) + 1
    # meanr = ranks.mean() + 1
    return r1, r5, r10

def train(epoch,model, data_loader, optimizer, scheduler,device, scaler_batch):
    # train
    train_epoch_loss = 0
    step_size = 100
    warmup_iterations = 1*step_size
    for batch_idx, (image, caption, idx) in enumerate(data_loader):
        scaler_batch = scaler_batch + 1
        print('Train batch_idx: [{}]'.format(batch_idx))
        model.train(True)
        image = image.to(device, non_blocking=True)
        # caption features
        text_input_all = tokenizer(caption,return_tensors="pt", padding=True, truncation=True)  #
        text_input_all = text_input_all.to(device)
        idx = idx.to(device, non_blocking=True)

        img_delta = torch.zeros_like(image)
        txt_delta = torch.zeros(text_input_all.input_ids.shape[0],text_input_all.input_ids.shape[1],768).to(device)
        print("img_delta:",img_delta.shape)
        print("txt_delta:",txt_delta.shape)

        for astep in range(10):
            # only add the noisy on one modality
            img_delta.float().requires_grad_()
            txt_delta.float().requires_grad_()

        loss_num = model.forward(image, text_input_all,idx,img_delta,txt_delta)
        print("loss_num:", loss_num)

        optimizer.zero_grad()
        loss_num.backward()
        optimizer.step()
        #epoch == 0 and 
        if epoch == 0 and batch_idx % step_size == 0:
            scheduler.step(batch_idx // step_size)

        train_epoch_loss = train_epoch_loss + float(loss_num)
        writer.add_scalar("train_loss_num", loss_num, scaler_batch)

    train_epoch_loss = train_epoch_loss / len(data_loader.dataset)
    return train_epoch_loss, scaler_batch

@torch.no_grad()
def evaluation(model, num_image,num_text,text_inputs ,text_padding_mask,text_token_type_ids,image_inputs,device):
    # test
    with torch.no_grad():
        model.eval()
        print('Computing features for evaluation...')
        start_time = time.time()

        score_matrix_i2t = torch.full((num_image, num_text), -100.0).to(device)
        score_matrix_t2i = torch.full((num_text, num_image), -100.0).to(device)
        shard_size = 200
        n_im_shard = num_image // shard_size
        print("n_im_shard:",n_im_shard)
        n_cap_shard = num_text // shard_size
        print("n_cap_shard:",n_cap_shard)
        for i in range(int(n_im_shard)):
            img_start, img_end = shard_size * i, min(shard_size * (i + 1), num_image)
            im = image_inputs[ img_start:img_end ].to(device)
            for j in range(int(n_cap_shard)):
                sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
                #text
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), num_text)
                s = text_inputs[cap_start:cap_end].to(device)      #(N,S)
                key_padding_mask = text_padding_mask[cap_start:cap_end].to(device)   #(N,S)
                token_type_ids = text_token_type_ids[cap_start:cap_end].to(device)
                #image
                # repeat_num = cap_end - cap_start
                scores = model.forward_testic(im, s, key_padding_mask,token_type_ids)
                score_matrix_i2t[img_start:img_end, cap_start:cap_end ] = scores
        sys.stdout.write('\n')

        for i in range(int(n_cap_shard)):
            cap_start, cap_end = shard_size * i, min(shard_size * (i + 1), num_text)
            s = text_inputs[ cap_start:cap_end ].to(device)  # (N,S)
            key_padding_mask = text_padding_mask[ cap_start:cap_end ].to(device)  # (N,S)
            token_type_ids = text_token_type_ids[ cap_start:cap_end ].to(device)
            print("key_padding_mask:", key_padding_mask.shape)
            for j in range(int(n_im_shard)):
                sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
                #image
                img_start, img_end = shard_size * j, min(shard_size * (j + 1), num_image)
                im = image_inputs[img_start:img_end].to(device)
                #image
                scores = model.forward_testci(im, s, key_padding_mask,token_type_ids)
                score_matrix_t2i[ cap_start:cap_end , img_start:img_end ] = scores
        sys.stdout.write('\n')

        # del image_inputs
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i):

    tr1,tr5,tr10 = match_paircoco_c2i5K(scores_t2i)
    ir1,ir5,ir10 = match_paircoco_i2c5K(scores_i2t)
    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr1,
                   'txt_r5': tr5,
                   'txt_r10': tr10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir1,
                   'img_r5': ir5,
                   'img_r10': ir10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result

def main(opt,config):
    device = torch.device(opt.device)
    #load train_Feature and val features
    train_features = np.load(config['train_features_path'],allow_pickle=True).item()
    valid_features = np.load(config['valid_features_path'],allow_pickle=True).item()

    #load train caption and val caption
    train_captions = [ ]
    train_image_ids = [ ]
    train_caption_json = json.load(open(config['train_caption_path'], 'r'))
    for i_json in train_caption_json:
        train_captions.append(i_json[ 'caption' ])
        train_image_ids.append(i_json[ 'image_id' ])
    print("train_captions:", len(train_captions))
    print("train_image_ids", len(train_image_ids))

    valid_captions = [ ]
    valid_image_ids = [ ]
    valid_caption_json = json.load(open(config['valid_caption_path'], 'r'))
    for i_json in valid_caption_json:
        for i_cap in i_json[ 'caption' ]:
            valid_captions.append(i_cap)
        valid_image_ids.append(i_json[ 'image_id' ])
    print("valid_captions:", len(valid_captions))
    print("valid_image_ids", len(valid_image_ids))

    train_data_dataset = train_dataset(train_features,train_captions,train_image_ids)
    valid_data_dataset = test_dataset(valid_features,valid_image_ids)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_data_dataset,
                                            batch_size=config['batch_size'],
                                            shuffle=True,
                                            pin_memory=True,
                                            num_workers=0)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_data_dataset,
                                            batch_size=config['val_batch'],
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=0)
    # text_encoder = BertModel.from_pretrained('bert-base-uncased')
    model = classifier_network5(tokenizer,config['image_dim'],config['cap_dim'],config['hidden_dim'])
    model = model.to(device)
    model_path = config['model_path']
    if model_path != '':
        msg = model.load_state_dict(torch.load(model_path))
    #前30epoch：lr:1e-4->lr_min:1e-5
    #后20epoch：lr:1e-5->lr_min:1e-6
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)  # Adam优化器
    scheduler = CosineLRScheduler(optimizer, t_initial=5, lr_min=config['lr_min'],
                                  decay_rate=1, warmup_lr_init=1e-5, warmup_t=1,
                                  t_in_epochs=True, )
    num_text = len(valid_captions)//5
    print("num_text", num_text)
    num_image = len(valid_dataloader.dataset.image)//5
    print("num_image", num_image)

    # get the image_inputs
    image_inputs = [ ]
    for image, img_id in valid_dataloader:
        image_inputs.append(image)
    image_inputs = torch.cat(image_inputs, dim=0)[:1000]
    print("image_inputs:", (image_inputs.shape))

    # get the text_inputs
    text_bs = 256
    text_inputs = [ ]
    text_padding_mask = [ ]
    text_token_type_ids = [ ]
    for i in range(0, num_text, text_bs):
        text = valid_captions[ i: min(num_text, i + text_bs) ]
        text_feats_data = tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt")
        text_feats_data = text_feats_data
        text_feats = text_feats_data[ 'input_ids' ]
        key_padding_mask = text_feats_data[ 'attention_mask' ]
        text_token_type_ids.append(text_feats_data[ 'token_type_ids' ])
        text_padding_mask.append(key_padding_mask)
        text_inputs.append(text_feats)  # ([32, 21, 768])
    text_inputs = torch.cat(text_inputs, dim=0)[:5000]
    text_padding_mask = torch.cat(text_padding_mask, dim=0)[:5000]
    text_token_type_ids = torch.cat(text_token_type_ids, dim=0)[:5000]
    print("text_inputs:", (text_inputs.shape))
    print("text_padding_mask:", text_padding_mask.shape)
    print("text_token_type_ids", text_token_type_ids.shape)

    scaler_batch_num = 0
    for epoch_flag in range(config['epoch']):
        scaler_batch = scaler_batch_num
        print('Train Epoch: [{}]'.format(epoch_flag))
        train_epoch_loss, scaler_batch_num = train(epoch_flag,model, train_dataloader, optimizer,scheduler, device, scaler_batch)
        writer.add_scalar("train_epoch_num", train_epoch_loss, epoch_flag)
        torch.save(model.state_dict(), "fin_model_coco.pkl")

        if (epoch_flag % 2 == 0):
            score_val_i2t, score_val_t2i = evaluation(model, num_image,num_text,text_inputs ,text_padding_mask,text_token_type_ids,image_inputs,device)
            print("score_val_i2t", score_val_i2t)
            print("score_val_t2i", score_val_t2i)
            val_result = itm_eval(score_val_i2t, score_val_t2i)
            print('val_result', val_result)
            writer.add_scalar("txt_r1", val_result[ 'txt_r1' ], epoch_flag)
            writer.add_scalar("txt_r5", val_result[ 'txt_r5' ], epoch_flag)
            writer.add_scalar("txt_r10", val_result[ 'txt_r10' ], epoch_flag)
            writer.add_scalar("img_r1", val_result[ 'img_r1' ], epoch_flag)
            writer.add_scalar("img_r5", val_result[ 'img_r5' ], epoch_flag)
            writer.add_scalar("img_r10", val_result[ 'img_r10' ], epoch_flag)

        torch.cuda.empty_cache()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--train_features_path', default="/home/10501002/10501009/COCO_dict/image_train_features.npy",
    #                     help='path to train_features')
    # parser.add_argument('--valid_features_path', default='/home/10501002/10501009/COCO_dict/image_val_features.npy',
    #                     help='path to valid_features')
    # parser.add_argument('--train_caption_path', default="/home/10501002/10501009/COCO_dict/train_captions.json",
    #                     help='path to train_caption')
    # parser.add_argument('--valid_caption_path', default="/home/10501002/10501009/COCO_dict/valid_captions.json")
    # parser.add_argument('--epoch', default=30, type=int,
    #                     help='Size of a training mini-batch.')
    # parser.add_argument('--batch_size', default=36, type=int,
    #                     help='Size of a training mini-batch.')
    # parser.add_argument('--val_batch', default=500, type=int,
    #                     help='Size of a training mini-batch.')
    # parser.add_argument('--image_dim', default=2048, type=int,
    #                     help='Dimensionality of the word embedding.')
    # parser.add_argument('--cap_dim', default=768, type=int,
    #                     help='Dimensionality of the word embedding.')
    # parser.add_argument('--hidden_dim', default=768, type=int,
    #                     help='Dimensionality of the word embedding.')
    parser.add_argument('--config', default='./configs/train_coco.yaml')
    parser.add_argument('--device', default='cuda')

    opt = parser.parse_args()

    config = yaml.load(open(opt.config, 'r'), Loader=yaml.Loader)

    # Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # yaml.dump(config, open(os.path.join(opt.output_dir, 'config.yaml'), 'w'))

    main(opt,config)
