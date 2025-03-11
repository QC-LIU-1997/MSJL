# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timm
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from collections import Counter,OrderedDict
import gc
from typing import List, Optional, Callable
import math
from timm.models.layers import  create_classifier
from timm.models import BasicBlock, ResNet


# %%
class IAMGEIterator(Dataset):
    def __init__(self,im_list,score_list,aug_flag):
        self.im_list = im_list
        self.score_list = score_list
        self.aug_flag = aug_flag

    def __read_image(self,idx):
        raw_im_path = '..//Temp_IQA//F_image//{}'.format(self.im_list[idx])
        raw_im = cv2.imread(raw_im_path,0)

        cropped_im = raw_im

        if self.aug_flag == 'B':
            img_transforms = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomRotation(15,expand=True),transforms.RandomPerspective(distortion_scale=0.3,p=0.5),
                                                 transforms.Resize((224,224)),transforms.Normalize((0.5), (0.5))])
            fin_im = img_transforms(cropped_im)
        if self.aug_flag == 'D':
            img_transforms = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize((0.5), (0.5))])
            fin_im = img_transforms(cropped_im)

        return fin_im

    def __read_label(self,idx):
        score = self.score_list[idx]
        fin_score_float = torch.tensor([score],dtype = torch.float32)
        return fin_score_float
    
    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, idx):
        fin_im = self.__read_image(idx)
        fin_score_float = self.__read_label(idx)
        data = {
            'fin_im': fin_im,
            'fin_score_float':fin_score_float
        }
        return data

# %%
def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    """

    Modified from https://github.com/cfzd/FcaNet.

    """
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """

    Modified from https://github.com/cfzd/FcaNet.

    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
                        
        return dct_filter

# %%
class ResNetMFM(nn.Module):
    def __init__(
            self,
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            conv_encoder_stride = 32
    ):
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        self.conv_encoder_stride = conv_encoder_stride
        super(ResNetMFM, self).__init__()
        self.resnet_encoder = ResNet(block = self.block,layers = self.layers, in_chans = self.in_chans, num_classes = self.num_classes)
        self.resnet_decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.resnet_encoder.num_features,
                out_channels=self.conv_encoder_stride ** 2, kernel_size=1),
            nn.PixelShuffle(self.conv_encoder_stride),
        )
    def forward(self, x):
        x = self.resnet_encoder.forward_features(x)
        x = self.resnet_decoder(x)
        return x


# %%
class ResNet_Tri(nn.Module):
    def __init__(
            self,
            block=BasicBlock,
            layers=[2,2,2,2],
            num_classes=1000,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            cardinality=1,
            base_width=64,
            stem_width=64,
            stem_type='',
            replace_stem_pool=False,
            block_reduce_first=1,
            down_kernel_size=1,
            avg_down=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            aa_layer=None,
            drop_rate=0.0,
            drop_path_rate=0.,
            drop_block_rate=0.,
            zero_init_last=True,
            block_args=None,
            fuse_flag = 'add',
    ):
        self.block = block
        self.layers = layers
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate
        super(ResNet_Tri, self).__init__()
        self.resnet_1 = ResNet(block = self.block,layers = self.layers, in_chans = self.in_chans, num_classes = self.num_classes)
        self.resnet_2 = ResNet(block = self.block,layers = self.layers, in_chans = self.in_chans, num_classes = self.num_classes)
        self.resnet_3 = ResNet(block = self.block,layers = self.layers, in_chans = self.in_chans, num_classes = self.num_classes)
        self.fuse_flag = fuse_flag  
        if self.fuse_flag == 'concat':
          self.total_num_features = self.resnet_1.num_features+self.resnet_2.num_features+self.resnet_3.num_features
          _, self.total_classifier = create_classifier(self.total_num_features, self.num_classes, pool_type='avg')
        if self.fuse_flag == 'att':
          """ Proposed FAF """
          self.raw_channels = 512
          self.embed_dim = 64
          self.fin_dim = 8
          self.total_num_features = self.resnet_2.num_features
          self.get_token = nn.Sequential(nn.Linear(self.raw_channels,self.embed_dim),nn.GELU(),nn.Linear(self.embed_dim,self.fin_dim),nn.GELU())
          mapper_x, mapper_y = get_freq_indices('top16')
          self.num_split = len(mapper_x)
          mapper_x = [temp_x for temp_x in mapper_x] 
          mapper_y = [temp_y for temp_y in mapper_y]

          self.dct_layer_1 = MultiSpectralDCTLayer(7, 7, mapper_x, mapper_y, 512)
          self.dct_layer_2 = MultiSpectralDCTLayer(7, 7, mapper_x, mapper_y, 512)
          self.dct_layer_3 = MultiSpectralDCTLayer(7, 7, mapper_x, mapper_y, 512)
          

          self.soft_fuse_pool = nn.AdaptiveAvgPool1d(1)
          self.soft_fuse_softmax = nn.Softmax(dim = 1)
          _, self.total_classifier = create_classifier(self.fin_dim, self.num_classes, pool_type='avg')

        
    def forward(self, x):
        x_1 = self.resnet_1.forward_features(x)
        x_2 = self.resnet_2.forward_features(x)
        x_3 = self.resnet_3.forward_features(x)
        

        if self.fuse_flag == 'concat':
          x_1 = self.resnet_1.global_pool(x_1)
          x_2 = self.resnet_2.global_pool(x_2)
          x_3 = self.resnet_3.global_pool(x_3)

          x_total = torch.concat([x_1,x_2,x_3],dim = 1)
          output_total = self.total_classifier(x_total)
        if self.fuse_flag == 'att':
          x_1 = self.get_token(self.dct_layer_1(x_1)).unsqueeze(1)
          x_2 = self.get_token(self.dct_layer_2(x_2)).unsqueeze(1)
          x_3 = self.get_token(self.dct_layer_3(x_3)).unsqueeze(1)
          x_total = torch.concat([x_1,x_2,x_3],dim = 1)
          x_fuse_pool = self.soft_fuse_pool(x_total)
          x_fuse_weights = self.soft_fuse_softmax(x_fuse_pool)*3
          x_total = x_total*x_fuse_weights

          output_total = self.total_classifier(x_total.mean(dim=1))

        return output_total

# %%
def create_model(model_str):
    mask_flag = model_str[13:-2]
    temp_model = ResNet_Tri(num_classes=1,in_chans=1,fuse_flag='att')
    low_pass_model = ResNet(block = BasicBlock,layers = [2,2,2,2], num_classes=1,in_chans=1)
    low_pass_pretrained_dict=torch.load('low_pass.pth').state_dict()
    low_pass_model_dict=low_pass_model.state_dict()
    low_pass_pretrained_dict = {k[15:]: v for k, v in low_pass_pretrained_dict.items() if k[15:] in low_pass_model_dict}
    low_pass_model_dict.update(low_pass_pretrained_dict)
    low_pass_model.load_state_dict(low_pass_model_dict)

    high_pass_model = ResNet(block = BasicBlock,layers = [2,2,2,2], num_classes=1,in_chans=1)
    high_pass_pretrained_dict=torch.load('high_pass.pth').state_dict()
    high_pass_model_dict=high_pass_model.state_dict()
    high_pass_pretrained_dict = {k[15:]: v for k, v in high_pass_pretrained_dict.items() if k[15:] in high_pass_model_dict}
    high_pass_model_dict.update(high_pass_pretrained_dict)
    high_pass_model.load_state_dict(high_pass_model_dict)

    resnet_pretrain = timm.create_model('resnet18',num_classes=1,pretrained=True,in_chans=1)     
    temp_model.resnet_1.load_state_dict(low_pass_model.state_dict())
    temp_model.resnet_2.load_state_dict(resnet_pretrain.state_dict())
    temp_model.resnet_3.load_state_dict(high_pass_model.state_dict())


    return temp_model

# %%
def train_score_model(model,aug_num,train_dl,test_dl_aug,test_dl_normal,epochs,opt,scheduler):
    criterion = nn.MSELoss()
    history = []
    for epoch in range(epochs):
        model.train()
        train_stats = []
        for train_dl_i in train_dl:
            batch_im = train_dl_i['fin_im']
            target = train_dl_i['fin_score_float']
            output =  model(batch_im)
            output=output.to(torch.float32)
            target=target.to(torch.float32)
            MSE_loss = criterion(output,target)
            opt.zero_grad()
            MSE_loss.backward()
            opt.step()
            train_stats.append(MSE_loss.item())
        history.append(np.mean(train_stats))
        scheduler.step()
        print('Temp Epoch: {}, Temp Loss: {}'.format(epoch+1,np.mean(train_stats)))
    return model, history


