# %%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import numpy as np
import copy
import decimal
import pandas as pd
import matplotlib.pyplot as plt
import timm
from skimage import io
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import Counter
from sklearn.manifold import TSNE
import scipy.stats
import gc
from typing import List, Optional, Callable
import math
from torch.utils.checkpoint import checkpoint
from timm.models.layers import DropPath, trunc_normal_
from timm.models import BasicBlock,Bottleneck,ResNet

# %%
def mask_torch_soft(x_freq,mask_r,mask_flag):
    mask = torch.zeros((x_freq.shape[0],x_freq.shape[-2],x_freq.shape[-1]))
    if mask_flag == True:
        for i in range(mask.shape[-2]):
            for j in range(mask.shape[-1]):
                distance_2 = (i-mask.shape[-2]/2)**2+(j-mask.shape[-1]/2)**2
                mask[:,i,j]=np.e ** (-1 * (distance_2  / (2 * mask_r ** 2)))

                    
    if mask_flag == False:
        for i in range(mask.shape[-2]):
            for j in range(mask.shape[-1]):
                distance_2 = (i-mask.shape[-2]/2)**2+(j-mask.shape[-1]/2)**2
                mask[:,i,j]=1-np.e ** (-1 * (distance_2  / (2 * mask_r ** 2)))
    return mask

# %%
class IAMGEIterator(Dataset):
    def __init__(self,im_list,mask_r = 10):
        self.im_list = im_list
        self.mask_r = mask_r

    def __read_image(self,idx):
        raw_im_path = '..//Temp_IQA//P_image//{}'.format(self.im_list[idx])
        raw_im = io.imread(raw_im_path)
        cropped_im = raw_im
        weak_aug_img_transforms = transforms.Compose([transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),transforms.RandomResizedCrop((224,224),scale=(0.75,1.0))
        ,transforms.Resize((224,224)),transforms.Normalize((0.5), (0.5))])

        fin_im = weak_aug_img_transforms(cropped_im)
        x_freq = torch.fft.fft2(fin_im,norm = 'ortho')
        # shift low frequency to the center
        x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))

        fin_mask_lower = mask_torch_soft(x_freq,self.mask_r,False)
        fin_mask_higher = mask_torch_soft(x_freq,self.mask_r,True)
                 
        x_freq_masked_unshift_lower = x_freq * fin_mask_lower
        x_freq_masked_lower = torch.fft.ifftshift(x_freq_masked_unshift_lower, dim=(-2, -1))
        fin_corrupted_lower = torch.fft.ifft2(x_freq_masked_lower,norm = 'ortho').real


        x_freq_masked_unshift_higher = x_freq * fin_mask_higher
        x_freq_masked_higher = torch.fft.ifftshift(x_freq_masked_unshift_higher, dim=(-2, -1))
        fin_corrupted_higher = torch.fft.ifft2(x_freq_masked_higher,norm = 'ortho').real


        return fin_im,fin_corrupted_lower,fin_mask_lower,fin_corrupted_higher,fin_mask_higher


    def __len__(self):
        return len(self.im_list)
    
    def __getitem__(self, index):
        fin_im,fin_corrupted_lower,fin_mask_lower,fin_corrupted_higher,fin_mask_higher = self.__read_image(index)
        data = {
            'fin_im': fin_im,
            'fin_corrupted_lower': fin_corrupted_lower,
            'fin_mask_lower': fin_mask_lower,
            'fin_corrupted_higher': fin_corrupted_higher,
            'fin_mask_higher': fin_mask_higher,
        }
        return data

# %%
class FrequencyLoss(nn.Module):
    """Frequency loss.

    Modified from https://github.com/EndlessSora/focal-frequency-loss.

    """

    def __init__(self,
                 loss_gamma=1.,
                 matrix_gamma=1.,
                 patch_factor=1,
                 ave_spectrum=False,
                 with_matrix=False,
                 log_matrix=False,
                 batch_matrix=False):
        super(FrequencyLoss, self).__init__()
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
    def tensor2freq(self, x):

        freq = torch.fft.fft2(x, norm='ortho')
        # shift low frequency to the center
        freq = torch.fft.fftshift(freq, dim=(-2, -1))
        # stack the real and imaginary parts along the last dimension
        freq = torch.stack([freq.real, freq.imag], -1)
        return freq

    def loss_formulation(self, recon_freq, real_freq):

        # frequency distance using (squared) Euclidean distance
        tmp = (recon_freq - real_freq) ** 2
        loss = torch.sqrt(tmp[..., 0] + tmp[..., 1] + 1e-12) ** self.loss_gamma

        return loss

    def forward(self, pred, target, mask):
        """Forward function to calculate frequency loss.

        Args:
            pred (torch.Tensor): Predicted tensor with shape (N, C, H, W).
            target (torch.Tensor): Target tensor with shape (N, C, H, W).
            matrix (torch.Tensor, optional): Element-wise spectrum weight matrix.
                Defaults to None.
        """

        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        mask = torch.tensor(mask,dtype = torch.float32)
        fin_loss = self.loss_formulation(pred_freq, target_freq)


        # calculate frequency loss
        return fin_loss.mean()

# %%
class ResNetSMFM(nn.Module):
    """SMFM.

    Modified from https://github.com/Jiahao000/MFM.

    """
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
        super(ResNetSMFM, self).__init__()
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
def train_MFM_model(model,train_dl,epochs,opt,mask_flag,scheduler,resume_flag = False,resume_path = None):

    criterion = FrequencyLoss()
    history = []
    # best_loss = 99999
    # best_epoch = 0
    # best_model_params = copy.deepcopy(model.state_dict())
    if resume_flag ==  False:
        init_epoch = 0
    else:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['parameter']) 
        opt.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        init_epoch = checkpoint['epoch']+1

    for epoch in range(init_epoch,epochs):
        
        model.train()
        train_stats = []
        epoch_pass_flag = np.random.randint(2)
        for train_dl_i in train_dl:
            target = train_dl_i['fin_im']

            if mask_flag == 'high_pass':
                batch_im = train_dl_i['fin_corrupted_lower']
                mask= train_dl_i['fin_mask_lower']
                 
                output =  model(batch_im)
                output=output.to(torch.float32)
                target=target.to(torch.float32)
                freq_loss = criterion(output,target,mask)
                opt.zero_grad()
                freq_loss.backward()
                opt.step()

            if mask_flag == 'low_pass':
                batch_im = train_dl_i['fin_corrupted_higher']
                mask= train_dl_i['fin_mask_higher']
            
                output =  model(batch_im)
                output=output.to(torch.float32)
                target=target.to(torch.float32)
                freq_loss = criterion(output,target,mask)
                opt.zero_grad()
                freq_loss.backward()
                opt.step()
            
            train_stats.append(freq_loss.item())


        history.append([np.mean(train_stats)])
        scheduler.step()
        checkpoint = {'parameter': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch}
        if resume_flag ==  True:
            torch.save(checkpoint, resume_path)
        print('Temp Epoch: {}, Temp Loss: {}'.format(epoch+1,np.mean(train_stats)))
    return model, history


