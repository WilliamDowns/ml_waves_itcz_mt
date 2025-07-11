'''
Code related to constructing, training, and evaluating networks.
'''

import numpy as np
import xarray as xr
import torch
import torch.nn as tnn
import torch.nn.functional as F
from torch import optim
import xbatcher
import xbatcher.loaders.torch
import wandb
import os
import logging
import warnings
import scipy.ndimage
import re
from joblib import Parallel, delayed

# Note this may be imported to other places
#os.environ["WANDB_MODE"] = "offline"

##############################################
# Defs of UNet components
##############################################


class ArbConv(tnn.Module):
    '''
    Convolution -> BN (Optional) -> ReLU, as many times as desired
    '''
    def __init__(self, in_channels, out_channels, conv_count=3, bn=True, kernel_size=3, padding=1,
                 bias=False):
        '''
        in_channels: int, number of input channels to first conv (input variables)
        out_channels: int, number of output channels from second conv / batch norm (number of output features to capture)
        conv_count: int, number of times to repeat the operation
        bn: bool, whether to do the two batch normalization layers
        kernel_size: int, side length of kernel to use in convolution
        padding: int, padding (in pixels) applied to each side of input
        bias: bool, whether to also calculate a learned bias of input
        '''
        super().__init__()
        self.conv = []#tnn.Sequential()
        for i in range(conv_count):
            self.conv.append(tnn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias))
            if bn:
                self.conv.append(tnn.BatchNorm2d(out_channels))
            self.conv.append(tnn.ReLU(inplace=True))
            in_channels=out_channels
        self.conv = tnn.Sequential(*self.conv)
    def forward(self, x):
        # take input data x and run our convolution block
        return self.conv(x)


class DoubleConv(tnn.Module):
    '''
    Convolution -> BN (Optional) -> ReLU, twice
    '''
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=True, kernel_size=3, padding=1,
                 bias=False):
        '''
        in_channels: int, number of input channels to first conv (input variables)
        out_channels: int, number of output channels from second conv / batch norm (number of output features to capture)
        mid_channels: int or None, number of output/input channels for conv -> batch normalization step.
                      Uses out_channels if mid_channels==None
        bn: bool, whether to do the two batch normalization layers
        kernel_size: int, side length of kernel to use in convolution
        padding: int, padding (in pixels) applied to each side of input
        bias: bool, whether to also calculate a learned bias of input
        '''
        super().__init__()
        if mid_channels == None:
            mid_channels = out_channels
        if bn:
            # do batch norms
            self.double_conv = tnn.Sequential(
                tnn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.BatchNorm2d(mid_channels),
                tnn.ReLU(inplace=True),                
                tnn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.BatchNorm2d(out_channels),
                tnn.ReLU(inplace=True))                

        else:
            # no batch norms
            self.double_conv = tnn.Sequential(
                tnn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.ReLU(inplace=True),
                tnn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.ReLU(inplace=True))
        
    def forward(self, x):
        # take input data x and run our double convolution block
        return self.double_conv(x)
        
class SingleConv(tnn.Module):
    '''
    Convolution -> BN (Optional) -> ReLU, once
    '''
    def __init__(self, in_channels, out_channels,  bn=True, kernel_size=3, padding=1,
                 bias=False):
        '''
        in_channels: int, number of input channels to first conv (input variables)
        out_channels: int, number of output channels from second conv / batch norm (number of output features to capture)
        bn: bool, whether to do the two batch normalization layers
        kernel_size: int, side length of kernel to use in convolution
        padding: int, padding (in pixels) applied to each side of input
        bias: bool, whether to also calculate a learned bias of input
        '''
        super().__init__()
        if bn:
            # do batch norms
            self.single_conv = tnn.Sequential(
                tnn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.BatchNorm2d(out_channels),
                tnn.ReLU(inplace=True),)

        else:
            # no batch norms
            self.single_conv = tnn.Sequential(
                tnn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
                tnn.ReLU(inplace=True),)
        
    def forward(self, x):
        # take input data x and run our double convolution block
        return self.single_conv(x)

class ArbDown(tnn.Module):
    '''
    Downscaling with dropoutfrac and a pooling function followed by some number of Convs
    '''
    def __init__(self, in_channels, out_channels, conv=1,
                 bn=True, kernel_size=3, padding=1, bias=False,
                 pool_func='avg', pool_len=2, dropout_frac=0.25):
        '''
        in_channels: int, number of input channels to first conv (input variables)
        out_channels: int, number of output channels from second conv / batch norm (number of output features to capture)
        #vars solely for Conv
        conv: int, number of convolutional layers to include
        bn: bool, whether to do the two batch normalization layers
        kernel_size: int, side length of kernel to use in convolution
        padding: int, padding (in pixels) applied to each side of input
        #vars solely for pool/dropout
        pool_func: str, type of pooling function to use, valid values: 'avg', 'max'
        pool_len: int, sidelength of pooling function
        dropout_frac: float, fraction for 2D dropout. valid values: 0 <= x < 1. If 0, no dropout performed
        '''
        super().__init__()
        pool_func=tnn.MaxPool2d if pool_func=='max' else tnn.AvgPool2d
        self.pool_conv = []
        if dropout_frac > 0:
            self.pool_conv.append(tnn.Dropout2d(dropout_frac))
        self.pool_conv.append(pool_func(pool_len))
        self.pool_conv.append(ArbConv(in_channels, out_channels, conv_count=conv, bn=bn, kernel_size=kernel_size,
                               padding=padding, bias=bias))
        self.pool_conv = tnn.Sequential(*self.pool_conv)
    def forward(self, x):
        # take input data x and run our pooling/dropout/conv
        return self.pool_conv(x)
    

class OutConv(tnn.Module):
    # Final (possible) output layer
    def __init__(self, in_channels, out_channels, padding=0, kernel_size=1):
        super(OutConv, self).__init__()
        self.conv = tnn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size)

    def forward(self, x):
        return self.conv(x)
    

class UpNoPadding(tnn.Module):
    '''
    Upsampling with not built in padding, followed by a Conv. Does not include the input from an old down layer
    '''    
    def __init__(self, in_channels, out_channels, bilinear=False,
                 bn=True, kernel_size=3, padding=1, bias=False, scale_factor=2,conv=1):
        super().__init__()
        self.scale_factor=scale_factor
        # by default, channel reduction / upsampling occurs via a transpose convolution step
        # if we just want to do bilinear interpolation as our upsampling and use normal convolutions
        # to reduce our channel count, set bilinear=True
        if bilinear:
            # basic unlearned upsampling
            self.up = tnn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            if conv==1:
                self.conv = SingleConv(in_channels, out_channels, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias)
            else:
                self.conv = DoubleConv(in_channels, out_channels, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias)                
        else:
            # TODO: this way learns weights (theoretically "smarter" upsampling)
            # TODO: should kernel_size=scale_factor here?
            self.up = tnn.ConvTranspose2d(in_channels, in_channels //2, kernel_size=2, stride=scale_factor)
            if conv == 1:
                self.conv = SingleConv(in_channels, out_channels, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias)
            else:
                self.conv = DoubleConv(in_channels, out_channels, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias)                
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class Fusion(tnn.Module):
    '''
    Concatenate all input tensors along channel dim, then execute a Conv on them.
    '''
    def __init__(self, in_channels, out_channels, bilinear=False,
                 bn=True, kernel_size=3, padding=1, bias=False, conv=1):

        super().__init__()
        self.conv = ArbConv(in_channels, out_channels, conv_count=conv, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias)
    def forward(self, tensors):
        # tensors is a list of all tensors to concatenate
        for i in range(1,len(tensors)):
            # Padding stuff
            x2 = tensors[i-1]
            x1 = tensors[i]
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            tensors[i] = x1
        tensors = torch.cat(tensors, dim=1)
        return self.conv(tensors)

class PadAndUpsampleList(tnn.Module):
    '''
    Pads and upsamples all tensors to have the same x/y size
    '''
    def __init__(self):
        super().__init__()
        self.up = tnn.Upsample(scale_factor=2, mode='bilinear')
    def forward(self, tensors):
        # tensors is a list of all tensors that may need padding
        # Assume tensors are in increasing order of resolution
        ## first, order tensors by their size
        ##tensors = tensors[np.argsort([t.size[2]*t.size[3] for t in tensors])]
        # pad and upsample all tensors iteratively
        for i, t1 in enumerate(tensors):
            
            for j, t2 in enumerate(tensors[i:]):
                diffY = t2.size()[2] - t1.size()[2]
                diffX = t2.size()[3] - t1.size()[3]
                t1 = F.pad(t1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
            tensors[i] = t1
        return tensors
    

class UNet_3Plus_no_first_skip(tnn.Module):
    '''
    UNet 3+ architecture adapted from Justin et al. 2023 
    
    n_channels: number of initial input channels
    n_classes: number of output classifications
    bilinear: whether to do bilinear upsampling or use a transpose convolution
    initial_target_channels: number of channels to output from first 
    n_tiers: number of vertical layers of network (i.e. number of pooling or upsampling layers, - 1)
    # conv args
    bn: bool, whether to do the two batch normalization layers        
    kernel_size: int, side length of kernel to use in convolution
    padding: int, padding (in pixels) applied to each side of input
    # downsampling args
    pool_func: str, type of pooling function to use, valid values: 'avg', 'max'
    pool_len: int, sidelength of pooling function
    dropout_frac: float, fraction for 2D dropout. valid values: 0 <= x < 1. If 0, no dropout performed
    conv: int, whether to do single (1) or double (2) convolution layers
    deep_sup: bool, whether to include deep supervision (side classification output from each layer)
    init_full_skip: bool, whether to include output of input convolution in full scale skip connections
    
    '''
    def __init__(self, n_channels, n_classes, bilinear=True, initial_target_channels=64, n_tiers=4,
                 bn=True, kernel_size=3, padding=1, bias=False, pool_func='avg', pool_len=2, dropout_frac=0.25,
                 conv=1, deep_sup=False, init_full_skip=True):
    
        super(UNet_3Plus_no_first_skip, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.initial_target_channels = initial_target_channels
        self.n_tiers = n_tiers
        self.bn = bn
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.pool_func = pool_func
        self.pool_len = pool_len
        self.dropout_frac = dropout_frac
        self.deep_sup = deep_sup
        self.init_full_skip = init_full_skip
        
        self.inc = (ArbConv(n_channels, initial_target_channels, conv_count=conv, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias))

        # create down layers (doubling number of channels at each layer)
        self.downs = tnn.ModuleList([(ArbDown(in_chan, out_chan, conv=conv, kernel_size=kernel_size, padding=padding, bias=bias,
                                              pool_func=pool_func, pool_len=pool_len, dropout_frac=dropout_frac))
                                     for in_chan, out_chan in zip([initial_target_channels*2**i for i in range(n_tiers-1)],
                                                                  [initial_target_channels*2**i for i in range(1, n_tiers)])])

        ### The super complicated up section ###
        self.CatChannels = self.initial_target_channels
        self.CatBlocks = self.n_tiers
        if not init_full_skip: self.CatBlocks-=1
        self.UpChannels = self.CatChannels * self.CatBlocks

        ## Each up layer receives input from its standard preceding up layer, plus input from every down layer (appropriately pooled)
        # There should be n_tiers input tensors of size input_target_channels that makeup each up layer
        self.up_lists = tnn.ModuleList()
        min_full_skip = -1 if init_full_skip else 0
        for i in range(self.n_tiers-2, -1, -1):
            up_list = []
            # downsampling layers first (pool these as needed)
            for j in range(min_full_skip+1, i, 1):
                up_list.append(ArbDown(initial_target_channels*2**j, self.initial_target_channels, conv=1, bn=bn, kernel_size=kernel_size, padding=padding,
                                       bias=bias, pool_func=pool_func, pool_len=2**(i-j), dropout_frac=0))
            # then a concatenation layer (the standard UNet skip connection)
            if i != 0 or init_full_skip:
                up_list.append(ArbConv(initial_target_channels*2**i, self.initial_target_channels, conv_count=1, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias))
                
            # then preceding upsampling layers (upsample these as needed)
            for j in range(i+1, self.n_tiers):
                # all but the deepest of these layers will have n_tiers*initial_target_channels as their number of channels
                if j == self.n_tiers-1: 
                    n_channels = initial_target_channels*2**(self.n_tiers-1)
                else:
                    n_channels = self.UpChannels
                up_list.append(UpNoPadding(n_channels, self.initial_target_channels, bn=bn, kernel_size=kernel_size, padding=padding,
                                           bilinear=bilinear, bias=bias, scale_factor=2**(j-i), conv=1))#conv=conv))
                
            # then a fusion layer where all the different inputs are combined and convolved
            up_list.append(Fusion(self.UpChannels, self.UpChannels, bn=bn, kernel_size=kernel_size, padding=padding, bias=bias, conv=conv))
            # append this set of n_tiers + 1 operations

            self.up_lists.extend(up_list)

        if not deep_sup:
            self.outc = (OutConv(self.UpChannels, n_classes, padding=padding, kernel_size=kernel_size))
        else:
            # one output from each layer, all usampled to original res
            self.paul = PadAndUpsampleList()
            self.outcs = tnn.ModuleList([OutConv(self.UpChannels, n_classes, padding=padding, kernel_size=kernel_size) for i in range(n_tiers-1)])
            self.outcs.append(OutConv(initial_target_channels*2**(self.n_tiers-1), n_classes, padding=padding, kernel_size=kernel_size))
        
    def forward(self, x):
        # complete forward operation structure of the net

        # first run initial convolution + downsampling layers
        init = self.inc(x)
        down_outs = [init]
        for func in self.downs:
            down_outs.append(func(down_outs[-1]))

        if not self.init_full_skip:
            tier_count=self.n_tiers-1
            down_start=1
        else:
            tier_count=self.n_tiers
            down_start=0
        # run complicated decoder layers
        # each tier has a certain number of ops
        for i in range(int(len(self.up_lists)/self.n_tiers)):
            # each tier of model has one input contributing here
            decoder_inputs = [f(d) for f, d in zip(self.up_lists[i*tier_count+i:i*tier_count+i+tier_count], down_outs[down_start:])]
            # replace a down_out with a new combined input layer from the bottom up
            down_outs[self.n_tiers-2-i] = self.up_lists[i*tier_count+i+tier_count](decoder_inputs)
        if not self.deep_sup:
            logits = [self.outc(down_outs[0])]
        else:
            # one output per tier of graph
            down_outs = self.paul(down_outs[::-1])[::-1]
            logits = [self.outcs[i](down_outs[i]) for i in range(self.n_tiers)]
        return logits
    
    def use_checkpointing(self):
        #setup operations for training
        self.inc = torch.utils.checkpoint(self.inc)
        self.downs = tnn.ModuleList([torch.utils.checkpoint(down) for down in self.downs])
        self.ups = tnn.ModuleList([torch.utils.checkpoint(up) for up in self.ups])
        self.outc = torch.utils.checkpoint(self.outc)
        
##############################################
# Methods for architecture-agnostic operations
##############################################

class xbatchTorchDataset(xbatcher.loaders.torch.MapDataset):
    # Made this stupid thing because xbatcher's Dataset doesn't transform to tensor correctly in getitem, also need a weight param
    def __init__(self, X_generator, y_generator, weight_generator=None, 
                 transform=None, target_transform=None, norm_means={}, norm_stds={}, channels_last=False, return_times=False):
        self.X_generator = X_generator
        self.y_generator = y_generator
        self.weight_generator = weight_generator
        self.hres_generator = hres_generator
        self.transform = transform
        self.target_transform = target_transform
        self.means = norm_means
        self.stds = norm_stds
        # testing this out to see if I can actually get it to work
        self.channels_last=channels_last
        self.return_times=return_times
    def __getitem__(self, idx):# -> Tuple[Any, Any]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
            if len(idx) == 1:
                idx = idx[0]
            else:
                raise NotImplementedError(
                    f"{type(self).__name__}.__getitem__ currently requires a single integer key"
                )

        # normalize if needed
        means = []
        stds = []
        if len(self.means.keys()) > 0 and len(self.stds.keys()) > 0:
            for var in list(self.X_generator[idx].data_vars):
                means.append(self.means[var])
                stds.append(self.stds[var])
        
            means = torch.as_tensor(np.array(means)[..., None, None])
            stds = torch.as_tensor(np.array(stds)[..., None, None])
            X_batch = (torch.as_tensor(self.X_generator[idx].to_array(dim='vars').data.astype('float32').squeeze())-means)/stds
        else:
            # this part is why I made this class; xbatcher's MapDataset doesn't accept more than one datavar for some reason
            X_batch = torch.as_tensor(self.X_generator[idx].to_array(dim='vars').data.astype('float32').squeeze())
            
        # y batch should be just labels (a dataarray)
        y_batch = torch.as_tensor(self.y_generator[idx].data.astype('float32').squeeze())
        if self.return_times:
            # need to convert times to ints to pass to tensor
            time = torch.as_tensor([int(''.join(re.split('[:-]', str(t))).replace('T', '')[:12]) for t in self.y_generator[idx]['time'].values])

        # NOTE: Have not implemented any transforms for weights or high res yet
        if self.weight_generator:
            weight_batch = torch.as_tensor(self.weight_generator[idx].data.astype('float32').squeeze())

        if self.hres_generator:
            hres_batch = torch.as_tensor(self.hres_generator[idx].data.astype('float32').squeeze())

        if self.channels_last:
            X_batch = X_batch.to(memory_format=torch.channels_last)
            Y_batch = Y_batch.to(memory_format=torch.channels_last)
        
        if self.transform:
            X_batch = self.transform(X_batch)            
        if self.target_transform:
            y_batch = self.target_transform(y_batch)

        if not self.weight_generator and not self.hres_generator:
            if not self.return_times:
                return X_batch, y_batch
            else:
                return X_batch, y_batch, time
        elif self.weight_generator and not self.hres_generator:
            return X_batch, y_batch, weight_batch
        elif not self.weight_generator and self.hres_generator:
            return X_batch, y_batch, hres_batch
        else:
            return X_batch, y_batch, weight_batch, hres_batch
            
        
def create_batches_from_xarray(data, batch_size=None, sample_dims={}, batch_dims={}, input_overlap={}, randomized_samples=True, num_workers=8,
                               cuda=True, label_var='label', channels_last=True, skip_zeros=True, skip_vals=[], sample_weight_var='', hres_var='',
                               hres_sample_dims={}, hres_batch_dims={}, concat_input_dims=True, norm_means={}, norm_stds={}, label_dims={}, return_times=False,
                               transform=None, target_transform=None):
    '''
    Divide data from an xarray Dataset into desired sample and batch sizes, then load them into a Pytorch Dataloader

    data: xr.Dataset, the full dataset to divide. Note that all data variables in data should share the same dimensions
    batch_size: int, size of batches to create. Note that if this does not divide evenly into the sample size the last batch will be thrown out
    sample_dims: dict, keys are names of dims in data and vals are the intended size of dimensions of each sample
    batch_dims: dict, keys are names of dims in data and vals are intended size of dimensions of each batch. Do not use alongside batch_size.
                Note that unfortunately this doesn't support random sampling
    input_overlap: dict, keys are names of dims in data and vals are intended amount each sample overlaps with next one. Not sure this is working
    randomized_samples: bool, whether each batch should be composed of a randomized set of samples from data or sequentially divided
    num_workers: int, number of workers to work on batching
    cuda: bool, whether to load the data into CUDA-pinned memory before returning them
    label_var: str, name of label dimension in data
    channels_last: bool, whether channels are in first dim and should be moved to last dim (can help speed up performance)
    skip_zeros: bool, whether to skip samples with no nonzero values
    skip_vals: list, which values in a sample force that sample to be skipped
    concat_input_dims: bool, whether to concatenate non-sample_dims into a single "sample" dimension
    sample_weight_var: str, name of optional sample weight variable in the data
    hres_var: str, name of optional sample weight variable in the data
    hres_sample_dims: dict, as for sample_dims but with different dimension names since hres will have different dims
    hres_batch_dims: dict, as for batch_dims but with different dimension names since hres will have different dims
    norm_means: dict, map of var names to mean across entire dataset. If empty, will not normalize this way
    norm_stds: dict, map of var names to std across entire dataset. If empty, will not normalize this way
    label_dims: dict, keys are names of dims for labels in data and vals are intended size of those dimensions. If empty, will use same as sample_dims.
                      Will also overwrite batch_dims for labels if this is not empty
    
    return : torch.datapipes object of the data arranged into batches
    '''

    print(input_overlap)
    if len(label_dims.keys()) > 0:
        label_batch_dims = label_dims
    else:
        label_dims = sample_dims
        label_batch_dims = batch_dims
    # divide into individual samples    
    input_samples = xbatcher.BatchGenerator(ds=data[[var for var in data.data_vars if var not in (label_var, sample_weight_var, hres_var)]],
                                            input_dims=sample_dims, batch_dims=batch_dims, concat_input_dims=concat_input_dims, input_overlap=input_overlap)
    label_samples = xbatcher.BatchGenerator(ds=data[label_var], input_dims=label_dims, batch_dims=label_batch_dims, concat_input_dims=concat_input_dims,
                                            input_overlap=input_overlap)
    
    if sample_weight_var != '':
        weight_samples = xbatcher.BatchGenerator(ds=data[sample_weight_var], input_dims=sample_dims, batch_dims=batch_dims, concat_input_dims=concat_input_dims,
                                                 input_overlap=input_overlap)
    else:
        weight_samples = None
    if hres_var != '':
        hres_samples = xbatcher.BatchGenerator(ds=data[hres_var], input_dims=hres_sample_dims, batch_dims=hres_batch_dims, concat_input_dims=concat_input_dims,
                                                 input_overlap=input_overlap)
    else:
        hres_samples = None
        
    # try not including samples at all where whole label is 0, or where a sample contains a skip value
    nonzero_indices = []
    print('initial sample count:', len(input_samples))
    if skip_zeros:
        for i in range(len(input_samples)):
            if i%500==0: print(i, 'skipping zeros, skip vals are', skip_vals, ', sample: ', label_samples[i])
            if not np.all(label_samples[i] == 0) and not np.any([np.any(label_samples[i]==val) for val in skip_vals]):                
                nonzero_indices.append(i)
    if skip_vals:
        for i in range(len(input_samples)):
            if i%500==0: print(i, 'not skipping zeros, skip vals are', skip_vals, ', sample: ', label_samples[i])
            if not np.any([np.any(label_samples[i]==val) for val in skip_vals]):
                nonzero_indices.append(i)
            else:
                print(label_samples[i])
    if len(skip_vals) > 0 or skip_zeros:
        input_samples = [input_samples[i] for i in nonzero_indices]
        label_samples = [label_samples[i] for i in nonzero_indices]
        if sample_weight_var != '':
            weight_samples = [weight_samples[i] for i in nonzero_indices]
        if hres_var != '':
            hres_samples = [hres_samples[i] for i in nonzero_indices]
            

    print('remaining samples', len(input_samples))
    # move channels to last dim to possibly speed things up
    def t_func(tens):
        return torch.moveaxis(tens, 0, 2)
    # this doesn't seem to work / be needed, so not doing it
    #if channels_last:
    if False: 
        if sample_weight_var == '':
            ds = xbatchTorchDataset(input_samples, label_samples, channels_last=channels_last)#transform=t_func)
        else:
            ds = xbatchTorchDataset(input_samples, label_samples, weight_generator=weight_samples, channels_last=channels_last) #transform=t_func)
    else:
        print('transform in use', transform)
        ds = xbatchTorchDataset(input_samples, label_samples, weight_generator=weight_samples, hres_generator=hres_samples,
                                norm_means=norm_means, norm_stds=norm_stds, return_times=return_times, transform=transform)
    
    return torch.utils.data.DataLoader(ds, shuffle=True, num_workers=num_workers,
                                       pin_memory=cuda, batch_size=batch_size, drop_last=True)






def train_model(model, data_train, data_val, device, epochs=5, batch_size=None, learning_rate=1e-4, save_checkpoint=True,
                amp=False, weight_decay=0, momentum=0.9, gradient_clipping=1.0, lr_reduce_factor=0.1, patience=10,
                scheduler=None, optimizer=None, loss_func=None, eval_interval=1, eval_func=None, checkpoint_dir='./checkpoints',
                channels_last=True, checkpoint_interval=1, sample_weight_var='', input_norm=False, input_norm_end=0, replace_nan=False,
                print_csi_metrics=False, csi_threshold=0.5, csi_distance=4, log=True, save_best=True, batcher=create_batches_from_xarray, **batch_args):
    '''
    Conduct training of a Neural Net
    
    model: tnn.Module subclass, a neural net class defined above
    data_train: xr.Dataset, data selected to use for training where all DataArrays share the same dims and one DataArray is the labels
    data_val: xr.Dataset, data selected to use for validation where all DataArrays share the same dims and one DataArray is the labels
    device: torch.device, which device to use for training ('cpu' or 'CUDA')
    epochs: int, number of epochs to run
    batch_size: int, size of input batches
    learning_rate: float, learning rate of model
    save_checkpoint: bool, whether to save model at end of each epoch
    amp: bool, whether to use mixed precision (uses less memory and is faster)
    weight_decay: float, penalty applied to loss function intended to reduce overfitting
    momentum: float, Nesterov momentum, helps accelerate convergence to minimum in stochastic gradient descent
    gradient_clipping: float, normalization factor used for avoiding exploding gradients
    lr_reduce_factor: float, factor to reduce learning rate by on plateau
    patience: int, number of eval_interval*epochs of plateaud learning rate to wait before decreasing it
    scheduler: torch.optim.lr_scheduler, the learning rate scheduler to use. Picks ReduceLROnPlateau if none given
    optimizer: torch.optim, the gradient optimization algorithm to use. Picks RMSprop if none given
    loss_func: various functions, the loss function to use. Must take (predict_array, target_array) as arguments. If None selected, use
               modified intersection over union from Niebler et al. 2022
    eval_interval: int, number of epochs to go through before each validation step OR how many batches to do eval after if negative (absolute value of negative number)
    eval_func: function, how to score evaluation phases. Uses loss_func if none specified
    channels_last: bool, whether channels are in first dim and should be moved to last dim (can help speed up performance)
    checkpoint_interval: int, how many epochs between checkpoints
    sample_weight_var: str, name of optional sample weight variable in the data
    input_norm: bool, whether to normalize input data in each layer to a mean of 0 and variance of 1 (commonly done, and done in batch norm layers later on)
    input_norm_end: int, position in arrays to end input normalization at. If 0, normalize all channels
    replace_nan: bool, whether to replace NaN values with 0 or not
    print_csi_metrics: bool, whether to print pod, sr, csi of validation dataset. Note this is configured only for UNet setup (map of classifications) right now
    csi_threshold: float, threshold to classify predicted probabilities as labels
    csi_distance: int, radius in grid points for determining a match for CSI calculations
    log: bool, whether to use wandb logging
    save_best: bool, whether to only save checkpoints that provide a new best eval score. Note this is configured for only lower = better loss
    batcher: func, what function to use to create batches (default create_batches_from_xarray)

    batch_args: keyword arguments to pass to batcher
    '''
    torch.set_num_threads(16)
    # Only enable this for debugging purposes; it will cause unwanted NaN flags otherwise (see https://discuss.pytorch.org/t/convolutionbackward0-returned-nan-values-in-its-0th-output/175571/3)
    #torch.autograd.set_detect_anomaly(True)
    
    # load up our data
    batch_args.update(dict(cuda=device=='CUDA', batch_size=batch_size))
    train_loader = batcher(data_train, channels_last=channels_last, sample_weight_var=sample_weight_var, **batch_args)
    val_loader = batcher(data_val, channels_last=channels_last, sample_weight_var=sample_weight_var,  **batch_args)
    
    # setup logging
    n_train = len(data_train)
    n_val   = len(data_val)

    if log:
        experiment = wandb.init(project='NN training', resume='allow', anonymous='must', mode='offline')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      save_checkpoint=save_checkpoint, amp=amp, n_train=n_train, n_val=n_val))
        logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
            Batch args: {batch_args}
        ''')


    if optimizer == None:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    # scheduler: how learning rate is adjusted as our epochs go
    if scheduler == None:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # automatic mixed precision (amp)-related object
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0

    min_eval_score = np.inf
    
    # Training time!
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        print('epoch', epoch)
        print('train_loader length', len(train_loader))
        for i, batch in enumerate(train_loader):
            data, labels = batch[0], batch[1]
            # load into memory
            if len(data.shape) == 3:
                data = data.unsqueeze(1)
            
            if channels_last:                
                data = data.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            else:
                data = data.to(device=device, dtype=torch.float32)

            # replace nans if neeeded
            if replace_nan:
                data = torch.nan_to_num(data,nan=0,posinf=0,neginf=0)
            # optionally normalize input data
            if input_norm:
                if input_norm_end == 0:
                    input_norm_end = data.shape[1]
                print(input_norm_end, data[:, :input_norm_end].shape)
                data[:, :input_norm_end] = F.normalize(data[:, :input_norm_end])
                
            labels = labels.to(device=device, dtype=torch.float32)
            if replace_nan:
                labels = torch.nan_to_num(labels,nan=0,posinf=0,neginf=0)
            
            if sample_weight_var != '':
                # load weights also. Didn't bother setting up channels last here yet
                sample_weights = batch[2].to(device=device, dtype=torch.float32)
            
            #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            with torch.cuda.amp.autocast(enabled=amp):
                labels_pred = model(data)
                if sample_weight_var == '':
                    loss = loss_func(labels_pred, labels)
                else:
                    loss = loss_func(labels_pred, labels, sample_weights)


            # block for printing weights, biases, and their gradients for each parameter in network
            '''
            for name, param in model.named_parameters():
                try:
                    print(name, float(param.grad.max()), float(param.grad.min()))
                except:
                    print(name, param)
            '''
            
            ## various optimizer steps to get our new gradient through backpropagation
            optimizer.zero_grad(set_to_none=True) # set gradients to 0 before doing backpropagation to avoid accumulating from previous batches
            # avoid exploding gradients
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Logging
            global_step += 1
            epoch_loss += loss.item()
            if log:
                experiment.log({'train loss': loss.item(), 'step' : global_step, 'epoch' : epoch})

            #print('Learning rate:', optimizer.param_groups[0]['lr'])


            ### Evaluation, which we do every eval_interval epochs or abs(eval_interval) batches
            if eval_interval < 0:
                cond = (global_step+1)%eval_interval == 0
            else:
                cond = (epoch+1)%eval_interval == 0 and i == len(train_loader)-1
            if cond:
                if log:
                    ## logging stuff from github
                    histograms = {}

                ## evaluate (could probably be its own function as it is in github)
                model.eval()
                eval_score = 0
                accuracy_score = 0
                if print_csi_metrics:
                    full_base_pred_match_list = []
                    full_base_targ_match_list = []
                final_batch_size_factor = 1

                # iterate through batches in validation set
                #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                with torch.cuda.amp.autocast(enabled=amp):
                    with torch.no_grad():
                        for j, batch in enumerate(val_loader):
                            data, labels = batch[0], batch[1]
                            # load into memory
                            data = data.to(device=device, dtype=torch.float32)#, memory_format=torch.channels_last)
                            if len(data.shape) == 3:
                                data = data.unsqueeze(1)
                            
                            labels = labels.to(device=device, dtype=torch.float32)
                            # replace nans if neeeded
                            if replace_nan:
                                data = torch.nan_to_num(data,nan=0,posinf=0,neginf=0)
                                labels = torch.nan_to_num(labels,nan=0,posinf=0,neginf=0)

                            if input_norm:
                                if input_norm_end == 0:
                                    input_norm_end = data.shape[1]
                                data[:, :input_norm_end] = F.normalize(data[:, :input_norm_end])

                            # predict                            
                            labels_pred = model(data)

                            if sample_weight_var != '':
                                # load weights also. Didn't bother setting up channels last here yet
                                #sample_weights = batch[2].to(device=device, dtype=torch.long)
                                sample_weights = batch[2].to(device=device, dtype=torch.float32)

                            # score batch
                            if sample_weight_var == '':
                                new_eval_score = eval_func(labels_pred, labels)
                            else:
                                new_eval_score = eval_func(labels_pred, labels, sample_weights)

                            # have to add this conditional in case of incongruent final batch size
                            if j == len(val_loader)-1 and len(val_loader)>1:
                                final_batch_size_factor = labels_pred[0].shape[0]/batch_size

                            eval_score += new_eval_score*final_batch_size_factor
                            if print_csi_metrics:
                                base_pred_match_list, base_targ_match_list = get_match_success_from_tensors(labels_pred[0], labels, label_threshold=csi_threshold,
                                                                                                            return_match_lists=True, distance=csi_distance)
                                full_base_pred_match_list.extend(base_pred_match_list)
                                full_base_targ_match_list.extend(base_targ_match_list)                                                          

                model.train()
                # for saving later, but also for reporting epoch
                if eval_interval < 0:
                    cond = save_checkpoint # haven't implemented more specific interval checking for this
                    check_suffix = round(global_step/(batch_size*abs(eval_interval)), 2)
                else:
                    cond = save_checkpoint and (epoch%checkpoint_interval == 0 or epoch==epochs-1) and i==len(train_loader)-1
                    check_suffix = epoch

                length_divisor = len(val_loader)-1+final_batch_size_factor
                eval_score /= length_divisor
                print('Eval score at epoch {}: {}'.format(check_suffix, eval_score))
                print('Learning rate at epoch {}: {}'.format(check_suffix, optimizer.param_groups[0]['lr']))
                if print_csi_metrics:
                    pod, sr, csi = pod_sr_csi(full_base_pred_match_list, full_base_targ_match_list)
                    print('POD at epoch {}: {}'.format(epoch, pod))
                    print('SR at epoch {}: {}'.format(epoch, sr))
                    print('CSI at epoch {}: {}'.format(epoch, csi))
                if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    # pass score to scheduler, which decides whether to possibly reduce learning rate or quit
                    scheduler.step(eval_score)
                else:
                    # scheduler depends only on time, not current score
                    scheduler.step()


                ## more logging
                if log:
                    logging.info('Validation score: {}'.format(eval_score))
                try:
                    if log:
                        experiment.log({'learning rate': optimizer.param_groups[0]['lr'],
                                        'validation score:': eval_score,
                                        'step' : global_step,
                                        'epoch' : epoch,
                                        **histograms})
                except:
                    pass


                # save model checkpoint
                if cond:
                    state_dict=model.state_dict()
                    if not save_best or (save_best and eval_score < min_eval_score):
                        torch.save(state_dict, checkpoint_dir + '/checkpoint_epoch{}.pth'.format(check_suffix))
                    min_eval_score = min(eval_score, min_eval_score)
                    logging.info(f'Checkpoint {epoch} saved!')
                if optimizer.param_groups[0]['lr'] < 1e-9:#1e-5:#1e-4:#1e-6:
                    return

############### Various functions related to generating predictions or extracting labels from predictions ########################
def predict_xr(model, ds, device, out_threshold=0.5, label_var='wave_labels', latitude_var='latitude', longitude_var='longitude',
               input_norm=False, each_class=False, class_names=[], fill_value=float('nan'), transform=None, climax_time_embed_val=-1):
    '''
    Produce an output from a 3D (Channels, lat, lon) xarray ds. Currently only for binary classification

    model: tnn.Module, our network
    ds: xr.Dataset, our inputs
    device: torch.device, cuda or cpu
    out_threshold: float, cutoff to denote a positive classification
    label_var: str, name of correct label variable
    input_norm: bool, whether inputs should be normalized first
    each_class: bool, whether to return probabilities for each class where probs are above out_threshold (if there is more than 1 positive class)
    class_names: list, ordered names of class values 1 ... n
    climax_time_embed_val: str, whether to use ClimaX time embedding when getting a sample. if not empty, use that value to create embedding array

    return: xr.DataArray, 2D (lat, lon) with classification data
    '''
    model.eval()

    if climax_time_embed_val > -1:
        ds['lead_embed'] = ds[label_var]*0+climax_time_embed_val
    
    if not transform:
        tensor = torch.as_tensor(ds[[var for var in ds.data_vars if var!= label_var]].to_array(dim='vars').data.astype('float32')).unsqueeze(0).to(device=device, dtype=torch.float32)
    else:
        tensor = ds[[var for var in ds.data_vars if var!= label_var]].to_array(dim='vars').data.astype('float32')
        tensor = torch.as_tensor(transform(tensor)).unsqueeze(0).to(device=device, dtype=torch.float32)
    if input_norm:
        tensor = F.normalize(tensor, dim=0)
    with torch.no_grad():
        #print(tensor.shape)
        output= model(tensor)[0].cpu()
        if output.shape[1] == 1:
            mask = torch.sigmoid(output)# > out_threshold        
        else:
            # have multiple classes, return which one it is
            mask = torch.softmax(output, dim=1)
            # use out_threshold as a cutoff for minimum of when to consider a non-negative (0) class as being a positive case
            mask_obj = torch.max(mask[:,1:,:],1)            
            if not each_class:
                mask_vals = mask_obj.values
                mask = mask_obj.indices+1                
                mask[mask_vals<out_threshold]=fill_value
            else:
                # return array of probabilities of each positive class where:
                # 1. all values below out_threshold are zeroed
                # 2. all values that are not maxima among the positive classes are zeroed
                mask_vals = mask_obj.values.unsqueeze(1)            
                mask[mask<out_threshold]=fill_value
                mask[mask!=mask_vals.expand(mask_vals.shape[0],mask.shape[1],mask_vals.shape[2],mask_vals.shape[3])] = fill_value
                mask = mask[:, 1:].permute(0,2,3,1)

    dims={latitude_var : ds.dims[latitude_var], longitude_var : ds[longitude_var]}
    coords={latitude_var : ds.coords[latitude_var], longitude_var : ds.coords[longitude_var]}
    if each_class:
        dims.update({'class':mask.shape[1]})
        coords.update({'class':class_names})
    return xr.DataArray(mask.squeeze().numpy(), dims=dims,
                        coords=coords)




def predict_xr_1D(model, ds, device, out_threshold=0.5, label_var='wave_labels', input_norm=False):
    '''
    Produce an output from a 3D (Channels, lat, lon) xarray ds to a single output value. Currently just binary classification

    model: tnn.Module, our network
    ds: xr.Dataset, our inputs
    device: torch.device, cuda or cpu
    out_threshold: float, cutoff to denote a positive classification
    label_var: str, name of correct label variable

    return: Output class value
    '''
    model.eval()
    tensor = torch.as_tensor(ds[[var for var in ds.data_vars if var!= label_var]].to_array(dim='vars').data.astype('float32')).unsqueeze(0).to(device=device, dtype=torch.float32)
    if input_norm:
        tensor = F.normalize(tensor, dim=0)
    with torch.no_grad():
        output= model(tensor).cpu()
        prob = torch.sigmoid(output)
        c = (prob > out_threshold)*1
    return c.squeeze().numpy(), prob.squeeze().numpy()


def label_feature_regions(array, structure=[[1,1,1],[1,1,1],[1,1,1]], threshold=0.5, trim_distance=0, return_data=False,
                          uniform_prob_offset=0, only_center_axis=False, trim_y=False, only_edges=True, alt_trim=False):
    '''
    Returns mask of regions in array where values are >= threshold. Each region in the mask is given a different value.

    array: numpy array, a 2D array of probabilities
    structure: array-like, the definition of connectivity (horizontal, vertical, and diagonal all count by default)
    threshold: minimum value to include as in a region, default: 0.5
    trim_distance: int, number of cells in x direction around center to trim regions 
    uniform_prob_offset: int, number of cells to the right to offset the calculation of max prob location.
                               Used when smudging truth labels which have the same value throughout each region.
    only_edges: attempt to only use edge indices to speed up matching process (note this would increase distance if one region was inside of another)
    alt_trim: realized there was a bug in original trimming code which would separate regions where maxima were not directly adjacent into multiple regions
    
    return: 2d numpy array with regions labeled
    '''
    data = array.copy()
    data[data<threshold] = 0
    # need this for some reason if there are nans
    data[np.isnan(data)] = 0

    labels, _ = scipy.ndimage.label(data, structure=structure)
        
    if trim_distance or only_center_axis:
        # only retain center points of regions plus trim_distance on either side
        temp_regs = [[(labels==val).nonzero(), val] for val in np.unique(labels) if val != 0]

        x_len = data.shape[1]
        y_len = data.shape[0]
        if alt_trim:
            temp_regs_new = temp_regs.copy(deep=True)
        for reg_i, [(y, x), val] in enumerate(temp_regs):
            if not trim_y:
                # trim only in x direction
                y_vals = np.unique(y)
                for y_val in y_vals:
                    # at each latitude, get maximum prob in data from region as center of axis,
                    # then set all points in data outside of trim_distance to 0
                    # (and make sure trim_distance points on either side are the correct val)
                    selects = np.argwhere(y==y_val)
                    ys, xs  = y[selects], x[selects]
                    max_x_i = np.argmax(data[ys,xs])
                    full_max_x_i = int(min(xs[max_x_i]+uniform_prob_offset, x_len-1))
                    max_x   = data[y_val, full_max_x_i]
                    # overwrite data
                    right_x_bound = int(min(full_max_x_i+trim_distance+1, x_len))
                    left_x_bound = int(max(full_max_x_i-trim_distance, 0))
                    data[ys,xs]=0
                    data[y_val, left_x_bound:right_x_bound] = max_x
                    #print(data[y_val][max(left_x_bound-4, 0):min(right_x_bound+4, x_len)])
                    # test code for making weights decay as you go further out
                    #data[y_val, left_x_bound:right_x_bound] = max_x/2
                    #data[y_val, full_max_x_i] = max_x
                    if alt_trim and only_center_axis:
                        # change indices to nan which have been zeroed out
                        temp_regs_new[reg_i][0][0][selects] = np.nan
                        temp_regs_new[reg_i][0][1][selects] = np.nan
                        unzero_index = np.argwhere(y==y_val & x == full_max_x_i)
                        temp_regs_new[reg_i][0][0][unzero_index]=val
                        temp_regs_new[reg_i][0][1][unzero_index]=val                
                        
            else:
                # try replacing NaNs with 0s cause of issues with region labeling
                data[np.isnan(data)] = 0
                # trim only in y direction
                x_vals = np.unique(x)
                for x_val in x_vals:
                    # at each longitude, get maximum prob in data from region as center of axis,
                    # then set all points in data outside of trim_distance to 0
                    # (and make sure trim_distance points on either side are the correct val)
                    selects = np.argwhere(x==x_val)
                    ys, xs  = y[selects], x[selects]
                    try:
                        max_y_i = np.nanargmax(data[ys,xs])
                    except ValueError:
                        # have overwritten this region from another one that was overlying it in latitude, so skip
                        print(data[ys,xs])
                        continue
                    full_max_y_i = ys[max_y_i]+uniform_prob_offset
                    max_y   = data[full_max_y_i, x_val]
                    # overwrite data
                    upper_y_bound = int(min(full_max_y_i+trim_distance+1, y_len))
                    lower_y_bound = int(max(full_max_y_i-trim_distance, 0))
                    data[ys,xs]=0
                    data[lower_y_bound:upper_y_bound, x_val] = max_y
                    # test code for making weights decay as you go further out
                    #data[y_val, left_x_bound:right_x_bound] = max_x/2
                    #data[y_val, full_max_x_i] = max_x                
        if return_data:
            return data
        # re-get regions with newly trimmed regions
        if alt_trim and only_center_axis:
            True
        else:
            labels, _ = scipy.ndimage.label(data, structure=structure)

        if only_edges and not alt_trim:
            # only retain edge points of labels to increase efficiency for matching
            temp_regs = [[(labels==val).nonzero(), val] for val in np.unique(labels) if val != 0]
            x_len = data.shape[1]
            y_len = data.shape[0]
            for [(y, x), val] in temp_regs:
                if not trim_y:
                    # trim only in x direction
                    y_vals = np.unique(y)
                    max_y = np.max(y_vals)
                    min_y = np.min(y_vals)
                    for y_val in y_vals:
                        # at each latitude, get lowest and highest x index, except at highest and lowest y index
                        # then set all other points in labels to 0
                        if y_val in (min_y, max_y):
                            continue
                        selects = np.argwhere(y==y_val)
                        ys, xs  = y[selects], x[selects]
                        max_x_i = np.max(xs)
                        min_x_i = np.min(xs)
                        # overwrite inner labels with 0
                        labels[ys, min_x_i+1:max_x_i]=0
    return labels


def extract_full_feature_regions(labels, connect_array=[]):
    '''
    Returns list of features where each feature is an array of all index pairs located in that region.
    Note these can be much larger than any TWD label and thus could technically score well if giant feature
    labels were predicted, but since this isn't used as a loss function I'm not too concerned.

    labels: 2d numpy array with regions labeled
    connect_array: list of longitude and latitude index thresholds for connecting nearby regions (should look like [lon, lat])
    '''
    label_list = [np.transpose(labels==val).nonzero() for val in np.unique(labels) if val != 0]
    if connect_array:
        lon_thresh = connect_array[0]
        lat_thresh = connect_array[1]
        label_extrema = [[np.min(label[0]), np.max(label[0]), np.min(label[1]), np.max(label[1])] for label in label_list]
        for i, label1 in enumerate(label_list):
            [min_lon1, max_lon1, min_lat1, max_lat1] = [v for v in label_extrema[i]]
            for j, label2 in enumerate(label_list):
                if i==j or label_list[i] is label_list[j]: # don't compare identical objects (which we will create if two regions are close enough)
                    continue
                [min_lon2, max_lon2, min_lat2, max_lat2] = [v for v in label_extrema[j]]
                # merge regions into one if they are within a distance of each other
                lon_dist = np.min([np.abs(diff) for diff in [min_lon1-min_lon2, min_lon1-max_lon2, max_lon1-min_lon2, max_lon1-max_lon2]])
                lat_dist = np.min([np.abs(diff) for diff in [min_lat1-min_lat2, min_lat1-max_lat2, max_lat1-min_lat2, max_lat1-max_lat2]])
                if lon_dist < lon_thresh and lat_dist < lat_thresh:
                    new_label = (np.append(label1[0], label2[0]), np.append(label1[1], label2[1]))
                    label_list[i], label_list[j] = new_label, new_label

        # remove duplicate objects
        label_list = [(np.array(v1), np.array(v2)) for (v1, v2) in (set([(tuple(label[0]), tuple(label[1])) for label in label_list]))]
    return label_list


def match_feature_regions_basic_distance(labels_1, labels_2, threshold=4, distance_array=None):
    '''
    Get whether each region in labels_1 and labels_2 has a matching buddy (non-exclusive),
    defined as when any parts of two regions are within threshold distance (in grid indices) of each other

    labels_1: list of arrays of feature indices
    labels_2: list of arrays of feature indices
    threshold: int, maximum (inclusive) distance in grid to denote a match
    distance_array: 4d np.array, distance between two pairs of indices. Makes things slower, not faster :(

    return: labels_1_matches and labels_2_matches, two lists of [index array, found_match] pairs
    '''
    labels_1 = [[a, False] for a in labels_1]
    labels_2 = [[a, False] for a in labels_2]

    # apologies to me for this n^4 complexity loop
    for l1 in labels_1:        
        l1_array, _ = l1
        for l2 in labels_2:
            l2_array, _ = l2
            # compare every coordinate pair to see if we have a match                
            for i1, j1 in zip(l1_array[0], l1_array[1]):
                if not np.any(distance_array):
                    distances = np.sqrt(np.square(l2_array[0]-i1)+np.square(l2_array[1]-j1))
                else:
                    distances = distance_array[j1,i1][l2_array[1], l2_array[0]]
                if np.any(distances<=threshold):
                    #print(min(distances))
                    l1[1] = True
                    l2[1] = True
                    break                
                        
            if l1[1]: break
    # and then we do it again but the other way for those labels in second group that haven't been paired
    for l2 in labels_2:
        if l2[1]:
            # already matched this one
            continue
        l2_array, _ = l2
        for l1 in labels_1:
            l1_array, _ = l1
            # compare every coordinate pair to see if we have a match
            for i2, j2 in zip(l2_array[0], l2_array[1]):
                distances = np.sqrt(np.square(l1_array[0]-i2)+np.square(l1_array[1]-j2))
                if np.any(distances<=threshold):                    
                    l1[1] = True
                    l2[1] = True
                    break
            if l2[1]: break
    
    return labels_1, labels_2


def pod_sr_csi(pred_features, targ_features):
    '''
    Calculate probability of detection, success rate, and critical success index for predicted feature
    set and target feature set.

    pred_features: list of bools (whether found a match), predicted by network
    targ_features: list of bools (whether found a match), true labels

    return: 3 floats between 0 and 1: pod, sr, csi
    '''
    pod = sum(targ_features)/len(targ_features)
    sr  = sum(pred_features)/len(pred_features)
    csi = 1/(1/pod + 1/sr -1)
    return pod, sr, csi

def generic_match_centroids(pred_labels, targ_labels, pred, targ, distance=10):
    '''
    Get centroids of regions in pred and targ and determine if they have a match
    '''
    for i, region in enumerate(pred_labels):
        # take centroid indices of region
        vals = pred[region[1], region[0]]
        centroid_x_i = int(np.average(region[0], weights=vals))
        centroid_y_i = int(np.average(region[1], weights=vals))
        pred_labels[i] = np.array([[centroid_x_i], [centroid_y_i]])
    for i, region in enumerate(targ_labels):
        # take centroid indices of region
        vals = targ[region[1], region[0]]
        centroid_x_i = int(np.average(region[0], weights=vals))
        centroid_y_i = int(np.average(region[1], weights=vals))
        targ_labels[i] = np.array([[centroid_x_i], [centroid_y_i]])

    
    return match_feature_regions_basic_distance(pred_labels, targ_labels, threshold=distance)
    

                      


def get_match_success_from_tensors(pred, targ, times=[], return_match_lists=True, label_threshold=0.5, distance=4,
                                   trim_distance=0, uniform_prob_offset=0, hurdat=[], hurdat_bounds=[], trim_edge_bounds=[],
                                   connect_array=[], do_wave_centers=False, longitudes=[], latitudes=[]):
    '''
    Full workflow of getting pod, sr, csi from batch of predictions and target labels

    pred: torch.tensor, predicted label probabilities of dims [sample, channels, x, y]
    targ: torch.tensor, predicted label probabilities of dims [sample, x, y]
    return_match_lists: bool, whether to return the lists of individual sample matches (if False) or the pod/sr/CSI
    label_threshold: float, minimum value for labeling predictions as true

    return: 3 floats between 0 and 1: pod, sr, csi
    '''
    pred = pred.squeeze(1).sigmoid().to(dtype=torch.float32).cpu().numpy()
    targ = targ.cpu().numpy()
    # have to pass only one sample at a time unfortunately
    pred_match_list = []
    targ_match_list = []
    for i in range(pred.shape[0]):
        # label the arrays
        pred_labeled = label_feature_regions(pred[i], threshold=label_threshold, trim_distance=trim_distance)
        targ_labeled = label_feature_regions(targ[i], threshold=0.01, trim_distance=trim_distance, uniform_prob_offset=uniform_prob_offset)
        
        if hurdat:
            time = str(int(times[i]))
            # mask out in box around storms
            time_storms = []
            for storm in hurdat:
                i = 0
                while i < len(storm['date']):
                    if ''.join([storm['date'][i], storm['time'][i]]) == time and storm['state'][i] in ('SS', 'SD', 'TS', 'TD', 'HU'):
                        time_storms.append((0-abs(float(storm['lon'][i][:-1])), float(storm['lat'][i][:-1])))
                        i = len(storm['date'])
                    i = i+1
            # 10x20 degree (40x80 cell) box around storm center
            for (storm_lon, storm_lat) in time_storms:
                # don't care about storms outside of bounds    
                center_storm_lon_i = np.argmin(np.abs(longitudes.values-storm_lon))
                center_storm_lat_i = np.argmin(np.abs(latitudes.values-storm_lat))
                min_lon_i = np.max([center_storm_lon_i-hurdat_bounds[0], 0])
                max_lon_i = np.min([center_storm_lon_i+hurdat_bounds[0]+1, len(longitudes)])
                min_lat_i = np.max([center_storm_lat_i-hurdat_bounds[1], 0])
                max_lat_i = np.min([center_storm_lat_i+hurdat_bounds[1]+1, len(latitudes)])
                # just mask out entire meridional column of grid since waves have varying extents             
                pred_labeled[min_lat_i:max_lat_i, min_lon_i:max_lon_i] = 0
                targ_labeled[min_lat_i:max_lat_i, min_lon_i:max_lon_i] = 0


        # get list of feature region arrays
        pred_labels = extract_full_feature_regions(pred_labeled, connect_array=connect_array)
        targ_labels = extract_full_feature_regions(targ_labeled, connect_array=connect_array)

        if do_wave_centers:
            # want to still have lines since maxima can be misaligned in latitude, so expand each array to a line axis 10 degrees tall and 0.25 degrees wide 
            # doing this here rather than in label_feature_regions to avoid issues with connect_array and axes that are not close to one another             
            # this does make all axes straight N/S, but that's not a huge deal                                

            for j, region in enumerate(pred_labels):
                # try out finding centroid of region           
                vals = pred[i][region[1], region[0]]
                lon_i = int(np.average(region[0], weights=vals))
                lat_i = int(np.average(region[1], weights=vals))
                min_lat_i = np.max([lat_i-20, 0])
                max_lat_i = np.min([lat_i+21, len(latitudes)])

                new_lats = np.arange(min_lat_i, max_lat_i, 1)
                new_lons = np.full(len(new_lats), lon_i)
                pred_labels[j] = (new_lons, new_lats)

            for j, region in enumerate(targ_labels):
                # these all have same value so just want mean  
                lat_i = round(np.mean(region[1]))
                lon_i = round(np.mean(region[0]))
                min_lat_i = np.max([lat_i-20, 0])
                max_lat_i = np.min([lat_i+21, len(latitudes)])

                new_lats = np.arange(min_lat_i, max_lat_i, 1)
                new_lons = np.full(len(new_lats), lon_i)
                targ_labels[j] = (new_lons, new_lats)


        # get matches between the two lists
        pred_matches, targ_matches = match_feature_regions_basic_distance(pred_labels, targ_labels, threshold=distance)
        pred_matches = [matched for [_, matched] in pred_matches]
        targ_matches = [matched for [_, matched] in targ_matches]
        # build complete list from this batch
        pred_match_list.extend(pred_matches)
        targ_match_list.extend(targ_matches)

    if return_match_lists:
        return pred_match_list, targ_match_list
    return pod_sr_csi(pred_match_list, targ_match_list)


def get_permutation_importance(model, input_data, device, batch_size=1, label_var='wave_labels', error_metric=None,
                               batch_based_averaging=True, amp=True, print_progress=True, return_times=False, **batch_args):
    '''
    Calculate and return permutation importance for each channel of a model across given test samples. Calculations follow
    structure of https://captum.ai/api/feature_permutation.html

    model: torch.module, a neural network
    data: xr.Dataset, data where all DataArrays share the same dims and one DataArray is the labels
    error_metric: class, a callable class for gauging performance of a feature. Should take (pred, targ) as arguments
    batch_based_averaging: bool, whether to average across batches / number of samples (if True) or compute as a single metric at the end once
                                 all samples are tallied. Specifically for stuff like CSI where every sample would have a different weight by number of cases in each sample

    return: dict of channel names and their permutation importance
    '''
    if not error_metric:
        error_metric=fss(3,3)
    model.eval()
    if print_progress:
        total_batches = len(input_data['time'])/batch_size
    data_loader = create_batches_from_xarray(input_data, label_var=label_var, batch_size=batch_size, return_times=return_times, **batch_args)
    final_batch_size_factor=1
    print('starting loop')
    # iterate through batches in dataset
    for i, batch in enumerate(data_loader):
        if print_progress:
            print(i/total_batches)
        #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        with torch.cuda.amp.autocast(enabled=amp):
            with torch.no_grad():                
                data, labels = batch[0], batch[1]
                if return_times:
                    times = batch[2]
                if i == 0:
                    if batch_based_averaging:
                        permutation_importance = {i : 0 for i in range(data.shape[1])}
                    else:
                        permutation_importance = {i: {'pred_match_list' : [], 'targ_match_list' : []} for i in range(data.shape[1])}
                        full_base_pred_match_list = []
                        full_base_targ_match_list = []                        
                # load into memory                    
                data = data.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)
                # predict using base data setup
                labels_pred = model(data)
                
                # have to add this conditional in case of incongruent final batch size
                if i == len(data_loader)-1 and len(data_loader)>1:
                    final_batch_size_factor = labels_pred[0].shape[0]/batch_size

                if batch_based_averaging:
                    if not return_times:
                        base_score = float(error_metric(labels_pred, labels)*final_batch_size_factor)
                    else:
                        base_score = float(error_metric(labels_pred, labels, times=times)*final_batch_size_factor)                        
                else:
                    if not return_times:
                        base_pred_match_list, base_targ_match_list = error_metric(labels_pred, labels)
                    else:
                        base_pred_match_list, base_targ_match_list = error_metric(labels_pred, labels, times=times)                        
                    full_base_pred_match_list.extend(base_pred_match_list)
                    full_base_targ_match_list.extend(base_targ_match_list)
                # permute batch for each channel, where values for that channel are shuffled (keeping structure for each sample) across all samples
                for c in range(data.shape[1]):
                    # get random ordering of sample indices, then use those to permute one channel's values
                    idx = torch.randperm(data.shape[0])
                    data_perm = data.clone()
                    data_perm[:, c] = data[idx, c].view(data[:,c].size())
                    # calculate performance of model when randomly shuffling one channel
                    if batch_based_averaging:
                        if not return_times:
                            perm_score = error_metric(model(data_perm), labels)
                        else:
                            perm_score = error_metric(model(data_perm), labels, times=times)
                        permutation_importance[c] = permutation_importance[c] + (float(perm_score)*final_batch_size_factor-base_score)
                    else:
                        if not return_times:
                            perm_pred_match_list, perm_targ_match_list = error_metric(model(data_perm), labels)
                        else:
                            perm_pred_match_list, perm_targ_match_list = error_metric(model(data_perm), labels, times=times)                            
                        permutation_importance[c]['pred_match_list'].extend(perm_pred_match_list)
                        permutation_importance[c]['targ_match_list'].extend(perm_targ_match_list)

    data_vars = [var for var in input_data.data_vars if var!= label_var]
    if batch_based_averaging:
        # average importance across batches and return in dict of channel name : importance
        length_divisor = len(data_loader)-1+final_batch_size_factor
        permutation_importance = {data_vars[i] : permutation_importance[i] / length_divisor for i in permutation_importance.keys()}
    else:
        base_pod, base_sr, base_csi = pod_sr_csi(full_base_pred_match_list, full_base_targ_match_list)
        for i in permutation_importance.keys():
            pred_matches = permutation_importance[i]['pred_match_list']
            targ_matches = permutation_importance[i]['targ_match_list']
            pod, sr, csi = pod_sr_csi(pred_matches, targ_matches)
            permutation_importance[i] = {'pod' : pod-base_pod, 'sr' : sr-base_sr, 'csi' : csi-base_csi}
        permutation_importance = {data_vars[i] : permutation_importance[i] for i in permutation_importance.keys()}
    return permutation_importance

def get_permutation_importance_pixelwise_csi(model, input_data, device, thresholds=[], out_threshold=0.1, match_dist=0,
                                             amp=True, land_mask=[], batch_size=1, label_var='imt_labels', print_progress=True, **batch_args):
    '''
    Calculate and return permutation importance for each channel of a model across given test samples using pixelwise CSI as a metric. Expects multiple classes

    model: torch.module, a neural network
    input_data: xr.Dataset, data where all DataArrays share the same dims and one DataArray is the labels
    
    return: dict of channel names and their permutation importance
    '''
    model.eval()
    if print_progress:
        total_batches = len(input_data['time'])/batch_size
    data_loader = create_batches_from_xarray(input_data, label_var=label_var, batch_size=batch_size, randomized_samples=True, return_times=True, **batch_args)
    with torch.cuda.amp.autocast(enabled=amp):
        # pre-create lookup table for time strs to xarray indices
        times_ints_to_indices = {int(''.join(re.split('[:-]', str(t))).replace('T', '')[:12]) : i for i, t in enumerate(input_data['time'].values)}
        permutation_indices = {i : [] for i in range(len(data_loader))}

        print('starting loop')
        print('device', device)
        # iterate through batches in dataset, accumulating predictions as numpy arrays to then calculate success from after
        for i, batch in enumerate(data_loader):
            if print_progress:
                print(i/total_batches)
            #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
            #with torch.cuda.amp.autocast(enabled=amp):
            if True:
                with torch.no_grad():                
                    data, _, times = batch[0], batch[1], batch[2]
                    # load into memory                    
                    data = data.to(device=device, dtype=torch.float32)#, memory_format=torch.channels_last)
                    preds = model(data)
                    # convert multichannel prediction to usable format
                    preds = preds[0].softmax(dim=1)
                    mask_obj = torch.max(preds[:, 1:,:],1) # only check along non-null case channels
                    # return array of probabilities of each positive class where:   
                    # 1. all values below out_threshold are zeroed                  
                    # 2. all values that are not maxima among the positive classes are zeroed                                                     
                    mask_vals = mask_obj.values.unsqueeze(1)                        
                    preds[preds<out_threshold]=float('nan')                    
                    preds[preds!=mask_vals.expand(mask_vals.shape[0],preds.shape[1],mask_vals.shape[2],mask_vals.shape[3])] = float('nan')
                    preds = preds[:, 1:]
                    if i==0:
                        # create accumulator array for the first time
                        pred_accumulator = np.zeros([len(input_data['time']), *preds.shape[1:]])
                            
                    # get time indices so can add to accumulator
                    time_indices = [times_ints_to_indices[int(t)] for t in times]                
                    pred_accumulator[time_indices] = preds.cpu().numpy()

                    # save permutation indices for later
                    permutation_indices[i] = torch.randperm(preds.shape[0])
     
    # save scores from this set of predictions as the baseline
    all_targs = input_data[label_var]

    def calc_csis(i):
        matches = {k : {'pred' : [], 'real' : []} for k in range(pred_accumulator.shape[1]+1)}
        targ = all_targs[i].copy()
        pred = pred_accumulator[i].copy()
        for k in range(pred_accumulator.shape[1]):
            # separately calculate matches for each class
            class_targ = xr.where(targ==k+1, targ, 0)
            class_pred = class_targ*0+pred[k]
            
            if len(land_mask) > 0:
                class_targ = class_targ * (land_mask==0)
                class_pred = class_pred * (land_mask==0)
            # calculate whether labels predicted match real labels or not
            pred_labeled = label_feature_regions(class_pred.data,threshold=thresholds[k], trim_y=True, only_center_axis=True)
            targ_labeled = label_feature_regions(class_targ.data,threshold=0.01, uniform_prob_offset=6, trim_y=True, only_center_axis=True)#
            # do pixelwise CSI for these (which are now line segments)
            pred_smudged = ((class_pred*0).fillna(0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
            targ_smudged = (class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

            pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
            pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
            targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)
            targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)
            
            # extract indices as objects
            pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
            pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
            pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]
            
            targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
            targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
            targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

            matches[k]['pred'].extend(pred_matches)
            matches[k]['real'].extend(targ_matches)
            
        # also calculate success of comparing matches between any classes

        non_class_targ=targ
        non_class_pred=non_class_targ*0+np.nan_to_num(pred, 0).sum(axis=0)

        pred_labeled=label_feature_regions(non_class_pred.data, threshold=thresholds[k], trim_y=True, only_center_axis=True)
        targ_labeled=label_feature_regions(non_class_targ.data, threshold=0.01, trim_y=True, only_center_axis=True, uniform_prob_offset=6)        

        # do pixelwise CSI for these (which are now line segments)
        pred_smudged = ((non_class_pred*0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
        targ_smudged = (non_class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

        pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
        pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
        targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)

        targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)

        # extract indices as objects
        pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
        pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
        pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]
        
        targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
        targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
        targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

        matches[k+1]['pred'].extend(pred_matches)
        matches[k+1]['real'].extend(targ_matches)
        
        return matches

    returns = Parallel(n_jobs=-1)(delayed(calc_csis)(i) for i in range(all_targs.shape[0]))
    matches = {k : {'pred' : [], 'real' : []} for k in range(pred_accumulator.shape[1]+1)}
    for r in returns:
        for k in matches.keys():
            for dest in ['pred', 'real']:
                matches[k][dest].extend(r[k][dest])
    
            
    # add up totals from each class
    base_csi_dict = {class_i : {'pod' : np.nan, 'sr' : np.nan, 'csi' : np.nan} for class_i in matches.keys()}
    for class_i in matches.keys():
        pred_match_list = []
        targ_match_list = []
        pred_matches = matches[class_i]['pred']
        targ_matches = matches[class_i]['real']

        for (xs, ys), matched in pred_matches:
            pred_match_list.extend([matched]*len(xs))
        for (xs, ys), matched in targ_matches:
            targ_match_list.extend([matched]*len(xs))
            
        [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
        base_csi_dict[class_i]['pod'] = pod_val
        base_csi_dict[class_i]['sr'] = sr_val        
        base_csi_dict[class_i]['csi'] = csi_val        

    
    print('baseline csi', base_csi_dict)
    data_vars = [str(var) for var in input_data.data_vars if str(var)!= label_var]    
    scores_by_var = {class_i : {str(var) : {'pod' : 0, 'sr' : 0, 'csi' : 0} for var in data_vars} for class_i in matches.keys()}
    # iterate through each variable, using saved randperm order to permute that variable in each batch and make predictions
    for c in range(data.shape[1]):
        var = list(input_data.data_vars)[c]
        print(var)
        with torch.cuda.amp.autocast(enabled=amp):
            for i, batch in enumerate(data_loader):
                if print_progress:
                    print(i/total_batches)
                #with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                #with torch.cuda.amp.autocast(enabled=amp):
                if True:
                    with torch.no_grad():                
                        data, _, times = batch[0], batch[1], batch[2]
                        # load into memory                    
                        data = data.to(device=device, dtype=torch.float32)#, memory_format=torch.channels_last)
                        # randomize channel
                        data[:, c] = data[permutation_indices[i], c].view(data[:, c].size())
                        preds = model(data)
                        # convert multichannel prediction to usable format
                        preds = preds[0].softmax(dim=1)
                        mask_obj = torch.max(preds[:, 1:,:],1) # only check along non-null case channels
                        # return array of probabilities of each positive class where:   
                        # 1. all values below out_threshold are zeroed                  
                        # 2. all values that are not maxima among the positive classes are zeroed                                                     
                        mask_vals = mask_obj.values.unsqueeze(1)                        
                        preds[preds<out_threshold]=float('nan')
                        preds[preds!=mask_vals.expand(mask_vals.shape[0],preds.shape[1],mask_vals.shape[2],mask_vals.shape[3])] = float('nan')
                        preds = preds[:, 1:]
                        if i==0:
                            # create accumulator array for the first time
                            pred_accumulator = np.zeros([len(input_data['time']), *preds.shape[1:]])

                        # get time indices so can add to accumulator
                        time_indices = [times_ints_to_indices[int(t)] for t in times]                
                        pred_accumulator[time_indices] = preds.cpu().numpy()

        # save scores from this set of predictions as the baseline
        all_targs = input_data[label_var]
        def calc_csis(i):
            matches = {k : {'pred' : [], 'real' : []} for k in range(pred_accumulator.shape[1]+1)}
            targ = all_targs[i].copy()
            pred = pred_accumulator[i].copy()
            for k in range(pred_accumulator.shape[1]):
                # separately calculate matches for each class
                class_targ = xr.where(targ==k+1, targ, 0)
                class_pred = class_targ*0+pred[k]

                if len(land_mask) > 0:
                    class_targ = class_targ * (land_mask==0)
                    class_pred = class_pred * (land_mask==0)

                # calculate whether labels predicted match real labels or not
                pred_labeled = label_feature_regions(class_pred.data,threshold=thresholds[k], trim_y=True, only_center_axis=True)
                targ_labeled = label_feature_regions(class_targ.data,threshold=0.01, uniform_prob_offset=6, trim_y=True, only_center_axis=True)#

                # do pixelwise CSI for these (which are now line segments)
                pred_smudged = ((class_pred*0).fillna(0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
                targ_smudged = (class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

                pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
                pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
                targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)
                targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)

                # extract indices as objects
                pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
                pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
                pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]

                targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
                targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
                targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

                matches[k]['pred'].extend(pred_matches)
                matches[k]['real'].extend(targ_matches)

            # also calculate success of comparing matches between any classes

            non_class_targ=targ
            non_class_pred=non_class_targ*0+np.nan_to_num(pred, 0).sum(axis=0)

            pred_labeled=label_feature_regions(non_class_pred.data, threshold=thresholds[k], trim_y=True, only_center_axis=True)
            targ_labeled=label_feature_regions(non_class_targ.data, threshold=0.01, trim_y=True, only_center_axis=True, uniform_prob_offset=6)        

            # do pixelwise CSI for these (which are now line segments)
            pred_smudged = ((non_class_pred*0)+pred_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()
            targ_smudged = (non_class_targ*0+targ_labeled).rolling(latitude=match_dist*2+1, longitude=match_dist*2+1, center=True, min_periods=1).sum()            

            pred_matches = xr.where((pred_labeled>0) & (targ_smudged>0), 1, 0)
            pred_non_matches = xr.where((pred_labeled>0) & (~(targ_smudged>0)), 1, 0)
            targ_matches = xr.where((targ_labeled>0) & (pred_smudged>0), 1, 0)
            targ_non_matches = xr.where((targ_labeled>0) & (~(pred_smudged>0)), 1, 0)

            # extract indices as objects
            pred_matches = extract_full_feature_regions(label_feature_regions(pred_matches.values, threshold=0.0001))
            pred_non_matches = extract_full_feature_regions(label_feature_regions(pred_non_matches.values, threshold=0.0001))
            pred_matches = [[vals, True] for vals in pred_matches] + [[vals, False] for vals in pred_non_matches]

            targ_matches = extract_full_feature_regions(label_feature_regions(targ_matches.values, threshold=0.0001))
            targ_non_matches = extract_full_feature_regions(label_feature_regions(targ_non_matches.values, threshold=0.0001))
            targ_matches = [[vals, True] for vals in targ_matches] + [[vals, False] for vals in targ_non_matches]

            matches[k+1]['pred'].extend(pred_matches)
            matches[k+1]['real'].extend(targ_matches)            
            return matches

        returns = Parallel(n_jobs=-1)(delayed(calc_csis)(i) for i in range(all_targs.shape[0]))
        matches = {k : {'pred' : [], 'real' : []} for k in range(pred_accumulator.shape[1]+1)}
        for r in returns:
            for k in matches.keys():
                for dest in ['pred', 'real']:
                    matches[k][dest].extend(r[k][dest])


        # add up totals from each class
        for class_i in matches.keys():
            pred_match_list = []
            targ_match_list = []
            pred_matches = matches[class_i]['pred']
            targ_matches = matches[class_i]['real']

            for (xs, ys), matched in pred_matches:
                pred_match_list.extend([matched]*len(xs))
            for (xs, ys), matched in targ_matches:
                targ_match_list.extend([matched]*len(xs))

            [pod_val, sr_val, csi_val] = pod_sr_csi(pred_match_list, targ_match_list)
            scores_by_var[class_i][var]['pod'] = pod_val - base_csi_dict[class_i]['pod']
            scores_by_var[class_i][var]['sr'] = sr_val - base_csi_dict[class_i]['sr']        
            scores_by_var[class_i][var]['csi'] = csi_val - base_csi_dict[class_i]['csi']       
            
            print(var, scores_by_var[class_i][var])
                
    
    return scores_by_var


############### Classes corresponding to loss functions ########################

class fss:
    '''
    # replicating Lagerquist and Uphoff 2022 FSS 
    # they use a Conv2d with custom mean kernel from Keras; this is not something natively supported in pytorch,
    # so instead I do a sliding window using Unfold    
    # original link: https://github.com/thunderhoser/loss_function_paper_2022/blob/main/loss_functions_journal_paper_2022.ipynb
    
    pred: tensor, predicted labels (singleton dimension at index 1, values need to be translated to sigmoid)
    targ: tensor, true labels (one class, 0 or 1)
    x_size: int, width of sliding window
    y_size: int, height of sliding window
    #weights: 
    
    '''
    def __init__(self, x_size=3, y_size=3):
        self.x_size = x_size
        self.y_size = y_size
    def __call__(self, pred, targ): #,weights=None):
        # assume pred is passed as a singleton list
        pred = pred[0]        
        pred_sig = pred.sigmoid()
        smoother = tnn.Unfold((self.y_size,self.x_size))#,(1,1),(0,0),(1,1))
        smoothed_pred = smoother(pred_sig).mean(dim=1)
        smoothed_targ = smoother(targ.unsqueeze(1).to(dtype=torch.float32)).mean(dim=1)
        actual_mse = torch.mean((smoothed_targ-smoothed_pred)**2)
        reference_mse = torch.mean(smoothed_targ**2+smoothed_pred**2)

        loss = actual_mse / reference_mse
        return loss

class fss_multi_tensors:
    # Same as above, but for multiple prediction tensors (e.g. when doing deep supervision)
    def __init__(self, x_size=3, y_size=3, drop_mask=None, eval_mode=False, norm_sample_weights=False, sample_skip_value=0):
        self.x_size = x_size
        self.y_size = y_size
        self.drop_mask = drop_mask
        self.eval_mode = eval_mode
        self.norm_sample_weights = norm_sample_weights
        self.sample_skip_value = -1 #value in labels to give 0 weights to
        if self.drop_mask is not None:
            # ~(tensor) produces weird input so need to use logical_not
            # self.drop_mask is set to be true for values we want to keep (the inverse of the drop_mask parameter)
            self.drop_mask=torch.as_tensor(self.drop_mask).logical_not().to(dtype=torch.float32)

    def __call__(self, preds, targ): 
        preds_sig = [pred.sigmoid() for pred in preds]
        if self.eval_mode:
            # only care about first predicted tensor
            preds_sig = [preds_sig[0]]
        smoother = tnn.Unfold((self.y_size,self.x_size))
        if self.drop_mask is not None:
            drop_mask = self.drop_mask.expand(preds_sig[0].shape)
            # don't include any pools that included a non-mask area in calculations
            drop_mask = smoother(drop_mask).to(dtype=torch.float32).mean(dim=1)==1
        smoothed_preds = [smoother(pred_sig).mean(dim=1) for pred_sig in preds_sig]
        if self.norm_sample_weights:  
            # assume targ values are already weighted; we want to normalize those weights
            # and implement them in the loss instead to help avoid abnormal loss values
            weights = targ.clone()
            weights[weights == 0] = 1
            if self.sample_skip_value:
                weights[weights == self.sample_skip_value] = 0
                targ[targ == self.sample_skip_value] = 0
            weights = weights/weights.sum()
            smoothed_weights = smoother(weights.unsqueeze(1).to(dtype=torch.float32)).mean(dim=1)
            targ = (targ>0)*1
        smoothed_targ = smoother(targ.unsqueeze(1).to(dtype=torch.float32)).mean(dim=1)
        losses = 1            
        if self.drop_mask is not None:
            smoothed_preds = [smoothed_pred[drop_mask] for smoothed_pred in smoothed_preds]
            smoothed_targ = smoothed_targ[drop_mask]
            if self.norm_sample_weights:
                smoothed_weights = smoothed_weights[drop_mask]
        for smoothed_pred in smoothed_preds:
            if not self.norm_sample_weights:
                actual_mse = torch.mean((smoothed_targ-smoothed_pred)**2)
                reference_mse = torch.mean(smoothed_targ**2+smoothed_pred**2)                
            else:
                actual_mse = torch.mean(((smoothed_targ-smoothed_pred)**2)*smoothed_weights)
                reference_mse = torch.mean((smoothed_targ**2+smoothed_pred**2)*smoothed_weights) 
            loss = actual_mse / reference_mse
            losses*=loss
            
        return losses

class fss_multi_tensors_and_classes:
    # Same as above, but for multiple prediction tensors and output classes
    def __init__(self, x_size=3, y_size=3, n_classes=3, drop_mask=None, eval_mode=False):
        self.x_size = x_size
        self.y_size = y_size
        self.n_classes=n_classes
        self.drop_mask = drop_mask
        self.eval_mode = eval_mode
        if self.drop_mask is not None:
            # ~(tensor) produces weird input so need to use logical_not
            # self.drop_mask is set to be true for values we want to keep (the inverse of the drop_mask parameter)
            self.drop_mask=torch.as_tensor(self.drop_mask).unsqueeze(0).unsqueeze(0).logical_not().to(dtype=torch.float32)
        #print(self.drop_mask.min(), self.drop_mask.max())
    def __call__(self, preds, targ): #,weights=None):
        preds_sig = [pred.softmax(dim=1) for pred in preds]
        if self.eval_mode:
            # skip negative class and only use first predicted tensor
            preds_sig = [preds_sig[0][:,1:]]
        smoother = tnn.Unfold((self.y_size,self.x_size))#,(1,1),(0,0),(1,1))
        if self.drop_mask is not None:
            drop_mask = self.drop_mask.expand(preds_sig[0].shape)
            drop_mask = smoother(drop_mask).to(dtype=torch.bool)
        smoothed_preds = [smoother(pred_sig) for pred_sig in preds_sig]

        # NOTE: ALL TARG VALUES WILL BE CONVERTED TO LONGS TO ACCOMODATE ONE_HOT, WHICH WILL CAUSE PROBLEMS IF VALUES ARE NOT WHOLE NUMBERS ALREADY
        smoothed_targ = smoother(tnn.functional.one_hot(targ.to(dtype=torch.long), self.n_classes).permute(0,3,1,2).to(dtype=torch.float32))
        
        # each smoothed object is now shape [batch_size, n_classes*x_size*y_size, n_possible_x*y_rectangles_in_grid]
        # (so last dim is which rectangle you're dealing with)
        # want to take mean of each x*y shape in dim=1, but only within each class, and leave class count intact,
        # so end up with [batch_size, n_classes, n_possible_x*y_rectangles_in_grid] (but permuted to save time
        # since that's what pooler expects)
        pooler = tnn.AvgPool1d(self.x_size*self.y_size)
        smoothed_preds = [pooler(smoothed_pred.permute(0,2,1)) for smoothed_pred in smoothed_preds]
        smoothed_targ  = pooler(smoothed_targ.permute(0,2,1))
        if self.drop_mask is not None:
            # don't include any pools that included a non-mask area in calculations    
            drop_mask = pooler(drop_mask.to(dtype=torch.long).permute(0,2,1))==1            
            smoothed_preds = [smoothed_pred[drop_mask] for smoothed_pred in smoothed_preds]
            smoothed_targ = smoothed_targ[drop_mask]
        losses = 1
        for smoothed_pred in smoothed_preds:
            actual_mse = torch.mean((smoothed_targ-smoothed_pred)**2)
            reference_mse = torch.mean(smoothed_targ**2+smoothed_pred**2)
            loss = actual_mse / reference_mse
            losses*=loss
        return losses

    
class rmse_lat_weighted:
    # latitude-weighted RMSE as used in various ML weather prediction works
    # NOTE: haven't tested with multiple targ variables
    def __init__(self, lat, eval_mode=False):
        lat_weights = np.cos(np.deg2rad(lat))
        lat_weights = lat_weights / lat_weights.mean()
        self.lat_weights = torch.as_tensor(lat_weights).unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32)
        self.eval_mode = eval_mode
        
    def __call__(self, preds, targ):
        rmse = 1
        for pred in preds:
            rmses = torch.zeros(pred.shape[1])
            for i in range(pred.shape[1]):
                rmses[i] = torch.sqrt(torch.mean((torch.pred[:, i]-targ[:, i]) ** 2 * self.lat_weights, dim=(-2, -1)))
            rmse*=torch.mean(rmses)
            if self.eval_mode:
                # only do first pred array
                break
        return rmse
        
    
class bce:
    # just pytorch BCELoss
    def __call__(self, pred, targ):
        # assume pred is passed as a singleton list
        pred = pred[0]        
        #pred_sig = pred#.sigmoid()
        pred_sig = pred.squeeze(1)
        #targ_big = targ.unsqueeze(1).to(dtype=torch.float32)
        targ_big = targ.to(dtype=torch.float32)
        loss_f = tnn.BCEWithLogitsLoss()
        return loss_f(pred_sig, targ_big)
    

class bce_sigmoid:
    # just pytorch BCELoss with sigmoid activation to get things in 0-1 range
    def __call__(self, pred, targ):
        # assume pred is passed as a singleton list
        pred = pred[0]                
        pred_sig = pred.sigmoid()
        targ_big = targ.unsqueeze(1).to(dtype=torch.float32)
        loss_f = tnn.BCEWithLogitsLoss()
        return loss_f(pred_sig, targ_big)

    
class class_get_match_success_from_tensors:
    '''
    Intended for use as an error metric class like other scoring classes, except this can't be a loss function since it's not differentiable
    '''
    def __init__(self, threshold, distance=4, uniform_prob_offset=0, trim_distance=0, hurdat=[], hurdat_bounds=[], trim_edge_bounds=[], connect_array=[],
                 do_wave_centers=False, longitudes=[], latitudes=[]):
        self.threshold=threshold
        self.distance=distance
        self.trim_distance=trim_distance
        self.uniform_prob_offset=uniform_prob_offset
        self.hurdat=hurdat
        self.trim_edge_bounds=trim_edge_bounds
        self.connect_array=connect_array
        self.do_wave_centers=do_wave_centers
        self.hurdat_bounds=hurdat_bounds
        self.longitudes=longitudes
        self.latitudes=latitudes
    def __call__(self, pred, targ, times=[]):
        if type(pred) is list:
            # for nets that send list of outputs
            pred=pred[0]
        return get_match_success_from_tensors(pred, targ, times=times, return_match_lists=True, label_threshold=self.threshold, distance=self.distance,
                                              trim_distance=self.trim_distance, uniform_prob_offset=self.uniform_prob_offset, hurdat=self.hurdat, hurdat_bounds=self.hurdat_bounds,
                                              trim_edge_bounds=self.trim_edge_bounds, connect_array=self.connect_array, do_wave_centers=self.do_wave_centers,
                                              longitudes=self.longitudes, latitudes=self.latitudes)

