# -*- coding: utf-8 -*-
"""chexpert-pwa_split1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mD13F85sEO9GnYn020armAuO9VHNqRaE
"""



import torch
torch.cuda.is_available()

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import os
import cv2
sns.set_style('whitegrid')
# %matplotlib inline
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import resnet18
from torchvision.models import ResNet
import sklearn
# import torchmetrics
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_train_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_train_1.npy"
Xtrain = np.load(path_x)
ytrain = np.load(path_y)

np.unique(ytrain,return_counts=True)


print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_val_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_val_1.npy"
Xval = np.load(path_x)
yval = np.load(path_y)

np.unique(yval,return_counts=True)


print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/X_test_1.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/chexpert_128/y_test_1.npy"
Xtest = np.load(path_x)
ytest = np.load(path_y)

np.unique(ytest,return_counts=True)
 

def extract_sample(n_way, n_support, n_query, datax, datay):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  label=[]
  K = np.random.choice(np.unique(datay), n_way, replace=True)
#   print(K)
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)

    sample_cls = perm[:(n_support+n_query)]

    sample.append(sample_cls)
    for i in range(n_support+n_query):
      label.append(cls)
#     print(len(sample_cls),len(sample_cls[0]),len(sample_cls[1]),len(sample_cls[2]))
#     print(len(sample))
  sample = np.array(sample)
#   print(sample.shape)
#   print(sample)
  sample = torch.from_numpy(sample).float()
  sample = sample.permute(0,1,4,2,3)
#   print(sample)
#   print(sample.shape)
#   myorder = [0,1,4,2,3]
#   K = [K[i] for i in myorder]
#   print(K)
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_labels':K,
      'n_label':label
      })

def extract_sample_test(K,n_way, n_support, n_query, datax, datay):

  sample = []
  label = []
  #K = np.choice(np.unique(datay), n_way, replace=False)
#   K = df_mod_test['Label'].unique()
#   print(K)
#   print(K)
  for cls in K:
    #datax_cls represents the images data 64*64*3 corresponding to a class
#     print(cls)
    datax_cls = datax[datay == cls]
#     print(datax_cls)
    #random permutations on datax_cls
    perm = np.random.permutation(datax_cls)
    #selects n_support + n_query
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
#     print(sample
    for i in range(n_support+n_query):
        label.append(cls)
  sample = np.array(sample)
#   print(sample.shape)
  sample = torch.from_numpy(sample).float()
  #sample size is: torch.Size([3, 6, 64, 64, 3])
#   print(sample.size())
  sample = sample.permute(0,1,4,2,3)
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label,
      'K':K
      })

def extract_sample_val(n_way, n_support, n_query, datax, datay):

  sample = []
  label = []
  #K = np.choice(np.unique(datay), n_way, replace=False)
  K = df_mod_val['Label'].unique()
  for cls in K:
    #datax_cls represents the images data 64*64*3 corresponding to a class
#     print(cls)
    datax_cls = datax[datay == cls]
#     print(datax_cls)
    #random permutations on datax_cls
    perm = np.random.permutation(datax_cls)
    #selects n_support + n_query
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
#     print(sample
    for i in range(n_support+n_query):
        label.append(cls)
  sample = np.array(sample)
#   print(sample.shape)
  sample = torch.from_numpy(sample).float()
  #sample size is: torch.Size([3, 6, 64, 64, 3])
#   print(sample.size())
  sample = sample.permute(0,1,4,2,3)
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label,
      'K':K
      })

def display_sample(sample):
  """
  Displays sample in a grid
  Args:
      sample (torch.Tensor): sample of images to display
  """
  #need 4D tensor to create grid, currently 5D
  sample_4D = sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])
  #make a grid
  out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
  plt.figure(figsize = (16,7))
  plt.imshow(out.permute(1, 2, 0))

sample_example = extract_sample(3, 4, 1, Xtrain, ytrain)
display_sample(sample_example['images'])

torch.cuda.empty_cache()

class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model

import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

def load_protonet_conv_se(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  def conv_block_se(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        SE_Block(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block_se(hid_dim, z_dim),
    Flatten(),
    nn.Linear(z_dim * 8 * 8, 768)  
    )
    
  return ProtoNet(encoder)

def load_protonet_conv_eca(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  def conv_block_eca(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        eca_layer(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block_eca(hid_dim, z_dim),
    Flatten(),
    nn.Linear(z_dim * 8 * 8, 768)  
    )
    
  return ProtoNet(encoder)

def load_protonet_conv_cbam(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  def conv_block_cbam(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        CBAM(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block_cbam(hid_dim, z_dim),
    Flatten(),
    nn.Linear(z_dim * 8 * 8, 768)  
    )
    
  return ProtoNet(encoder)

def load_protonet_conv(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  def conv_block_se(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        SE_Block(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, z_dim),
    Flatten(),
    nn.Linear(z_dim * 8 * 8, 768)  
    )
#   print(encoder.shape)
  return ProtoNet(encoder)

def one_hot(indices, depth, dim=-1, cumulative=True):
    new_size = []
    for ii in range(len(indices.size())):
        if ii == dim:
            new_size.append(depth)
        new_size.append(indices.size()[ii])
    if dim == -1:
        new_size.append(depth)
    
    out = torch.zeros(new_size)
    indices = torch.unsqueeze(indices, dim)
    out = out.scatter_(dim, indices.data.type(torch.LongTensor), 1.0)
    return Variable(out)

class ProtoNet(nn.Module):
  def __init__(self, encoder):
    """
    Args:
        encoder : CNN encoding the images in sample
        n_way (int): number of classes in a classification task
        n_support (int): number of labeled examples per class in the support set
        n_query (int): number of labeled examples per class in the query set
    """
    super(ProtoNet, self).__init__()
    self.encoder = encoder.cuda()
    #self.encoder=encoder

  def set_forward_loss(self, sample):
    """
    Computes loss, accuracy and output for classification task
    Args:
        sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
    Returns:
        torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    sample_images = sample['images'].cuda()
    #sample_images=sample['images']
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
   
    #target indices are 0 ... n_way-1
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds=target_inds.cuda()
#     print(target_inds)
#     target_inds = target_inds.argmax(1).cpu().numpy()
#     target_inds = torch.argmax(target_inds.argmax(1).cuda(),dim=1)
   
    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
   
#     print(x.shape)
    z = self.encoder.forward(x)
#     print(z.shape)
#     print(z.size(-1))
    z_dim = z.size(-1) #usually 64
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
#     print(z_proto.shape)
    z_query = z[n_way*n_support:]
#     print(z_query.shape)

    #compute distances
    dists = euclidean_dist(z_query, z_proto)
    
    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
   
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
#     print(y_hat)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
#     metric=MulticlassAUROC(num_classes=n_way, average="macro", thresholds=None)
#     auc_roc=metric(y_hat, target_inds.squeeze())
    

    # You need the labels to binarize

    # Binarize y_test with shape (n_samples, n_classes)
#     y_test = label_binarize(y_test, classes=labels)

#     # Binarize ypreds with shape (n_samples, n_classes)
#     ypreds = label_binarize(ypreds, classes=labels)

    y_proba=F.log_softmax(-dists,dim=1).reshape(-1,n_way)
    y_proba=torch.exp(y_proba)
#     auc_roc=roc_auc_score(target_inds.squeeze().reshape(-1).cpu().numpy(),y_proba.cpu().detach().numpy(),average='macro',multi_class='ovo')

    y_train = []
    for i in range(n_way):
        for j in range(n_support):
            y_train.append(i)
    y_train = torch.tensor(y_train).cuda()
    y_train = torch.unsqueeze(y_train, dim=0)

    nClusters = len(np.unique(y_train.data.detach().cpu().numpy()))
    support_labels = torch.arange(0, nClusters).cuda().long()
    temp = []
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/chexpert/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
    for i in support_labels:
      j = n_labels[i]
      loc = df.loc[(df['Abnormality_Label'] == j)]
      loc = loc.values.tolist()
      temp.append(loc[0][1:])
    signatures = np.array(temp)
    signatures = torch.tensor(signatures).cuda()
    signatures = torch.unsqueeze(signatures, dim=0)
    new_loss = torch.cdist(z_proto.float(), signatures.float(), p=2)
    diag = torch.diagonal(new_loss, 0)
    sign_loss = torch.mean(diag)

    # print('Loss Val: ',loss_val.item())
    # print('Signature Loss: ',sign_loss)
    loss = loss_val + 0.5*sign_loss
    return loss, {
        'loss': loss.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat,
#         'auc_roc':auc_roc
        }
  def set_forward_loss_1(self, sample):
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
    

    #target indices are 0 ... n_way-1
    #these are only required for the query samples
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()
    
    y_test = []
    for i in range(n_way):
        for j in range(n_query):
            y_test.append(i)
    y_test = torch.tensor(y_test).cuda()
    y_test = torch.unsqueeze(y_test, dim=0)
    
    prob_test = one_hot(y_test, n_way).cuda()
    prob_test = torch.squeeze(prob_test, dim=0).detach().cpu().numpy()

    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
    
    z = self.encoder.forward(x)
    z_dim = z.size(-1) #usually 64
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]
    
    #compute distances
    dists = euclidean_dist(z_query, z_proto)
    
    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    prob = torch.exp(log_p_y)
    prob = torch.squeeze(prob, dim=1).detach().cpu().numpy()
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    #metrics
    gt = torch.squeeze(y_test, dim=0).detach().cpu().numpy()
    gt_list = list(gt)
    c = len(gt_list)
    pr =  torch.squeeze(y_hat, dim=0)
    pr = torch.squeeze(pr, dim=0).detach().cpu().numpy()
    pr = np.reshape(pr, c)

    y_train = []
    for i in range(n_way):
        for j in range(n_support):
            y_train.append(i)
    y_train = torch.tensor(y_train).cuda()
    y_train = torch.unsqueeze(y_train, dim=0)

    nClusters = len(np.unique(y_train.data.detach().cpu().numpy()))
    support_labels = torch.arange(0, nClusters).cuda().long()
    temp = []
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/chexpert/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
    for i in support_labels:
      j = n_labels[i]
      loc = df.loc[(df['Abnormality_Label'] == j)]
      loc = loc.values.tolist()
      temp.append(loc[0][1:])
    signatures = np.array(temp)
    signatures = torch.tensor(signatures).cuda()
    signatures = torch.unsqueeze(signatures, dim=0)
    new_loss = torch.cdist(z_proto.float(), signatures.float(), p=2)
    diag = torch.diagonal(new_loss, 0)
    sign_loss = torch.mean(diag)
    # print('Loss Val: ',loss_val.item())
    # print('Signature Loss: ',sign_loss)
    loss = loss_val + 0.5*sign_loss
    

    return loss, {
        'loss': loss.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat,
        'gt': gt,
        'pr': pr,
        'y_test':prob_test,
        'y_pred':prob
        }
  def test_shape(self,sample):
    sample_images = sample['images'].cuda()
    #sample_images=sample['images']
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]

    #target indices are 0 ... n_way-1
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds=target_inds.cuda()
    #     target_inds = target_inds.argmax(1).cpu().numpy()
    #     target_inds = torch.argmax(target_inds.argmax(1).cuda(),dim=1)

    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

    z = self.encoder.forward(x)
    z_dim = z.size(-1) #usually 64
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]
    

    #compute distances
    dists = euclidean_dist(z_query, z_proto)

    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    y_proba=F.log_softmax(-dists,dim=1).reshape(-1,n_way)
    y_proba=torch.exp(y_proba)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    x, y_hat = log_p_y.max(2)
    #     print(y_hat.shape(),' ',target_inds))
    print(y_hat)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
#     print(y_hat.reshape(-1).cpu().numpy())
#     print(target_inds.squeeze().reshape(-1).cpu().numpy())
#     for i in range(n_way):
#         for j in range(n_query):
#             print(torch.exp(log_p_y[i][j]))

#     print(y_proba)
    y_proba=F.log_softmax(-dists,dim=1).reshape(-1,n_way)
    y_proba=torch.exp(y_proba)
    auc_roc=roc_auc_score(target_inds.squeeze().reshape(-1).cpu().numpy(),y_proba.cpu().detach().numpy(),average='macro',multi_class='ovo')
    
    

    return loss_val, {
        'loss': loss_val.item(),
        'acc': acc_val.item(),
        'y_hat': y_hat,
        'auc_roc':auc_roc
        }


def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

from tqdm import tqdm_notebook
from tqdm import tnrange

def train(eps, model, optimizer, train_x, train_y, val_x, val_y, n_way, n_support, n_query, max_epoch, epoch_size,path):
#def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
  """
  Trains the four models of protonet
  Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
  """
  model_dict={0:"Protonet",1:"Protonet_SE-Net",2:"Protonet_CBAM",3:"Protonet_ECA"}
  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler=[]
  for i in range(len(model)):
      s = optim.lr_scheduler.StepLR(optimizer[i], 1, gamma=0.5, last_epoch=-1)
      scheduler.append(s)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop
#   flag_nor,flag_senet,flag_cbam,flag_eca=0,0,0,0
  prev_val_acc_nor,prev_val_acc_senet,prev_val_acc_cbam,prev_val_acc_eca=0,0,0,0
  prev_val_loss_nor,prev_val_loss_senet,prev_val_loss_cbam,prev_val_loss_eca=1000,1000,1000,1000
#   prev_val_acc = 0
#   prev_val_loss = 1000
  flag = [0,0,0,0]
  prev_val_loss=[prev_val_loss_nor,prev_val_loss_senet,prev_val_loss_cbam,prev_val_loss_eca]
  prev_val_acc=[prev_val_acc_nor,prev_val_acc_senet,prev_val_acc_cbam,prev_val_acc_eca]
#   train_loss = []
#   train_acc = []
#   val_loss = []
#   val_acc = []
  while stop==False:
    
    running_loss_nor,running_loss_senet,running_loss_cbam,running_loss_eca = 0.0, 0.0, 0.0, 0.0
    running_acc_nor,running_acc_senet,running_acc_cbam,running_acc_eca = 0.0, 0.0, 0.0, 0.0
    running_loss = [running_loss_nor,running_loss_senet,running_loss_cbam,running_loss_eca ]
    running_acc = [running_acc_nor,running_acc_senet,running_acc_cbam,running_acc_eca]
    epoch_loss=[0,0,0,0]
    epoch_acc=[0,0,0,0]
    for episode in range(epoch_size):
    # for episode in tnrange(epoch_size, desc="Epoch {:d} Train".format(epoch+1)):
    #   print(f"Epoch {epoch+1} Train - Episode {episode}")
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
      for i in range(len(model)):
        if flag[i]==1:
            continue
        else:
          optimizer[i].zero_grad()
          loss, output = model[i].set_forward_loss(sample)
          running_loss[i] += output['loss']
          running_acc[i] += output['acc']
          loss.backward()
          optimizer[i].step()
    for i in range(len(model)):
        epoch_loss[i] = running_loss[i] / epoch_size
        epoch_acc[i] = running_acc[i] / epoch_size
#     print(type(epoch_loss))
    for i in range(len(model)):  
        if flag[i]==0:
            print('{}: Epoch {:d} Train -- Loss: {:.4f} Acc: {:.4f}'.format(model_dict[i],epoch+1,epoch_loss[i], epoch_acc[i]))
        
    epoch += 1
#     train_loss.append(epoch_loss)
#     train_acc.append(epoch_acc)
#     print(train_loss, train_acc)
    for i in range(len(model)):
        if flag[i]==0:
            scheduler[i].step()
    
    val_loss_nor,val_loss_senet,val_loss_cbam,val_loss_eca = 0.0, 0.0, 0.0, 0.0
    val_acc_nor,val_acc_senet,val_acc_cbam,val_acc_eca = 0.0, 0.0, 0.0, 0.0
    val_loss = [val_loss_nor,val_loss_senet,val_loss_cbam,val_loss_eca]
    val_acc = [val_acc_nor,val_acc_senet,val_acc_cbam,val_acc_eca]
    avg_loss = [0,0,0,0]
    avg_acc = [0,0,0,0]
    for episode in range(epoch_size):
    # for episode in tnrange(epoch_size, desc="Validation Epoch {:d}".format(epoch)):
      sample = extract_sample(3, 5, 5, val_x, val_y)
      for i in range(len(model)):
        if flag==1:
            continue
        else:
          valdiation_loss, val_output = model[i].set_forward_loss(sample)
          val_loss[i] += val_output['loss']
          val_acc[i] += val_output['acc']
        
    for i in range(len(model)):
        avg_loss[i]= val_loss[i] / epoch_size
        avg_acc[i] = val_acc[i] / epoch_size
    
    for i in range(len(model)):
        if flag[i]==1:
            continue
        else:
            
            if epoch == 1:
                  prev_val_acc[i] = avg_acc[i]
                  prev_val_loss[i] = avg_loss[i]

            if epoch > 1:

                loss_diff = prev_val_loss[i] - avg_loss[i]
                loss_diff = abs(loss_diff)
                if round(loss_diff, 3) < eps or round(loss_diff, 3) == 0.001:
                    if epoch > 5 or epoch == 5:
                        torch.save(model[i],path+'_'+model_dict[i]+'.pth')
                        # torch.save({'epoch': epoch,'model_state_dict': model[i].state_dict(),
                        #             'optimizer_state_dict': optimizer[i].state_dict(),'loss': avg_loss[i],}
                        #              ,path+'_'+model_dict[i]+ '_dict.pth')
                        flag[i] = 1 
                print('{} :Prev Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(model_dict[i],prev_val_loss[i], prev_val_acc[i]))
                print(loss_diff)
                prev_val_acc[i] = avg_acc[i]
                prev_val_loss[i] = avg_loss[i]

                print('{} :Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(model_dict[i],avg_loss[i], avg_acc[i]))
                print("\n")
#     print(type(avg_loss))
#     val_loss.append(avg_loss)
#     val_acc.append(avg_acc)
#     print(val_loss, val_acc)
    c=0
    for i in range(len(flag)):
        if(flag[i]==1):
            c+=1
    if c==4:
        stop=True

print(Xtrain.shape, ytrain.shape)

print(Xval.shape, yval.shape)


# optimizer_nor = optim.Adam(model_nor_5.parameters(), lr = 0.001)
# optimizer_se = optim.Adam(model_se_5.parameters(), lr = 0.001)
# optimizer_cbam = optim.Adam(model_cbam_5.parameters(), lr = 0.001)
# optimizer_eca = optim.Adam(model_eca_5.parameters(), lr = 0.001)

# model = [model_nor_5,model_se_5,model_cbam_5,model_eca_5]
# optimizer = [optimizer_nor, optimizer_se, optimizer_cbam, optimizer_eca]





from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_auc_score
import subprocess
def test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_ls):
  gt=[]
  pr=[]
  running_loss = 0.0
  running_acc = 0.0
  model = torch.load(temp_str + '.pth')
  for episode in range(epoch_size):
#   for episode in tnrange(test_episode):
    sample = extract_sample_test(test_ls,n_way, n_support, n_query, test_x, test_y)
    loss, output = model.set_forward_loss_1(sample)
    running_loss += output['loss']
    running_acc += output['acc']
    for i in output['gt']:
      gt.append(i)
    for i in output['pr']:
      pr.append(i)
    # print("y_test: ", output['y_test'], "\nprob: ", output['y_pred'])
    if lock == 1:
      y_test = output['y_test']
      prob = output['y_pred']
      lock = False
    else:
      y_test = np.concatenate((y_test, output['y_test']), axis=0)
      prob = np.concatenate((prob, output['y_pred']), axis=0)
    
  avg_loss = running_loss / test_episode
  avg_acc = running_acc / test_episode
  print('Test results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  precision, recall, fscore, support = score(gt, pr)
  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))
  # print("gt: ", gt, "\npr: ", pr)
  # print("y_test: ", y_test,"\nprob: ", prob)
  # print("y_test: ", y_test.shape(),"\nprob: ", prob.shape())
  auroc_values = []
  # labels = ['No Finding', 'Pleural_Thickening', 'Pneumothorax'] 
  for i in range(n_way):
      auroc_values.append(roc_auc_score(y_test[:,i], prob[:,i], multi_class='ovo'))
  c = 0
  for i in test_ls:
    print("AUROC for ",i,": ", auroc_values[c])
    c += 1
    
  return auroc_values[0], auroc_values[1], auroc_values[2],precision,recall,fscore

path = "/home/maharathy1/MTP/implementation/protonet_attention_signature_loss/models/split1/half_sign_loss/"

for i in range(10):
    print("Model: ",i + 1)
    model_nor_5=load_protonet_conv(x_dim=(3,128,128),
        hid_dim=128,
        z_dim=768,)
    model_se_5=load_protonet_conv_se(x_dim=(3,128,128),
        hid_dim=128,
        z_dim=768,)
    model_cbam_5=load_protonet_conv_cbam(x_dim=(3,128,128),
        hid_dim=128,
        z_dim=768,)
    model_eca_5=load_protonet_conv_eca(x_dim=(3,128,128),
        hid_dim=128,
        z_dim=768,)
    optimizer_nor = optim.Adam(model_nor_5.parameters(), lr = 0.001)
    optimizer_se = optim.Adam(model_se_5.parameters(), lr = 0.001)
    optimizer_cbam = optim.Adam(model_cbam_5.parameters(), lr = 0.001)
    optimizer_eca = optim.Adam(model_eca_5.parameters(), lr = 0.001)

    model = [model_nor_5,model_se_5,model_cbam_5,model_eca_5]
    optimizer = [optimizer_nor, optimizer_se, optimizer_cbam, optimizer_eca]
    n_way = 8
    n_support = 5
    n_query = 5

    n_way_val = 3
    n_support_val = 5
    n_query_val = 1

    train_x = Xtrain
    train_y = ytrain

    val_x=Xval
    val_y=yval

    max_epoch = 2
    epoch_size = 500
    temp_str = path + 'chexpert_pwas_split1_128*128_' + str(i)
    train(0.005, model, optimizer, train_x, train_y, val_x,val_y, n_way,n_support,n_query,max_epoch, epoch_size, temp_str)

test_labels=np.unique(ytest)
print(test_labels)

# test_labels = df_mod_test['Label'].unique()

test_labels = test_labels.tolist()
print(test_labels)

model_dict={0:"Protonet",1:"Protonet_SE-Net",2:"Protonet_CBAM",3:"Protonet_ECA"}

for k in range(len(model_dict)):
    list_1 = []
    list_2 = []
    list_3 = []
    precision = []
    recall = []
    f1score = []
    for i in range(10):
        ab_1 = 0
        ab_2 = 0
        ab_3 = 0
        p = []
        r = []
        f = []
        n_way = 3
        n_support = 5
        n_query = 1
        test_x = Xtest
        test_y = ytest
        test_episode = 1000
        lock = 1
        temp_str = path + 'chexpert_pwas_split1_128*128_' + str(i) + '_'+model_dict[k]
        ab_1, ab_2, ab_3, p, r, f = test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_labels)
        list_1.append(ab_1)
        list_2.append(ab_2)
        list_3.append(ab_3)
        precision.append(p)
        recall.append(r)
        f1score.append(f)
    arr_1 = np.array(list_1)
    arr_2 = np.array(list_2)
    arr_3 = np.array(list_3)
    pre = np.array(precision)
    re = np.array(recall)
    f1 = np.array(f1score)
    mean_pre = np.round(np.mean(pre, axis=0), decimals=4)
    mean_re = np.round(np.mean(re, axis=0), decimals=4)
    mean_f1 = np.round(np.mean(f1, axis=0), decimals=4)
    std_pre = np.round(np.std(pre, axis=0), decimals=4)
    std_re = np.round(np.std(re, axis=0), decimals=4)
    std_f1 = np.round(np.std(f1, axis=0), decimals=4)
    mean_auroc = []
    mean_auroc.append(np.round(np.mean(arr_1),4))
    mean_auroc.append(np.round(np.mean(arr_2),4))
    mean_auroc.append(np.round(np.mean(arr_3),4))
    std_auroc = []
    std_auroc.append(np.round(np.std(arr_1),4))
    std_auroc.append(np.round(np.std(arr_2),4))
    std_auroc.append(np.round(np.std(arr_3),4))
    print(f'For {model_dict[k]}\n')
    
    print("Mean AUROC of ",test_labels[0],": ", mean_auroc[0],"\n Std AUROC of ",test_labels[0],":", std_auroc[0])
    print("Mean AUROC of ",test_labels[1],": ", mean_auroc[1],"\n Std AUROC of ",test_labels[1],":", std_auroc[1])
    print("Mean AUROC of ",test_labels[2],": ", mean_auroc[2],"\n Std AUROC of ",test_labels[2],":", std_auroc[2])
    data = {
        'mean_pre': mean_pre,
        'std_pre': std_pre,
        'mean_re': mean_re,
        'std_re': std_re,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'mean_auroc': mean_auroc,
        'std_auroc': std_auroc
        }
    df = pd.DataFrame(data, index=test_labels)
    df.to_csv(path+'output'+'_'+model_dict[k]+'.csv')

