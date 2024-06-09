import torch
torch.cuda.is_available()
from tqdm import tqdm_notebook
from tqdm import trange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score
import pandas as pd
import numpy as np
from array import *
#from PIL import Image
import cv2
import os
import random
print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/X_test_5.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/y_test_5.npy"
X_test = np.load(path_x)
y_test = np.load(path_y)
np.unique(y_test,return_counts=True)
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision


def extract_sample_test(K, n_way, n_support, n_query, datax, datay):

  sample = []
  label = []
  #K = np.choice(np.unique(datay), n_way, replace=False)
  #K = df_mod_test['Finding Labels'].unique()
  for cls in K:
    #datax_cls represents the images data 64*64*3 corresponding to a class
    datax_cls = datax[datay == cls]
    #random permutations on datax_cls
    perm = np.random.permutation(datax_cls)
    #selects n_support + n_query
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
    for i in range(n_support+n_query):
        label.append(cls)
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  #sample size is: torch.Size([3, 6, 64, 64, 3])
  sample = sample.permute(0,1,4,2,3)
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label
      })
def display_sample(sample):
    # Need 4D tensor to create a grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    # Make a grid with a single row (horizontal grid)
    out = torchvision.utils.make_grid(sample_4D, nrow=sample_4D.shape[0])
    plt.figure(figsize=(20, 16))
    plt.imshow(out.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("/home/maharathy1/MTP/implementation/protonet_attention_signature_loss/images/nih_1.pdf")

#All required functions to make test function run.
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
class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

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
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/nih/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
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
    df = pd.read_csv("/home/maharathy1/MTP/implementation/MIA/embeddings/nih/Sign_7-dmis-lab-biobert-v1_1_mean.csv")
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
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)
test_labels = np.unique(y_test)
print(type(test_labels))
test_labels = test_labels.tolist()
print(type(test_labels))
print(test_labels)
path = '/home/maharathy1/MTP/implementation/protonet_attention_signature_loss/models/nih/split5/'
def test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_ls):
  gt=[]
  pr=[]
  running_loss = 0.0
  running_acc = 0.0
  model = torch.load(temp_str + '.pth')
  for episode in trange(test_episode):
    sample = extract_sample_test(test_ls, n_way, n_support, n_query, test_x, test_y)
    sample_images = sample['images']
    print(sample['images'].shape)
    print(n_support, n_query)
    x_query = sample_images[:, n_support:]
    print(x_query.shape)
    print(sample['n_label'])
    display_sample(x_query)
    loss, output = model.set_forward_loss_1(sample)
    running_loss += output['loss']
    running_acc += output['acc']
    for i in output['gt']:
      gt.append(i)
    for i in output['pr']:
      pr.append(i)
    print("gt: ", gt, "\npr: ", pr)
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
  for i in range(n_way):
      auroc_values.append(roc_auc_score(y_test[:,i], prob[:,i], multi_class='ovo'))
  c = 0
  for i in test_ls:
    print("AUROC for ",i,": ", auroc_values[c])
    c += 1
    
  return auroc_values[0], auroc_values[1], auroc_values[2], precision, recall, fscore
list_1 = []
list_2 = []
list_3 = []
precision = []
recall = []
f1score = []
j = 2
print("Best Model: ", j)
ab_1 = 0
ab_2 = 0
ab_3 = 0
p = []
r = []
f = []
n_way = 3
n_support = 5
n_query = 1
test_x = X_test
test_y = y_test
test_episode = 1
lock = 1
temp_str = path +'nih_pwas_split5_128*128_1_Protonet_ECA'
ab_1, ab_2, ab_3, p, r, f = test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_labels)
print(test_labels[0],": ", ab_1)
print(test_labels[1],": ", ab_2)
print(test_labels[2],": ", ab_3)
print("\n")
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