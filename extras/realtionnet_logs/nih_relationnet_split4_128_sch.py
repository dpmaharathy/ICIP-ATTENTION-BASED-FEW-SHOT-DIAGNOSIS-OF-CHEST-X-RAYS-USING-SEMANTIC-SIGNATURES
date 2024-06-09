# -*- coding: utf-8 -*-
"""relationnet.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rDq3ww-ylLydh7Tde6GLadYZ64S76yd4
"""

import torch
torch.cuda.is_available()

import pandas as pd
import numpy as np
from array import *
#from PIL import Image
import cv2
import os
import random


print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/X_train_4.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/y_train_4.npy"
X_train = np.load(path_x)
y_train = np.load(path_y)

print(np.unique(y_train,return_counts=True))


print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/X_val_4.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/y_val_4.npy"
X_val = np.load(path_x)
y_val = np.load(path_y)

print(np.unique(y_val,return_counts=True))


print("Time to load train arrays")
path_x = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/X_test_4.npy"
path_y = "/home/maharathy1/MTP/implementation/MIA/npy_dataset_128/nih_128/y_test_4.npy"
X_test = np.load(path_x)
y_test = np.load(path_y)

print(np.unique(y_test,return_counts=True))

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

def extract_sample(n_way, n_support, n_query, datax, datay):
  sample = []
  label = []
  K = np.random.choice(np.unique(datay), n_way, replace=False)
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
#   print(sample.size())
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label
      })

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
#   print(sample.size())
  #after permute: torch.Size([3, 6, 3, 64, 64])
  return({
      'images': sample,
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query,
      'n_label': label
      })

"""## **Required Methods**"""

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

"""## **Model**"""

torch.cuda.empty_cache()

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

import torch.nn.init as init

def load_protonet_conv(**kwargs):
    x_dim = kwargs['x_dim']
    hid_dim = kwargs['hid_dim']
    z_dim = kwargs['z_dim']

    def conv_block_1(in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ]
        
        # Apply Xavier initialization to the Conv2d layer weights
        init.xavier_normal_(layers[0].weight)
        init.constant_(layers[0].bias, 0)  # Optional: Initialize biases to zeros
        
        return nn.Sequential(*layers)

    def conv_block_2(in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels, momentum=1, affine=True),
            nn.LeakyReLU()
        ]
        
        # Apply Xavier initialization to the Conv2d layer weights
        init.xavier_normal_(layers[0].weight)
        init.constant_(layers[0].bias, 0)  # Optional: Initialize biases to zeros
        
        return nn.Sequential(*layers)

    encoder = nn.Sequential(
        conv_block_1(x_dim[0], hid_dim),
        conv_block_1(hid_dim, hid_dim),
        conv_block_2(hid_dim, hid_dim),
        conv_block_2(hid_dim, z_dim)
    )

    return ProtoNet(encoder)


import torch.nn.init as init

class RelationNet(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        layers= []
        layers += [nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]
        layers += [nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2)]

        self.layer1 = nn.Sequential(*layers)
        layers= []
        layers += [nn.Linear(64*8*8,64), nn.ReLU()]

        # Xavier Initialization for Linear Layers
        init.xavier_normal_(layers[0].weight)
        init.constant_(layers[0].bias, 0)  # Optional: You can initialize biases to zeros
        layers += [nn.Linear(64,1), nn.Sigmoid()]

        self.layer2 = nn.Sequential(*layers)

    def forward(self, x):
        x=x.view(-1,128,32,32)
        x=self.layer1(x)
        x = x.view(x.size(0),-1)
        return self.layer2(x)

class ProtoNet(nn.Module):
  def __init__(self, encoder):
    super(ProtoNet, self).__init__()
    self.encoder = encoder.cuda()


  def set_forward_loss(self, sample, RN, MSE):
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]


    target_inds = []
    for i in range(n_way):
      for j in range(n_query):
        target_inds.append(i)
    target_inds = torch.Tensor(target_inds)
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()

    targets = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    targets = Variable(targets, requires_grad=False)
    targets = targets.cuda()


    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

    z = self.encoder.forward(x)
  
    z_support = z[:n_way*n_support]
    z_query = z[n_way*n_support:]

    z_support = z_support.reshape(n_way, n_support, 64, 32, 32).sum(1).unsqueeze(0).repeat(n_way*n_query,1,1,1,1)
    # print('Z support size: ',z_support.size())
    z_query = z_query.unsqueeze(0).repeat(n_way,1,1,1,1).transpose(0,1)
    # print('Z query: ',z_query.size())
    z_concat = torch.cat((z_support, z_query), 2)
    # print(z_concat.size())
    score=RN(z_concat)
    # print(score.size())
    score=score.reshape(n_way*n_query,n_way)
    # score = RN(z_concat).reshape(n_way*n_query, n_way)
    # print("score: ",score)
    prob = nn.Softmax(dim=1)(score)
#     print("prob_size: ",prob.size(),"\nprob: ", prob)
    y_hat = torch.argmax(prob,dim=1)
#     print("y_hat: ",y_hat)
    y_hat = y_hat.view(n_way,-1)
#     print("y_hat_new: ",y_hat)
#     print("y_hat: ", y_hat.size(),"\ny_hat: ", y_hat)

    y_test = []
    for i in range(n_way):
        for j in range(n_query):
            y_test.append(i)
    y_test = torch.tensor(y_test).cuda()
    y_test = torch.unsqueeze(y_test, dim=0)

    prob_test = one_hot(y_test, n_way)
    prob_test = torch.squeeze(prob_test, dim=0).cuda()
    # print("prob_test: ", prob_test.size(),"\nprob_test: ", prob_test)

    loss = MSE(score, prob_test)
    acc = torch.eq(y_hat, targets.squeeze()).float().mean()

    prob_test = prob_test.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()

    #metrics
    gt = torch.squeeze(y_test, dim=0).detach().cpu().numpy()
    gt_list = list(gt)
    c = len(gt_list)
    pr =  torch.squeeze(y_hat, dim=0)
    pr = torch.squeeze(pr, dim=0).detach().cpu().numpy()
    pr = np.reshape(pr, c)


    return loss, {
        'loss': loss.item(),
        'acc': acc.item(),
        'y_hat': y_hat,
        'gt': gt,
        'pr': pr,
        'y_test':prob_test,
        'y_pred':prob
        }


  def set_forward_loss_1(self, sample, RN, MSE):
    sample_images = sample['images'].cuda()
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    n_labels = sample['n_label']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]


    target_inds = []
    for i in range(n_way):
      for j in range(n_query):
        target_inds.append(i)
    target_inds = torch.Tensor(target_inds)
    target_inds = Variable(target_inds, requires_grad=False)
    target_inds = target_inds.cuda()

    targets = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    targets = Variable(targets, requires_grad=False)
    targets = targets.cuda()


    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                   x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

    z = self.encoder.forward(x)

    z_support = z[:n_way*n_support]
    z_query = z[n_way*n_support:]

    z_support = z[:n_way*n_support].reshape(n_way, n_support, 64,32,32).sum(1).unsqueeze(0).repeat(n_way*n_query,1,1,1,1)
    z_query = z[n_way*n_support:].unsqueeze(0).repeat(n_way,1,1,1,1).transpose(0,1)
    # print('Z support: ',z_support.size())
    z_concat = torch.cat((z_support, z_query), 2)

    score = RN(z_concat).reshape(n_way*n_query, n_way)
    
    prob = nn.Softmax(dim=1)(score)
    y_hat = torch.argmax(prob,dim=1)
    y_hat = y_hat.view(n_way,-1)

    y_test = []
    for i in range(n_way):
        for j in range(n_query):
            y_test.append(i)
    y_test = torch.tensor(y_test).cuda()
    y_test = torch.unsqueeze(y_test, dim=0)

    prob_test = one_hot(y_test, n_way)
    prob_test = torch.squeeze(prob_test, dim=0).cuda()
#     print("prob_test: ", prob_test.size())

    loss = MSE(score, prob_test)

    prob_test = prob_test.detach().cpu().numpy()
    prob = prob.detach().cpu().numpy()
    acc = torch.eq(y_hat, targets.squeeze()).float().mean()

    #metrics
    gt = torch.squeeze(y_test, dim=0).detach().cpu().numpy()
    gt_list = list(gt)
    c = len(gt_list)
    pr =  torch.squeeze(y_hat, dim=0)
    pr = torch.squeeze(pr, dim=0).detach().cpu().numpy()
    pr = np.reshape(pr, c)


    return loss, {
        'loss': loss.item(),
        'acc': acc.item(),
        'y_hat': y_hat,
        'gt': gt,
        'pr': pr,
        'y_test':prob_test,
        'y_pred':prob
        }

"""## **Training**"""

from tqdm import tqdm_notebook
from tqdm import trange

train_loss = []
train_acc = []
test_loss = []
test_acc = []

def train(eps, model,optimizer, train_x, train_y, val_x, val_y, n_way, n_support, n_query, max_epoch, epoch_size, temp_str):
# def train(model, train_x, train_y, val_x, val_y, n_way, n_support, n_query, max_epoch, epoch_size, temp_str):


  Emb = model
  RN = RelationNet().cuda()
  MSE = nn.MSELoss().cuda()

  Emb_optim = torch.optim.Adam(Emb.parameters(), lr=3e-4)

  RN_optim = torch.optim.Adam(RN.parameters(),  lr=3e-4)

  Emb_scheduler = optim.lr_scheduler.StepLR(Emb_optim, step_size=5, gamma=0.5, last_epoch=-1)
  RN_scheduler = optim.lr_scheduler.StepLR(RN_optim, step_size=5, gamma=0.5, last_epoch=-1)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop
  prev_val_acc = 0
  prev_val_loss = 1000
  flag = 0

  while flag != 1:
    running_loss = 0.0
    running_acc = 0.0
    for episode in trange(epoch_size, desc="Epoch {:d} Train".format(epoch+1)):
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y)

      Emb.zero_grad()
      RN.zero_grad()
    #   optimizer.zero_grad()
    #   RN_optim.zero_grad()
      loss, output = Emb.set_forward_loss(sample, RN, MSE)
      running_loss += output['loss']
      running_acc += output['acc']

      loss.backward()

      Emb_optim.step()
      RN_optim.step()

    epoch_loss = running_loss / epoch_size
    epoch_acc = running_acc / epoch_size

    print('Epoch {:d} Train -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc))
    x = epoch_loss
    y = epoch_acc
    epoch += 1
    train_loss.append(round(x,4))
    train_acc.append(round(y,4))

    val_loss = 0.0
    val_acc = 0.0
    for episode in trange(epoch_size, desc="Validation Epoch {:d}".format(epoch)):
      sample = extract_sample(3, 5, 5, val_x, val_y)
      valdiation_loss, val_output = Emb.set_forward_loss(sample, RN, MSE)
      val_loss += val_output['loss']
      val_acc += val_output['acc']
    avg_loss = val_loss / epoch_size
    avg_acc = val_acc / epoch_size
    a = avg_loss
    b = avg_acc
    test_loss.append(round(a,4))
    test_acc.append(round(b,4))

    if epoch == 1:
          prev_val_acc = avg_acc
          prev_val_loss = avg_loss

    if epoch > 1:
        loss_diff = prev_val_loss - avg_loss
        loss_diff = abs(loss_diff)
        if round(loss_diff, 3) < eps or round(loss_diff, 3) == eps:
            if epoch > 5 or epoch == 5:
                torch.save(Emb,temp_str+'_Emb_' + '.pth')
                torch.save(RN,temp_str+ '_RN_'+ '.pth')
                flag = 1
        print('Prev Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(prev_val_loss, prev_val_acc))
        print(loss_diff)
        prev_val_acc = avg_acc
        prev_val_loss = avg_loss

    '''
    if epoch == max_epoch:
        torch.save(Emb,'best_model_path_emb_' + temp_str + '.pth')
        torch.save(RN,'best_model_path_rn_' + temp_str + '.pth')
        flag = 1
    '''
    print('Val Results -- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
    print("\n")
    Emb_scheduler.step()
    RN_scheduler.step()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_fscore_support as score

"""## **Testing**"""

def test(temp_str, test_x, test_y, n_way, n_support, n_query, test_episode, lock, test_ls):
  gt=[]
  pr=[]

  MSE = nn.MSELoss().cuda()
  running_loss = 0.0
  running_acc = 0.0
  Emb = torch.load( temp_str+'_Emb_' + '.pth')
  RN = torch.load( temp_str+'_RN_' + '.pth')
  for episode in trange(test_episode):
    sample = extract_sample_test(test_ls, n_way, n_support, n_query, test_x, test_y)
    loss, output = Emb.set_forward_loss_1(sample, RN, MSE)
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

  auroc_values = []
  for i in range(n_way):
      auroc_values.append(roc_auc_score(y_test[:,i], prob[:,i], multi_class='ovo'))
  c = 0

  return auroc_values[0], auroc_values[1], auroc_values[2], precision, recall, fscore

path='/home/maharathy1/MTP/implementation/relationnet/models/split4/'

for i in range(10):
    print("Model: ", i + 1)
    model = load_protonet_conv(
        x_dim=(3,128,128),
        hid_dim=64,
        z_dim=64,
        )
    optimizer = optim.Adam(model.parameters(), lr = 3e-4)
    n_way = 9
    n_support = 5
    n_query = 5
    X = X_train
    y = y_train
    X_val = X_val
    y_val = y_val
    max_epoch = 20
    epoch_size = 500
    temp_str = path+'nih_relationnet_split4_sch' + str(i)
    train(0.005, model,optimizer, X, y, X_val,y_val, n_way, n_support, n_query, max_epoch, epoch_size,temp_str)
#     train(model, X, y, X_val,y_val, n_way, n_support, n_query, max_epoch, epoch_size, temp_str)

"""## **Model**"""

test_labels=np.unique(y_test)
print(type(test_labels))

list_1 = []
list_2 = []
list_3 = []
precision = []
recall = []
f1score = []
for j in range(10):
    print("Best Model: ", j + 1)
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
    test_episode = 1000
    lock = 1
    temp_str = path+'nih_relationnet_split4_sch' + str(j)
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

print(arr_1, arr_2, arr_3)

print(pre, "\n", re,"\n", f1)

mean_pre = np.round(np.mean(pre, axis=0), decimals=4)
mean_re = np.round(np.mean(re, axis=0), decimals=4)
mean_f1 = np.round(np.mean(f1, axis=0), decimals=4)

print(mean_pre, "\n", mean_re,"\n", mean_f1)

std_pre = np.round(np.std(pre, axis=0), decimals=4)
std_re = np.round(np.std(re, axis=0), decimals=4)
std_f1 = np.round(np.std(f1, axis=0), decimals=4)

print(std_pre, "\n", std_re,"\n", std_f1)

mean_auroc = []
mean_auroc.append(np.round(np.mean(arr_1),4))
mean_auroc.append(np.round(np.mean(arr_2),4))
mean_auroc.append(np.round(np.mean(arr_3),4))

std_auroc = []
std_auroc.append(np.round(np.std(arr_1),4))
std_auroc.append(np.round(np.std(arr_2),4))
std_auroc.append(np.round(np.std(arr_3),4))

print(mean_auroc, "\n", std_auroc)

print("Mean AUROC of ",test_labels[0],": ", mean_auroc[0],"\n Std AUROC of ",test_labels[0],":", std_auroc[0])
print("Mean AUROC of ",test_labels[1],": ", mean_auroc[1],"\n Std AUROC of ",test_labels[1],":", std_auroc[1])
print("Mean AUROC of ",test_labels[2],": ", mean_auroc[2],"\n Std AUROC of ",test_labels[2],":", std_auroc[2])

print(train_loss, train_acc)
print(test_loss, test_acc)

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

# Create the dataframe
df = pd.DataFrame(data, index=test_labels)

df.to_csv(path+'output_relationnet_split4_sch.csv')

data_text = {
    'mean_pre': mean_pre,
    'std_pre': std_pre,
    'mean_re': mean_re,
    'std_re': std_re,
    'mean_f1': mean_f1,
    'std_f1': std_f1,
    'mean_auroc': mean_auroc,
    'std_auroc': std_auroc,
    'training_loss': train_loss,
    'training_acc': train_acc,
    'testing_loss': test_loss,
    'testing_acc': test_acc
}

with open(path+'final_results.txt', 'w') as f:
    for key, value in data_text.items():
        f.write(f'{key}: {value}\n')

# # plotting code
# import matplotlib.pyplot as plt
# e = list(range(0, len(train_acc)))
# print("Figure - 1")
# plt.title('Model Performance 1')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.plot(e,train_acc,label = "Training Accuracy")
# plt.plot(e,test_acc,label = "Validation Accuracy")
# plt.legend()
# plt.show()
# plt.savefig('acc.png')
# print("\n")
# print("Figure - 2")
# plt.title('Model Performance 2')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.plot(e,train_loss,label = "Training Loss")
# plt.plot(e,test_loss,label = "Validation Loss")
# plt.legend()
# plt.show()
# plt.savefig('loss.png')

# !zip -r output.zip ./