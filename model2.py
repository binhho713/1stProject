import argparse
import os, sys
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch

from GMM import *
from encoder import *

batch_size = 64
latent_dim = 100
n_classes = 10
img_shape = (1, 32, 32)
img_size = 32
nums = 10
use_cuda = torch.cuda.is_available()
is_training = 1

#Init encoder
shifting_encoder = MLP_shifting()
std_encoder = MLP_std()
var_encoder = MLP_var()
pred = MLP()

if use_cuda:
    shifting_encoder = shifting_encoder.cuda()
    std_encoder = std_encoder.cuda()
    var_encoder = var_encoder.cuda()
    pred = pred.cuda()
    print("use_cuda")

#Load data

import scipy.io

data = scipy.io.loadmat('mix_ts_10_GMM.mat')
data = np.concatenate((data['mix_ts'], data['labels']),axis=1)
np.random.shuffle(data)
time_series, truth_labels_list = np.split(data, (1024,), axis=1)
n_timeSeries = len(time_series)

stp = scipy.io.loadmat('stick_patterns.mat')
peak = np.tile(stp['peak'], (batch_size, 1, 1, 1)) # bs, ts, 200, 1
mu = np.tile(stp['mu'], (batch_size, 1, 1, 1)) # bs, ts, 200, 1
peak = torch.tensor(peak).float().cuda()
mu = torch.tensor(mu).float().cuda()
"""
testdata = scipy.io.loadmat('test_ts.mat')
testdata = np.concatenate((testdata['mix_ts'], testdata['labels']),axis=1)
np.random.shuffle(testdata)
test_time_series, test_labels_list = np.split(testdata, (1024,), axis=1)
n_ts = len(test_time_series)
"""
"""
ts1 = np.load("./4by4.npy") #10000, 16, 1, 32, 32
ts2 = np.load("./4by4_5678.npy") #10000, 16, 1, 32, 32
time_series = np.maximum(ts1, ts2) #10000, 16, 1, 32, 32
n_timeSeries = len(time_series)

ls1 = np.load("./4by4_labels.npy") #10000, 16
ls2 = np.load("./4by4_5678_labels.npy") #10000, 16
truth_labels_list = np.concatenate((ls1.reshape(-1, 16, 1), ls2.reshape(-1, 16, 1)), axis = 2) #100, 16, 2
"""
gen_labels = torch.LongTensor(np.arange(nums * batch_size, dtype = "int32") % nums)
if use_cuda:
    gen_labels =  gen_labels.cuda()

#Entropy function
def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim=1), dim=0)

#Function to find k lagerst numbers
def k_lagerst(input, k):
    z = np.argsort(input, axis = 1)
    output = np.ndarray(shape = (len(input),k))
    for i in range(len(input)):
        for j in range(k):
            output[i][-j-1] = z[i][-j-1]
    return output

def normalize(input, dim):
    min = torch.min(input, dim= dim)[0]
    max = torch.max(input, dim= dim)[0]
    x = (input.view(-1, 1024) - min)/(max - min)
    return x

def KL_dis(target_a, target_b):
    target_a += 1e-7
    P = torch.transpose(torch.transpose(target_a, 0, 1) / (torch.sum(target_a, dim=1)[0] + 1e-7), 0, 1)
    target_b += 1e-7
    Q = torch.transpose(torch.transpose(target_b, 0, 1) / (torch.sum(target_b, dim=1)[0] + 1e-7), 0, 1)
    x = P * torch.log(P / Q)
    return torch.sum(x, dim=1)[0]

def JS_dis(input_xrd, xrd_prime):
    print(KL_dis(input_xrd, xrd_prime), KL_dis(xrd_prime, input_xrd))
    return (0.5 * KL_dis(input_xrd, xrd_prime) + 0.5 * KL_dis(xrd_prime, input_xrd))

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

pred_path = "./models/pred_ts_m2.model"
var_path = "./models/var_encoder_ts.model"
std_path = "./models/std_encoder_ts.model"
shifting_path = "./models/shifting_encoder_ts.model"

check_freq = 100
lr = 0.00001

if is_training:
#----------
# Training
#----------
    for _epoch_ in range(10000):


        ts_mix_lst = []
        labels_list = []
        running_label_acc = 0
        running_loss = 0
        k_loss = 0
        cnt = 0

        for idx in range(n_timeSeries):

            ts_mix = time_series[idx] #1, 1024
            ts_mix_lst.append(ts_mix)

            truth_labels = truth_labels_list[idx] #1, 2
            labels_list.append(truth_labels)

            if (len(ts_mix_lst) == batch_size):
                ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs * 1024
                ts_mix = Variable(torch.tensor(ts_mix.reshape(-1, 1, 1024)).float(), requires_grad=False) # bs, 1024
                if use_cuda:
                    ts_mix = ts_mix.cuda()

                if(use_cuda):
                    labels_distribution = pred(ts_mix).cuda()  # bs, 10
                    c = std_encoder(ts_mix).cuda()  # bs, 1, 10, 100
                    shift_ratio = shifting_encoder(ts_mix).cuda()  # bs, 1, 10, 100
                    shift_var = var_encoder(ts_mix).cuda()  # bs, 1, 10, 100

                else:
                    labels_distribution = pred(ts_mix)
                    c = std_encoder(ts_mix)
                    shift_ratio = shifting_encoder(ts_mix)
                    shift_var = var_encoder(ts_mix)

                optimizer = torch.optim.Adam(list(pred.parameters()) + list(std_encoder.parameters()) + list(shifting_encoder.parameters())
                                             + list(var_encoder.parameters()), lr=lr)

                labels = labels_distribution.cpu().data.numpy()
                labels = k_lagerst(labels, 2)
                ll = np.reshape(labels_list,(-1, 2))  #bs, 2
                eqn = []
                for i in range(batch_size):
                    h = []
                    for j in range(2):
                        if labels[i][j] in ll[i]:
                            h.append(1)
                        else:
                            h.append(0)
                    eqn.append(h)

                label_acc = (np.sum(eqn)/(batch_size * 2)).astype("float32")

                #generate image
                #haha = z.view(-1, 100)
                peak_ts = peak * shift_var
                mu_ts = (mu.view(-1, 200, 1).permute(1, 2, 0) * shift_ratio.view(-1)).permute(2, 0, 1).view(batch_size, 10, 200, 1)
                gen_ts = GMM(peak_ts, mu_ts, c) #bs*10, 1, 1024

                #arr1 = gen_img.detach().cpu().numpy()
                gen_ts = gen_ts.view(-1, 1, 1024).permute(1, 2, 0) * labels_distribution.view(-1)#1,1024,bs*10
                gen_ts = gen_ts.permute(2, 0, 1).view(batch_size,10,1,1024) # bs, 10, 1, 1024
                #gen_img = gen_img.permute(1, 2, 0) * labels_distribution.view(-1)  # 1,1024,bs*10
                #gen_img = gen_img.permute(2, 0, 1).view(batch_size,10,1,1024)
                #gen_mix = torch.FloatTensor(size=(64, 1, 1024)).cuda()
                #for i in range(batch_size):
                    #gen_mix[i] = gen_img[i][int(labels[i][0])] + gen_img[i][int(labels[i][1])]
                gen_ts = torch.sum(gen_ts, dim= 1) #bs, 1, 1024
                #arr4 = gen_mix.detach().cpu().numpy()
                #arr2 = ts_mix.detach().cpu().numpy()
                #arr3 = labels_distribution.detach().cpu().numpy()

                #reconstruct loss
                loss_rec = 0
                loss_jsd = 0
                cri = torch.nn.MSELoss()
                loss_rec = cri(ts_mix, gen_ts)
                loss_jsd = JS_dis(normalize(ts_mix, dim=2), normalize(gen_ts, dim=2))
                #loss_rec /= (1.0 * labels_distribution.size(0))

                #k_sparity constrain
                c = np.log(2) - 1e-6
                c = torch.tensor(c).float()
                k_sparity = torch.maximum(entropy(labels_distribution),c).float()

                #scale_recon = 0.001

                loss = 0.05*loss_rec + loss_jsd + 0.1*k_sparity
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                ts_mix_lst = []
                labels_list = []
                running_label_acc += label_acc
                running_loss += loss_rec.item()
                k_loss += k_sparity.item()
                cnt += 1

                #save model
                if (cnt % check_freq == 0):
                    print("#epoch = %d, data = %d, running_label_acc = %f, running_loss = %f, k_loss= %f" % (
                    _epoch_, cnt * batch_size, running_label_acc / cnt, running_loss / cnt, k_loss/cnt))

                    save_model(pred, pred_path)
                    save_model(shifting_encoder, shifting_path)
                    save_model(var_encoder, var_path)
                    save_model(std_encoder, std_path)

                    running_label_acc = 0
                    running_loss = 0
                    cnt = 0

else:
# ----------
# Testing
# ----------
    sep.load_state_dict(torch.load("models/sep_tsv2.model"))
    sep.eval()
    pred.load_state_dict(torch.load("models/pred_tsv2.model"))
    pred.eval()

    ts_mix_lst = []
    labels_list = []
    d = 0
    for idx in range(n_ts):

        ts_mix = test_time_series[idx]  # 1, 1024
        ts_mix_lst.append(ts_mix)

        test_labels = test_labels_list[idx]  # 1, 2
        labels_list.append(test_labels)

        if (len(ts_mix_lst) == batch_size):

            d += batch_size

            ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs * 1024
            ts_mix = Variable(torch.tensor(ts_mix.reshape(-1, 1, 1024)).float(), requires_grad=False)

            if use_cuda:
                ts_mix = ts_mix.cuda()

            labels_distribution = pred(ts_mix)

            pred_labels = labels_distribution.cpu().data.numpy()
            pred_labels = k_lagerst(pred_labels, 2)
            ll = np.reshape(labels_list, (-1, 2))  # bs, 2
            eqn = []
            for i in range(batch_size):
                h = []
                for j in range(2):
                    if pred_labels[i][j] in ll[i]:
                        h.append(1)
                    else:
                        h.append(0)
                eqn.append(h)

            accuracy = (np.sum(eqn) / (batch_size * 2)).astype("float32")

            print("#data = %d, #accuracy = %f" % (d, accuracy))

            ts_mix_lst = []
            labels_list = []