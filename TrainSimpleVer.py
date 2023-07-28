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

from resnet import *

batch_size = 64
latent_dim = 100
n_classes = 10
img_shape = (1, 32, 32)
img_size = 32
nums = 10
use_cuda = torch.cuda.is_available()


#Generator to reconstruct input
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        #print ("label: " labels)
        #print ("!!!!!!!!!!!!!!!1 ", self.label_emb(labels).size(), " ", noise.size())
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

#Init generator & load pretrained generative decoder
generator = Generator()
if torch.cuda.is_available():
    print ("use cuda")
    generator = generator.cuda()
generator.load_state_dict(torch.load("models/G-180.model"))
generator.eval()

#Init seperator and predictor
sep = resnet18(predictor=False)
pred = resnet18(predictor=True)

if use_cuda:
    sep = sep.cuda()
    pred = pred.cuda()
    print("use_cuda")

#Load data
ts1 = np.load("./4by4.npy") #10000, 16, 1, 32, 32
ts2 = np.load("./4by4_5678.npy") #10000, 16, 1, 32, 32
time_series = np.maximum(ts1, ts2) #10000, 16, 1, 32, 32
n_timeSeries = len(time_series)

ls1 = np.load("./4by4_labels.npy") #10000, 16
ls2 = np.load("./4by4_5678_labels.npy") #10000, 16
truth_labels_list = np.concatenate((ls1.reshape(-1, 16, 1), ls2.reshape(-1, 16, 1)), axis = 2) #100, 16, 2

gen_labels = torch.LongTensor(np.arange(16 * nums * batch_size, dtype = "int32") % nums)
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


#All different constraints
#def gen_alldiff_constraints(nums, batch_size):
#return all_diffs

# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)

pred_path = "./models/pred.model-9992"
sep_path = "./models/sep.model-9992"

check_freq = 100
lr = 0.0001

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

        ts_mix = time_series[idx] #16, 1, 32, 32
        ts_mix_lst.append(ts_mix)
        
        truth_labels = truth_labels_list[idx] #16, 2
        labels_list.append(truth_labels)

        if (len(ts_mix_lst) == batch_size):
            ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs*16, 1, 32, 32
            ts_mix = Variable(torch.tensor(ts_mix).float(), requires_grad=False)
            if use_cuda:
                ts_mix = ts_mix.cuda()

            labels_distribution = pred(ts_mix) #bs*16, 10
            if(use_cuda):
                z = sep(torch.tensor(ts_mix.reshape(-1, 1, 32, 32)).float()).cuda() #bs*16, 1, 10, 100
            else:
                z = sep(torch.tensor(ts_mix.reshape(-1, 1, 32, 32)).float())

            optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()), lr=lr)

            labels = labels_distribution.cpu().data.numpy()
            labels = k_lagerst(labels, 2)
            ll = np.reshape(labels_list,(-1, 2)) #bs*16, 2
            eqn = []
            for i in range(batch_size*16):
                h = []
                for j in range(2):
                    if labels[i][j] in ll[i]:
                        h.append(1)
                    else:
                        h.append(0)
                eqn.append(h)

            label_acc = (np.sum(eqn)/(batch_size*16*2)).astype("float32")

            #generate image
            gen_img = generator(z.view(-1,100), gen_labels) #bs*16*10, 1, 32, 32
            gen_mix = gen_img.permute(1, 2, 3, 0) * labels_distribution.view(-1)#1,32,32,bs*16*10
            gen_mix = gen_mix.permute(3, 0, 1, 2).view(batch_size*16,10,1,32,32) # bs*16, 10, 1, 32, 32
            h = []
            array = gen_mix.detach().cpu().numpy()
            for i in range(len(array)):
                copy = []
                for j in range(2):
                    copy.append(array[i][int(labels[i][j])])
                h.append(copy)
            gen_mix = torch.tensor(np.array(h)).float().cuda()
            gen_mix = torch.reshape(gen_mix, (batch_size*16, 2, 1, 32, 32))
            #arr = gen_mix.detach().cpu().numpy()
            gen_mix = torch.max(gen_mix.permute( 2, 3, 4, 0, 1), dim=4)[0] # 1, 32, 32, bs*16
            gen_mix = gen_mix.permute(3, 0, 1, 2).view(-1, 1, 32, 32)  # bs*16, 1, 32, 32
            arr5 = ts_mix.detach().cpu().numpy()
            arr3 = gen_mix.detach().cpu().numpy()

            #reconstruct loss
            loss_rec = 0
            cri = torch.nn.L1Loss()
            loss_rec = cri(ts_mix.view(-1, 1, 32, 32), gen_mix)
            #loss_rec /= (1.0 * labels_distribution.size(0))

            #k_sparity constrain
            c = np.log(2) - 1e-6
            c = torch.tensor(c).float()
            k_sparity = torch.maximum(entropy(labels_distribution),c).float()

            #scale_recon = 0.001

            loss = loss_rec + 0.1*k_sparity
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
                save_model(sep, sep_path)

                running_label_acc = 0
                running_loss = 0
                cnt = 0




