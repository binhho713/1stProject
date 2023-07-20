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

#Load data
time_series = np.load("")
time_series = np.reshape(time_series, ( -1, 1, 32, 32))
n_timeSeries = len(time_series)

#Entropy function
def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim=1), dim=0)

#Function to find k lagerst numbers
def k_lagerst(input, k):
    z = np.argsort(input)[::-1]
    output = []
    for i in range(k):
        output.append(z[i])
    return output

#All different constraints
#def gen_alldiff_constraints(nums, batch_size):
#return all_diffs

# Function to save the model
def saveModel():
    path = "./models.pth"
    torch.save(model.state_dict(), path)

check_freq = 100
ts_mix_lst = []
running_label_acc = 0
running_loss = 0
cnt = 0

#----------
# Training
#----------
for _epoch_ in range(10000):

    for idx in range(n_timeSeries):

        ts_mix = time_series[idx]
        ts_mix = np.reshape(ts_mix, (1, 32, 32))

        ts_mix_lst.append(ts_mix)

        if (len(ts_mix_lst) == batch_size):
            ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs, 1, 32, 32
            ts_mix = Variable(torch.tensor(ts_mix).float(), requires_grad=False)

            if use_cuda:
                ts_mix = ts_mix.cuda()

            labels_distribution = pred(ts_mix)

            if(use_cuda):
                z = sep(torch.tensor(ts_mix.reshape(-1, 1, 32, 32)).float()).cuda()
            else:
                z = sep(torch.tensor(ts_mix.reshape(-1, 1, 32, 32)).float())

            optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()), lr=lr)

            labels = k_lagerst(labels_distribution, truth_labels.size(0))
            eqn = np.equal(labels, truth_labels).astype("int")
            label_acc = (np.sum(eqn) == truth_labels.size(0)).astype("float32")

            #generate image
            gen_img = generator(z.view(-1,100), gen_label) #bs*10, 1, 32, 32
            gen_mix = gen_imgs.permute(1, 2, 3, 0) * label_distribution.view(-1)
            gen_mix = gen_mix.view(1, 32, 32, batch_size, 10)
            gen_mix = torch.sum(gen_mix, dim=4)  # avg by distribution 1, 32, 32, bs
            gen_img_demix = gen_mix.permute(3, 0, 1, 2)  # bs, 1, 32, 32 #only used for visualization
            gen_mix = gen_mix.view(-1, 32, 32)  # bs, 1, 32, 32

            #reconstruct loss
            cri = torch.nn.L1Loss()
            loss_cre = cri(ts_mix.view(-1, 32, 32), gen_img)
            loss_cre /= (1.0 * labels_distribution.size(0))

            #k_sparity constrain
            c = np.log(2) - 0.001
            k_sparity = np.maximum(entropy(labels_distribution),c)

            scale_recon = 0.001

            loss = scale_recon*loss_mix + 0.01*k_sparity
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            s_mix_lst = []
            running_label_acc += label_acc
            running_loss += scale_recon * loss.item()
            cnt += 1

            #save model
            if (cnt % check_freq == 0):
                print("#puzzle = %d, sudoku_acc = %f, label_acc = %f, recon_loss = %f" % (
                cnt * batch_size, running_label_acc / cnt, running_loss / cnt))

                save_model(pred, pred_path)
                save_model(sep, sep_path)

                running_label_acc = 0
                running_loss = 0
                cnt = 0




