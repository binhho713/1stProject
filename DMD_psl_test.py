import argparse
import os, sys
import numpy as np
import math

from sklearn.metrics import accuracy_score

from torch.autograd import Variable
import copy
import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import logging

from encoder_psl import *

batch_size = 128
latent_dim = 200
n_classes = 40
ts_size = 108
nums = 40
use_cuda = torch.cuda.is_available()
is_training = 1

#Tao class generator de dong bo moi model cGAN da train
class Generator(nn.Module):
    def __init__(self, noise_dim=200, condition_dim=40, output_dim=108, embedding_dim=100):
        super(Generator, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1536),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1536, 2048),
            nn.LeakyReLU(0.2, inplace=True),    
            nn.Linear(2048, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)


# Init generator & load pretrained generative decoder
generator = Generator()
if torch.cuda.is_available():
    print("use cuda")
    generator = generator.cuda()
generator.load_state_dict(torch.load("models/G_ts_psl_40_linear-240.model"))
generator.eval()

# Init seperator and predictor
sep = resnet18(predictor=False)
pred = resnet18(predictor=True)

if use_cuda:
    sep = sep.cuda()
    pred = pred.cuda()
    print("use_cuda")

#code for logger
def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger with a file and console handler."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

# Load data

import scipy.io

trainData = np.load("test_mixture_40_psl.npy", allow_pickle=True)
time_series, truth_labels_list = np.split(trainData, (trainData.shape[1]-3,), axis=1)
arrInx = np.arange(0, time_series.shape[0], 1)
np.random.shuffle(arrInx)
n_timeSeries = len(time_series)
gen_labels = torch.LongTensor(np.arange(nums * batch_size, dtype="int32") % nums)
if use_cuda:
    gen_labels = gen_labels.cuda()


# Entropy function
def entropy(x):
    return torch.mean(-torch.sum(x * torch.log(x + 1e-9), dim=1), dim=0)


# Function to find k lagerst variable in input array
def k_lagerst(input, k):
    z = np.argsort(input, axis=1)
    output = np.ndarray(shape=(len(input), k))
    for i in range(len(input)):
        for j in range(k):
            output[i][-j - 1] = z[i][-j - 1]
    return output

def k_lagerst_label(input, k, label):
    z = np.argsort(input, axis=1)
    output = np.ndarray(shape=(len(input), k))
    for i in range(len(input)):
        for j in range(k):
            output[i][-j - 1] = label[int(z[i][-j - 1])]
    return output

def JS_dis(input_xrd, xrd_prime):
    print(KL_dis(input_xrd, xrd_prime), KL_dis(xrd_prime, input_xrd))
    return (0.5 * KL_dis(input_xrd, xrd_prime) + 0.5 * KL_dis(xrd_prime, input_xrd))


# Function to save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)


pred_path = "./models/pred_DMD_ASV_2_"
sep_path = "./models/sep_DMD_ASV_2_"

check_freq = 100
save_freq = 10
lr = 0.00001


if is_training:
    # ----------
    # Training
    # ----------

    for _epoch_ in range(40000):

        ts_mix_lst = []
        labels_list = []
        running_label_acc = 0
        running_loss = 0
        k_loss = 0
        cnt = 0

        for idx in range(n_timeSeries):

            ts_mix = time_series[arrInx[idx]][:ts_size]  # 1, 1024
            ts_mix_lst.append(ts_mix)

            truth_labels = truth_labels_list[arrInx[idx]]  # 1, 2
            labels_list.append(truth_labels)

            if (len(ts_mix_lst) == batch_size):
                ts_mix = np.concatenate(ts_mix_lst, axis=0)  # bs * 1024
                ts_mix = Variable(torch.tensor(ts_mix.reshape(-1, 1, ts_size)).float(), requires_grad=False)  # bs, ts_size
                if use_cuda:
                    ts_mix = ts_mix.cuda()

                labels_distribution = pred(ts_mix)  # bs, 10
                if (use_cuda):
                    z = sep(torch.tensor(ts_mix).float()).cuda()  # bs, 1, ts_classes, 100
                else:
                    z = sep(torch.tensor(ts_mix).float())

                optimizer = torch.optim.Adam(list(pred.parameters()) + list(sep.parameters()), lr=lr)

                #Tinh acc
                labels = labels_distribution.cpu().data.numpy()
                labels = k_lagerst(labels, 3)
                eqn = []
                for i in range(batch_size):
                    h = 0
                    for j in range(3):
                        if labels_list[i][j] in labels[i]:
                            h += 1
                    h /= 3
                    eqn.append(h)

                label_acc = (np.sum(eqn) / len(eqn)).astype("float32")

                # generate image
                gen_ts = generator(z.view(-1, 200), gen_labels)  # bs*ts_classes, 1, ts_size
                gen_ts = torch.reshape(gen_ts, (-1, 1, ts_size))
                gen_ts = gen_ts.permute(1, 2, 0) * labels_distribution.view(-1)  # 1, ts_size, bs*ts_classes
                gen_ts = gen_ts.permute(2, 0, 1).view(batch_size, n_classes, 1, ts_size)  # bs, ts_classes, 1, ts_size
                gen_ts = torch.sum(gen_ts, dim=1)  # bs, 1, ts_size

                # reconstruct loss
                loss_rec = 0
                cri = torch.nn.MSELoss()
                loss_rec = cri(ts_mix, gen_ts)

                # k_sparity constrain DRNet
                c = np.log(3) - 1e-4 #phan ra thanh 3 ts thanh phan -> 4-pole(+ts input)
                c = torch.tensor(c).float()
                # k_sparity = torch.maximum(entropy(labels_distribution),c).float()
                k_sparity = torch.maximum(torch.zeros(1).cuda(), entropy(labels_distribution) - c).float()

                # scale_recon = 0.001

                loss = loss_rec + 0.1 * k_sparity
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

                ts_mix_lst = []
                labels_list = []
                running_loss += loss_rec.item()
                k_loss += k_sparity.item()
                running_label_acc += label_acc
                cnt += 1

                # save model
                if (cnt % check_freq == 0):
                    # print("#epoch = %d, data = %d, running_loss = %f, k_loss= %f" % (
                    # _epoch_, cnt * batch_size, running_loss / cnt, k_loss/cnt))

                    print("#epoch = %d, data = %d, running_label_acc = %f, running_loss = %f, k_loss= %f" % (
                        _epoch_, cnt * batch_size, running_label_acc / cnt, running_loss / cnt, k_loss / cnt))

                    running_label_acc = 0
                    running_loss = 0
                    cnt = 0


