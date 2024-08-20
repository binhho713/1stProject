import argparse
import os
import numpy as np
import math

#import torchvision.transforms as transforms
#from torchvision.utils import save_image

#from torch.utils.data import DataLoader
#from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=171, help="number of classes for dataset")
parser.add_argument("--ts_size", type=int, default=108, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--save_epoch", type=int, default=20, help="interval between each save")
opt = parser.parse_args()
print(opt)

batch_size = opt.batch_size

ts_shape = (opt.channels, opt.ts_size)

cuda = True if torch.cuda.is_available() else False

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)
        self.ln = nn.Linear(opt.n_classes, 8 * 8)
        self.ln1 = nn.Linear(opt.latent_dim, 8 * 8 * 128)

        self.deConv1 = nn.ConvTranspose1d(129, 64, kernel_size=4, stride=4)
        self.deConv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deConv3 = nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1)
        self.ln2 = nn.Linear(1024, int(np.prod(ts_shape)))
        self.relu = nn.ReLU()

    def forward(self, noise, labels):

        x = self.ln1(noise)
        x = torch.reshape(x, (-1, 128, 8*8))
        y = self.label_emb(labels)
        y = self.ln(y)
        y = torch.reshape(y, (-1, 1, 8*8))
        ts = torch.cat((x, y), 1)
        ts = self.deConv1(ts)
        ts = self.relu(ts)
        ts = self.deConv2(ts)
        ts = self.relu(ts)
        ts = self.deConv3(ts)
        ts = self.relu(ts)
        ts = self.ln2(ts)

        return ts

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)
        self.ln = nn.Linear(opt.n_classes, int(np.prod(ts_shape)))

        self.conv1 = nn.Conv1d(2, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ln1 = nn.Linear(6 * 128, 1)

    def forward(self, ts, labels):

        y = self.label_embedding(labels)
        y = self.ln(y)
        y = torch.reshape(y, (-1, 1, 108))
        out = torch.reshape(ts, (-1, 1, 108))
        out = torch.cat((out, y), dim=1)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = torch.flatten(out, 1)
        out = self.ln1(out)

        return out

# Loss functions
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
def generate_ts():

    x = np.zeros((10, 1024))

    for i in range(10):
        x[i] = np.random.normal(loc=0.0, scale=1.0, size=1024)

    np.save('base_ts.npy', x)
    s = np.zeros((10000, 1025))
    j = 0
    for i in range(10):
        h = np.random.normal(loc=0.0, scale=1.0, size=1000)
        for k in range(1000):
            t = x[i]*h[k]
            noise = np.random.normal(loc=0.0, scale=0.04, size = 1024)
            t += noise
            l = i
            s[j] = np.concatenate((t, np.array([l])), axis=0)
            j += 1

    np.save('train_ts.npy', s)

#generate_ts()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
#data = np.load('train_ts.npy')
data = np.load('train_psl.npy')

for epoch in range(opt.n_epochs):

    copy_data = data
    np.random.shuffle(copy_data)

    while len(copy_data) >= batch_size:

        ts_list, copy_data = np.split(copy_data, (batch_size,), axis=0)
        ts, labels = np.split(ts_list, (ts_list.shape[1]-1 ,), axis=1)

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_ts = Variable(torch.tensor(ts).float())
        labels = Variable(torch.LongTensor(labels).reshape(-1))

        if torch.cuda.is_available():
            real_ts = real_ts.cuda()
            labels = labels.cuda()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_ts = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_ts, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_ts, labels)
        d_real_loss = adversarial_loss(validity_real, valid)
        #accuracy = torch.sum(torch.eq(gen_labels, labels), dtype=float) / 64

        # Loss for fake images
        validity_fake = discriminator(gen_ts.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [data %d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, len(copy_data), d_loss.item(), g_loss.item())
        )

    if (epoch) % opt.save_epoch == 0:
        torch.save(generator.state_dict(), "models/G_ts_psl-%d.model" % (epoch))
        torch.save(discriminator.state_dict(), "models/D_ts_psl-%d.model" % (epoch))
