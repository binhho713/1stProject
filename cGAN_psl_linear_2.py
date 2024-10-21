import argparse
import os
import numpy as np
import math

from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, time_series_data, conditions):
        self.time_series_data = time_series_data
        self.conditions = conditions

    def __len__(self):
        return len(self.time_series_data)

    def __getitem__(self, idx):
        return self.time_series_data[idx], self.conditions[idx]

# Hyperparameters
noise_dim = 200
condition_dim = 171
output_dim = 108
batch_size = 128
learning_rate = 0.0002
num_epochs = 400

class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim=171, output_dim=108, embedding_dim=100):
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
            nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),    
            nn.Linear(2048, output_dim)
        )

    def forward(self, noise, condition):
        y = self.embedding(condition)
        x = torch.cat([noise, y], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim=108, condition_dim=171, embedding_dim=100):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(condition_dim, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(input_dim + embedding_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.Linear(2048, 2048),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.Linear(2048, 1536),
            nn.LeakyReLU(0.2, inplace=True),
	    nn.Linear(1536, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, time_series, condition):
        y = self.embedding(condition)
        x = torch.cat([time_series, y], dim=1)
        return self.model(x)

# Instantiate the models
generator = Generator(noise_dim, condition_dim, output_dim)
discriminator = Discriminator(output_dim, condition_dim)
generator.cuda()
discriminator.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()
criterion.cuda()

#data
data = np.load("train_cGAN_psl.npy", allow_pickle=True)
time_series_data, conditions = np.split(data, (data.shape[1]-1,), axis=1)

dataset = TimeSeriesDataset(torch.tensor(time_series_data, dtype=torch.float32).cuda(),
                            torch.tensor(conditions, dtype=torch.long).cuda())
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

save_epoch = 10
#train loop
for epoch in range(num_epochs):
    for i, (real_time_series, condition) in enumerate(data_loader):
        batch_size = real_time_series.size(0)
        condition = condition.reshape(-1)

        # Real and fake labels
        real_labels = torch.ones(batch_size, 1).cuda()
        fake_labels = torch.zeros(batch_size, 1).cuda()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Real time series
        d_loss_real = criterion(discriminator(real_time_series, condition), real_labels)

        # Fake time series
        noise = torch.randn(batch_size, noise_dim).cuda()
        fake_time_series = generator(noise, condition)
        d_loss_fake = criterion(discriminator(fake_time_series.detach(), condition), fake_labels)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()

        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()

        outputs = discriminator(fake_time_series, condition)
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()

        optimizer_G.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}] - d_loss: {d_loss:.4f}, g_loss: {g_loss:.4f}')

    if (epoch) % save_epoch == 0:
        torch.save(generator.state_dict(), "models/G_ts_psl_linear_2-%d.model" % (epoch))
        torch.save(discriminator.state_dict(), "models/D_ts_psl_linear_2-%d.model" % (epoch))