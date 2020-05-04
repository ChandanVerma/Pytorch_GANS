import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np 
import matplotlib.pyplot as plt 
from datetime import datetime
import os 

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean= (0.5,), std= (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root= '/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/',
                                            train = True,
                                            download = True, 
                                            transform = transforms)

batch_size = 128
train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= batch_size, shuffle= True)

## Discriminator
D = nn.Sequential(
    nn.Linear(784, 512),
    nn.LeakyReLU(0.2),
    nn.Linear(512, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1)
)

## Generator
latent_dim = 100
G = nn.Sequential(
    nn.Linear(latent_dim, 256),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(256, momentum= 0.7),
    nn.Linear(256, 512),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(512, momentum= 0.7),
    nn.Linear(512, 1024),
    nn.LeakyReLU(0.2),
    nn.BatchNorm1d(1024, momentum= 0.7),
    nn.Linear(1024, 784),
    nn.Tanh()
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
D = D.to(device)
G = G.to(device)

criterion = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr = 0.0002, betas= (0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr = 0.0002, betas= (0.5, 0.999))

## saving images back to 0,1
def scale_img(img):
    out = (img + 1)/2
    return out

if not os.path.exists('/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/gan_images'):
    os.makedirs('/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/gan_images')

ones_ = torch.ones(batch_size, 1).to(device)
zeros_ = torch.zeros(batch_size, 1).to(device)

g_losses = []
d_losses = []

for epoch in range(200):
    for inputs, _ in train_dataloader:

        n = inputs.size(0)
        inputs = inputs.reshape(n, 784).to(device)

        ones = ones_[:n]
        zeros = zeros_[:n]

        ## Train discriminator

        ## real images
        real_outputs = D(inputs)
        d_loss_real = criterion(real_outputs, ones)

        ## fake images
        noise = torch.randn(n, latent_dim).to(device)
        fake_images = G(noise)
        fake_outputs = D(fake_images)
        d_loss_fake = criterion(fake_outputs, zeros)

        ## gradient descent step
        d_loss = 0.5 * (d_loss_fake + d_loss_real)
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        d_loss.backward()


        ## Train generator
        ## do it twice

        for _ in range(2):
            ## fake images
            noise = torch.randn(n, latent_dim).to(device)
            fake_images = G(noise)
            fake_outputs = D(fake_images)

            ## reverse the label
            g_loss = criterion(fake_outputs, ones)

            ## gradient descent step
            d_optimizer.zero_grad()
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())
    
    print(f'Epoch: {epoch}, d_loss : {d_loss.item()}, g_loss: {g_loss.item()}')

    fake_images = fake_images.reshape(-1, 1, 28, 28)
    save_image(scale_img(fake_images), f'/home/chandanv/Drive/Courses/Pytorch:AI/Pytorch_AI/data/gan_images/{epoch+1}.png')