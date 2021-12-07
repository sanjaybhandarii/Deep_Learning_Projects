import numpy as np
import pandas as pd
import torch
from torch._C import device
from torch.optim import optimizer
import torchvision
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader,random_split
from torchvision import transforms
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

data_dir = 'dataset'

train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
test_dataset  = torchvision.datasets.MNIST(data_dir, train=False, download=True)

tensr_transform = transforms.Compose([
transforms.ToTensor(),
])

train_dataset.transform = tensr_transform
test_dataset.transform = tensr_transform

train_length = len(train_dataset)
test_length = len(test_dataset)

train_data, val_data = random_split(train_dataset, [int(train_length*0.8),int(train_length*0.2)])
batch_size = 256

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



class Encoder(torch.nn.Module):
    def __init__(self,latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.batch2 = torch.nn.BatchNorm2d(16)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.batch3 = torch.nn.BatchNorm2d(32)
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0)
        
        self.linear1 = torch.nn.Linear(1*1*64, 32)
        self.linear3 = torch.nn.Linear(32, latent_dim)
        self.linear4 = torch.nn.Linear(32, latent_dim)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0


    def forward(self, x):
        x = x.to(device)
       
        x = F.relu(self.conv1(x))
        
        x = F.relu(self.batch2(self.conv2(x)))
        
        x = F.relu(self.batch3(self.conv3(x)))
        
        x = F.relu(self.conv4(x))
        
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.linear1(x))
        
        mu = self.linear3(x)
        sigma = torch.exp(self.linear4(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z 

    
class Decoder(torch.nn.Module):
    def __init__(self,latent_dim):
        super(Decoder, self).__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            nn.Linear(32,1*1*64),
            nn.ReLU(True)
        )
        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(64, 1, 1))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_lin(x)
       
        x = self.unflatten(x)
        
        x = self.decoder_conv(x)
        
        x = torch.sigmoid(x)
        return x

class VAE(torch.nn.Module):
    def __init__(self,latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = x.to(device)
        z = self.encoder(x)
        z = self.decoder(z)
        return z


torch.manual_seed(0)

d = 4 #latent dim
vae = VAE(latent_dim=d)
lr = 1e-3

optim = torch.optim.Adam(vae.parameters(), lr=lr, weight_decay=1e-5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Selected device : {device}')

def train_vae(vae, device,dataloader, optimizer):
    vae.train()
    vae.to(device)
    train_loss = 0

    for x, _ in dataloader:
        x = x.to(device)
        x_hat = vae(x)

        loss = ((x - x_hat)**2).sum() + vae.encoder.kl


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    print(f'Train loss : {train_loss/len(dataloader.dataset)}')
    return train_loss / len(dataloader.dataset)


def test_vae(vae, device, dataloader):
    vae.eval()
    vae.to(device)
    val_loss = 0
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            x_encoded = vae.encoder(x)
            x_hat = vae.decoder(x_encoded)
            loss = ((x - x_hat)**2).sum() + vae.encoder.kl
            val_loss += loss.item()
    return val_loss / len(dataloader.dataset)
        

def plot_ae_outputs(encoder,decoder,n=5):
    plt.figure(figsize=(10,4.5))
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img = test_dataset[i][0].unsqueeze(0).to(device)
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
         rec_img  = decoder(encoder(img))
      plt.imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
        ax.set_title('Original images')
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
      if i == n//2:
         ax.set_title('Reconstructed images')
    plt.show() 


num_epochs = 20

for epoch in range(num_epochs):
    train_loss = train_vae(vae, device, train_loader, optim)
    val_loss = test_vae(vae, device, test_loader)
    print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))

plot_ae_outputs(vae.encoder,vae.decoder,n=5)