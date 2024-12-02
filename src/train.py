import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from model import Autoencoder, CNN

def add_noise(img, noise_factor=0.5):
    noise = torch.randn_like(img) * noise_factor 
    noisy_img = img + noise 
    noisy_img = torch.clamp(noisy_img, 0.0, 1.0)
    return noisy_img

TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 66
EPOCHS =50
LR = 0.01
device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = datasets.ImageFolder('../dataset/covid19/train',
                                     transforms.Compose([
                                         transforms.Grayscale(),
                                         transforms.Resize((256, 256)),
                                         transforms.ToTensor()
                                     ]))

train_dataloader =DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
if __name__ == '__main__':
    autoencoder = Autoencoder().to(device)
    if device == "cuda":
        print("GPU avaliable")
    else:
        print("GPU not avaliable")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    
    autoencoder.train()
    print("Start training...")
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for img, label in tqdm(train_dataloader):
            img = img.to(device)
            noisy_img = add_noise(img)
            optimizer.zero_grad()
            reconstructed = autoencoder(noisy_img)
            loss = criterion(reconstructed, img)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / TRAIN_BATCH_SIZE
        print(f"[Autoencoder]\tEpoch: {epoch}\tLoss: {avg_loss}")
