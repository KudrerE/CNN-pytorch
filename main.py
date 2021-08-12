import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import os
from torchvision.transforms import ToTensor, Lambda
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Model import NeuralNetwork
from Load import CustomImageDataset
from Model import train_loop
from Model import test_loop
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


learning_rate = 1e-5
batch_size = 64
epochs = 5

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)
optimizer = torch.optim.Adam(params=model.parameters(),lr=learning_rate,betas=(0.9, 0.999))

digit_dataset = CustomImageDataset(
    r"C:\Users\kudre\Desktop\mavhine_try_data\train.csv",
    transform=ToTensor())





train, validate = torch.utils.data.random_split(digit_dataset, [38000, 4000],
                                                generator=torch.Generator().manual_seed(0))


train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(validate, batch_size=batch_size, shuffle=True)

epochs = 10
losses= np.zeros((epochs))
x=np.arange(epochs)
acc=np.zeros((epochs))
print(x)
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    losses[t],acc[t]=test_loop(test_dataloader, model, loss_fn)

print("Done!")
print(losses,x)
plt.subplot(2,1,1)
plt.plot(x,losses)
plt.subplot(2,1,2)
plt.plot(x,acc)
plt.show()