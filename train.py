import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import VAE, Encoder, Decoder
import tqdm
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    # parameters
    latent_dim = 64
    data_dim = 784
    batch_size = 128
    epochs = 200

    # load data
    train_iter = DataLoader(datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True, drop_last=True)
    test_iter = DataLoader(datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True, drop_last=True)

    # model
    encoder = Encoder(latent_dim, data_dim).to(torch.device('cuda'))
    decoder = Decoder(latent_dim, data_dim).to(torch.device('cuda'))
    model = VAE(encoder, decoder).to(torch.device('cuda'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.5)

    # training
    Trange = tqdm.trange(epochs)
    loss_list = []
    for epoch in Trange:
        model.train()
        for X, y in train_iter:
            X = X.to(torch.device('cuda')).reshape(batch_size, -1)
            loss = model(X).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(torch.device('cuda')).reshape(batch_size, -1)
                loss = model(X).mean()
                test_loss += loss.item()
        
        test_loss /= len(test_iter)
        Trange.set_postfix(test_loss=test_loss)
        loss_list.append(test_loss)

    # save
    torch.save(model.state_dict(), "model.pth")

    plt.plot(np.arange(len(loss_list)), np.array(loss_list))
    plt.show()