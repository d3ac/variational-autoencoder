import torch
from torch import nn
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import VAE, Encoder, Decoder
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"])
    parser.add_argument("--path", type=str, default="/home/d3ac/Desktop/dataset")

    args = parser.parse_args()
        
    # parameters
    latent_dim = args.latent_dim
    
    batch_size = args.batch_size
    epochs = args.epochs

    # load data
    if args.dataset == "MNIST":
        train_iter = DataLoader(datasets.MNIST(root=args.path, train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True, drop_last=True)
        test_iter = DataLoader(datasets.MNIST(root=args.path, train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True, drop_last=True)
        data_dim = 28 * 28
    else:
        train_iter = DataLoader(datasets.CIFAR10(root=args.path, train=True, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=True, num_workers=7, pin_memory=True, drop_last=True)
        test_iter = DataLoader(datasets.CIFAR10(root=args.path, train=False, download=True, transform=transforms.ToTensor()), batch_size=batch_size, shuffle=False, num_workers=7, pin_memory=True, drop_last=True)
        data_dim = 32 * 32 * 3
    # model
    encoder = Encoder(latent_dim, 1 if args.dataset == "MNIST" else 3).to(torch.device('cuda'))
    decoder = Decoder(latent_dim, data_dim).to(torch.device('cuda'))
    model = VAE(encoder, decoder).to(torch.device('cuda'))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.3)

    # training
    Trange = tqdm.trange(epochs)
    train_loss_list = []
    test_loss_list = []
    for epoch in Trange:
        model.train()
        train_loss = 0
        for X, y in train_iter:
            X = X.to(torch.device('cuda'))
            loss = model(X).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        scheduler.step()

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(torch.device('cuda'))
                loss = model(X).mean()
                test_loss += loss.item()
        
        test_loss /= len(test_iter)
        train_loss /= len(train_iter)
        Trange.set_postfix({"test_loss": test_loss, "train_loss": train_loss})        
        test_loss_list.append(test_loss)
        train_loss_list.append(train_loss)

    # save
    if args.dataset == "MNIST":
        torch.save(model.state_dict(), "model-mnist.pth")
    else:
        torch.save(model.state_dict(), "model-cifa10.pth")
    plt.plot(np.arange(epochs)[1:], train_loss_list[1:], label="train_loss")
    plt.plot(np.arange(epochs)[1:], test_loss_list[1:], label="test_loss")
    plt.legend()
    plt.show()