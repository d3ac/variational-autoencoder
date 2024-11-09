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
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "CIFAR10"])
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
    encoder = Encoder(latent_dim, data_dim).to(torch.device('cuda'))
    decoder = Decoder(latent_dim, data_dim).to(torch.device('cuda'))
    model = VAE(encoder, decoder).to(torch.device('cuda'))
    if args.dataset == "MNIST":
        model.load_state_dict(torch.load('model-mnist.pth', weights_only=True))
    else:
        model.load_state_dict(torch.load('model-cifa10.pth', weights_only=True))

    # test
    with torch.no_grad():
        for X, y in test_iter:
            pic_size = X.shape[-1]
            X = X.to(torch.device('cuda')).reshape(batch_size, -1)
            new_X = model.reconstruct(X, pic_size, args.dataset)
            X = X.cpu().numpy().reshape(batch_size, -1, pic_size, pic_size)
            new_X = new_X.cpu().numpy().reshape(batch_size, -1, pic_size, pic_size)
            
            fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 2, 4))
            for i in range(batch_size):
                axes[0, i].imshow(X[i].transpose(1, 2, 0), cmap='gray')
                axes[0, i].axis('off')
                axes[1, i].imshow(new_X[i].transpose(1, 2, 0), cmap='gray')
                axes[1, i].axis('off')
            
            axes[0, 0].set_title('Original')
            axes[1, 0].set_title('Reconstructed')
            plt.show()
            break