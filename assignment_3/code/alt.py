import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.nn.functional import binary_cross_entropy


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        # Construct generator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear args.latent_dim -> 128
        #   LeakyReLU(0.2)
        #   Linear 128 -> 256
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 256 -> 512
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 512 -> 1024
        #   Bnorm
        #   LeakyReLU(0.2)
        #   Linear 1024 -> 784
        #   Output non-linearity

        self.latent_dim = latent_dim
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, z):
        # Generate images from z
        out = self.model(z)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        # return discriminator score for img
        out = self.model(img)

        return out


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device):
    D_losses = []
    G_losses = []
    for epoch in range(args.n_epochs):
        print("Epoch:", epoch)
        for i, (imgs, _) in enumerate(dataloader):
            batch_count = epoch * len(dataloader) + i

            imgs = imgs.to(device)
            batch_size = imgs.shape[0]
            imgs = imgs.reshape(batch_size, -1)

            z = torch.randn(batch_size, generator.latent_dim, device=device)
            gen_imgs = generator(z)

            d_x = discriminator(imgs)
            d_g_z = discriminator(gen_imgs)

            ones = torch.ones(d_g_z.shape, device=device)

            # Train Generator
            # ---------------
            loss_G = binary_cross_entropy(d_g_z, ones)

            optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            if batch_count % args.d_train_interval == 0:
                loss_D = binary_cross_entropy(d_x, ones) + binary_cross_entropy(ones - d_g_z, ones)

                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            # Save Images
            # -----------
            if batch_count % args.save_interval == 0:
                print(f'epoch: {epoch} batches: {batch_count} L_G: {loss_G.item():0.3f} L_D: {loss_D.item():0.3f}')
                D_losses.append(loss_D.item())
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = gen_imgs.reshape(batch_size, 1, 28, 28)
                save_image(gen_imgs[:25],
                           f'GAN_images/{batch_count}.png',
                           nrow=5, normalize=True)


def main(args):
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize models and optimizers
    generator = Generator(args.latent_dim)
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    parser.add_argument('--d_train_interval', type=int, default=1,
                        help='train discriminator (only) every D_TRAIN_INTERVAL iterations')
    args = parser.parse_args()

    main(args)
