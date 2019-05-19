import argparse
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt


class Generator(nn.Module):
    def __init__(self, x_dims):
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
        #   Linear 1024 -> 768
        #   Output non-linearity

        relu_leak = 0.2
        img_len = x_dims[-1] * x_dims[-2]


        self.layers = [nn.Linear(ARGS.latent_dim, 128),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(128, 256),
                      nn.BatchNorm1d(256),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(256, 512),
                      nn.BatchNorm1d(512),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(512, 1024),
                      nn.BatchNorm1d(1024),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(1024, img_len),
                      # nn.Sigmoid()
                      nn.BatchNorm1d(img_len),
                      nn.Tanh()
                       ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, z):
        # Generate images from z
        out = self.model(z)

        return out


class Discriminator(nn.Module):
    def __init__(self, x_dims):
        super(Discriminator, self).__init__()

        # Construct distriminator. You are free to experiment with your model,
        # but the following is a good start:
        #   Linear 784 -> 512
        #   LeakyReLU(0.2)
        #   Linear 512 -> 256
        #   LeakyReLU(0.2)
        #   Linear 256 -> 1
        #   Output non-linearity

        relu_leak = 0.2

        # suggested implementation
        img_len = x_dims[-1] * x_dims[-2]
        self.layers = [nn.Linear(img_len, 512),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(512, 256),
                      nn.LeakyReLU(relu_leak),
                      nn.Linear(256, 1),
                      nn.Sigmoid()
                       ]
        self.model = nn.Sequential(*self.layers)

    def forward(self, img):
        # return discriminator score for img
        out = self.model(img)

        return out


def save_loss_plot(G_curve, D_curve):
    plt.figure(figsize=(12, 6))
    x = [1] + list(torch.arange(ARGS.save_interval, ARGS.save_interval * len(G_curve), ARGS.save_interval))
    plt.plot(x, G_curve, label='Generator loss')
    plt.plot(x, D_curve, label='Discriminator curve')
    plt.legend()
    plt.xlabel('Training batches')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('GAN_loss.pdf')


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, x_dims):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    D_losses = []
    G_losses = []

    for epoch in range(ARGS.n_epochs):
        print("Epoch:", epoch)

        for i, (imgs, _) in enumerate(dataloader, start=1):
            x = imgs.view(-1, x_dims[-1] * x_dims[-2]).to(device)

            # Train Generator
            # ---------------
            z = torch.randn(imgs.shape[0], ARGS.latent_dim, device=device)
            gen_imgs = generator(z)
            D_G_out = discriminator(gen_imgs)
            ones = torch.ones(D_G_out.shape, device=device)
            G_loss = binary_cross_entropy(D_G_out, ones)
            optimizer_G.zero_grad()
            G_loss.backward(retain_graph=True)
            optimizer_G.step()

            # Train Discriminator
            # -------------------
            D_real_out = discriminator(x)
            D_loss = binary_cross_entropy(D_real_out, 0.9 * ones) + binary_cross_entropy(ones - D_G_out, 0.9* ones)
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % ARGS.save_interval == 0 or batches_done == 1:
                print(f"[Batches done {batches_done}] Gen Loss: {G_loss.item()} Dis Loss: {D_loss.item()}")
                D_losses.append(D_loss.item())
                G_losses.append(G_loss.item())
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                gen_imgs = gen_imgs.view(x_dims)
                save_image(gen_imgs[:25],
                           'GAN_images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)
    save_loss_plot(G_losses, D_losses)

def main():
    # Create output image directory
    os.makedirs('GAN_images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))
                       ])),
        batch_size=ARGS.batch_size, shuffle=True)

    # Initialize models and optimizers
    x_dims = next(iter(dataloader))[0].shape

    generator = Generator(x_dims[1:])
    discriminator = Discriminator(x_dims[1:])
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=ARGS.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=ARGS.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, x_dims)

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
    ARGS = parser.parse_args()

    main()
