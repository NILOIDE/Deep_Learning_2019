import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

from datasets.bmnist import bmnist

import os
from scipy.stats import norm


class Encoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.shared_layers = nn.Sequential(nn.Linear(x_dim, hidden_dim),
                              nn.ReLU())
        self.mu_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))
        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, z_dim))

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        shared_output = self.shared_layers(input)
        mean = self.mu_head(shared_output)
        std = self.sigma_head(shared_output)

        return mean, std


class Decoder(nn.Module):

    def __init__(self, x_dim, hidden_dim=500, z_dim=20):
        super().__init__()
        self.layers = [nn.Linear(z_dim, hidden_dim), nn.ReLU(),
                       nn.Linear(hidden_dim, x_dim), nn.Sigmoid()]
        self.model = nn.Sequential(*self.layers)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean with shape [batch_size, 784].
        """

        mean = self.model(input)

        return mean


class VAE(nn.Module):

    def __init__(self, x_dims, hidden_dim=500, z_dim=20):
        super().__init__()

        self.z_dim = z_dim
        self.x_dims = x_dims
        self.encoder = Encoder(x_dim=x_dims[-2]*x_dims[-1], hidden_dim=hidden_dim, z_dim=z_dim)
        self.decoder = Decoder(x_dim=x_dims[-2]*x_dims[-1], hidden_dim=hidden_dim, z_dim=z_dim)
        self.loss = nn.BCELoss(reduction='none')

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, log_std = self.encoder.forward(input)
        noise = torch.normal(torch.zeros_like(mean), torch.ones_like(mean))
        z = mean + log_std.exp() * noise
        out = self.decoder.forward(z)

        l_reg = 0.5 * (torch.sum(log_std.exp() + mean**2 - log_std, dim=1) - 1)
        l_recon = torch.sum(self.loss(out, input), dim=1)
        average_negative_elbo = torch.mean(l_recon + l_reg, dim=0)

        return average_negative_elbo

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        with torch.no_grad():
            z = torch.normal(torch.zeros(n_samples, self.z_dim), torch.ones(n_samples, self.z_dim))
            sampled_imgs = self.decoder.forward(z).view(n_samples, *self.x_dims[1:])
            im_means = sampled_imgs.mean(dim=0)

        return sampled_imgs, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    epoch_elbo_list = []

    for i, batch in enumerate(data):
        model.zero_grad()
        batch = batch.view(batch.shape[0], -1)
        epoch_elbo = model.forward(batch)
        epoch_elbo.backward()
        # nn.utils.clip_grad_norm(model.parameters(), max_norm=ARGS.max_norm)
        optimizer.step()
        epoch_elbo_list.append(epoch_elbo)

    average_epoch_elbo = torch.tensor(epoch_elbo_list).mean()

    return average_epoch_elbo


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def create_sample_grid(model, epoch):
    img_samples, img_means = model.sample(ARGS.num_rows ** 2)
    sampled_grid = make_grid(img_samples, nrow=ARGS.num_rows)
    mean_grid = make_grid(img_means, nrow=ARGS.num_rows)
    path = "./VAE_samples/"
    os.makedirs(path, exist_ok=True)
    save_image(sampled_grid, path + f"train_{epoch}.png")
    save_image(mean_grid, path + f"train_{epoch}_mean.png")

def main():

    data = bmnist()[:2]  # ignore test split
    x_dims = next(iter(data[0])).shape
    model = VAE(x_dims=x_dims, z_dim=ARGS.zdim)
    optimizer = torch.optim.Adam(model.parameters())
    create_sample_grid(model, 0)

    train_curve, val_curve = [], []
    for epoch in range(1, ARGS.epochs+1):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        if epoch % 5 == 0:
            create_sample_grid(model, epoch)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        grid = torch.empty((ARGS.manifold_rows,ARGS.manifold_rows,2))
        x = torch.linspace(norm.ppf(0.01), norm.ppf(0.99), ARGS.manifold_rows)
        x = x[:,None]
        grid[:,:,0] = x.repeat(1,ARGS.manifold_rows)
        grid[:,:,1] = x.repeat(1,ARGS.manifold_rows).t()
        img_samples = model.decoder.forward(grid.view(ARGS.manifold_rows*ARGS.manifold_rows, 2))
        img_samples = img_samples.view(ARGS.manifold_rows**2, *x_dims[-3:])
        sampled_grid = make_grid(img_samples, nrow=ARGS.manifold_rows)
        save_image(sampled_grid, f"VAE_manifold.png")

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--max_norm', default=10, type=float,
                        help='gradient clip value')
    parser.add_argument('--num_rows', default=10, type=int,
                        help='number of rows in sample image')
    parser.add_argument('--manifold_rows', default=20, type=int,
                        help='number of rows in sample image')

    ARGS = parser.parse_args()

    main()
