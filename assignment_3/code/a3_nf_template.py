import argparse

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from datasets.mnist import mnist
import os
from torchvision.utils import make_grid, save_image


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_prior(x):
    """
    Compute the elementwise log probability of a standard Gaussian, i.e.
    N(x | mu=0, sigma=1).
    """

    logp = (- np.log(np.sqrt(2 * np.pi)) - 0.5 * x**2).sum(dim=1)

    return logp


def sample_prior(size):
    """
    Sample from a standard Gaussian.
    """

    sample = np.random.normal(0, 1, size)
    sample = torch.from_numpy(sample)
    if torch.cuda.is_available():
        sample = sample.cuda()

    return sample


def get_mask():
    mask = np.zeros((28, 28), dtype='float32')
    for i in range(28):
        for j in range(28):
            if (i + j) % 2 == 0:
                mask[i, j] = 1

    mask = mask.reshape(1, 28*28)
    mask = torch.from_numpy(mask)

    return mask


class Coupling(torch.nn.Module):
    def __init__(self, c_in, mask, n_hidden=1024):
        super().__init__()
        self.n_hidden = n_hidden

        # Assigns mask to self.mask and creates reference for pytorch.
        self.register_buffer('mask', mask)

        # Create shared architecture to generate both the translation and
        # scale variables.
        # Suggestion: Linear ReLU Linear ReLU Linear.
        self.layers = [nn.Linear(c_in, n_hidden),
                       nn.ReLU(),
                       nn.Linear(n_hidden, n_hidden),
                       nn.ReLU()]
        self.nn = nn.Sequential(*self.layers)

        self.t = nn.Sequential(nn.Linear(n_hidden, c_in))
        self.s = nn.Sequential(nn.Linear(n_hidden, c_in), nn.Tanh())

        # The nn should be initialized such that the weights of the last layer
        # is zero, so that its initial transform is identity.
        self.t[-1].weight.data.zero_()
        self.t[-1].bias.data.zero_()
        self.s[-2].weight.data.zero_()
        self.s[-2].bias.data.zero_()

    def forward(self, z, ldj, reverse=False):
        # Implement the forward and inverse for an affine coupling layer. Split
        # the input using the mask in self.mask. Transform one part with
        # Make sure to account for the log Jacobian determinant (ldj).
        # For reference, check: Density estimation using RealNVP.

        # NOTE: For stability, it is advised to model the scale via:
        # log_scale = tanh(h), where h is the scale-output
        # from the NN.

        z_m = self.mask * z
        z_h = self.nn(z_m)
        log_scale = self.s(z_h)
        transform = self.t(z_h)

        if not reverse:
            z = z_m + (1 - self.mask) * (z * log_scale.exp() + transform)
            # no exp needed since we are summing over the log determinants
            ldj += torch.sum((1 - self.mask) * log_scale, dim=1)
        else:
            z = z_m + (1 - self.mask) * (z - transform) * (- log_scale).exp()
            ldj = torch.zeros_like(ldj)
        return z, ldj


class Flow(nn.Module):
    def __init__(self, shape, n_flows=4):
        super().__init__()
        channels, = shape

        mask = get_mask()

        self.layers = torch.nn.ModuleList()

        for i in range(n_flows):
            self.layers.append(Coupling(c_in=channels, mask=mask))
            self.layers.append(Coupling(c_in=channels, mask=1-mask))

        self.z_shape = (channels,)

    def forward(self, z, logdet, reverse=False):
        if not reverse:
            for layer in self.layers:
                z, logdet = layer(z, logdet)
        else:
            for layer in reversed(self.layers):
                z, logdet = layer(z, logdet, reverse=True)

        return z, logdet


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.flow = Flow(shape)

    def dequantize(self, z):
        return z + torch.rand_like(z)

    def logit_normalize(self, z, logdet, reverse=False):
        """
        Inverse sigmoid normalization.
        """
        alpha = 1e-5

        if not reverse:
            # Divide by 256 and update ldj.
            z = z / 256.
            logdet -= np.log(256) * np.prod(z.size()[1:])

            # Logit normalize
            z = z*(1-alpha) + alpha*0.5
            logdet += torch.sum(-torch.log(z) - torch.log(1-z), dim=1)
            z = torch.log(z) - torch.log(1-z)

        else:
            # Inverse normalize
            logdet += torch.sum(torch.log(z) + torch.log(1-z), dim=1)
            z = torch.sigmoid(z)

            # Multiply by 256.
            z = z * 256.
            logdet += np.log(256) * np.prod(z.size()[1:])

        return z, logdet

    def forward(self, input):
        """
        Given input, encode the input to z space. Also keep track of ldj.
        """
        z = input
        ldj = torch.zeros(z.size(0), device=z.device)

        z = self.dequantize(z)
        z, ldj = self.logit_normalize(z, ldj)

        z, ldj = self.flow(z, ldj)

        # Compute log_pz and log_px per example

        log_pz = log_prior(z)
        log_px = log_pz + ldj

        return log_px

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Sample from prior and create ldj.
        Then invert the flow and invert the logit_normalize.
        """
        z = sample_prior((n_samples,) + self.flow.z_shape).float()
        ldj = torch.zeros(z.size(0), device=z.device).float()

        z, ldj = self.flow.forward(z, ldj, reverse=True)

        z, ldj = self.logit_normalize(z, ldj, reverse=True)

        return z


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average bpd ("bits per dimension" which is the negative
    log_2 likelihood per dimension) averaged over the complete epoch.
    """

    bpd_list = []

    for i, (x, t) in enumerate(data, start=1):
        x = x.to(DEVICE)

        logp = model.forward(x)
        loss = -logp.mean()

        if model.training:
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        bpd_list.append(loss.item()/np.log(2)/784)
        if i % 100 == 0:
            print(f"[Batches done {i}] Loss: {loss.item()} BPD: {loss.item()/np.log(2)/784}")

    avg_bpd = np.mean(bpd_list)
    return avg_bpd


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average bpd for each.
    """
    traindata, valdata = data

    model.train()
    train_bpd = epoch_iter(model, traindata, optimizer)

    model.eval()
    val_bpd = epoch_iter(model, valdata, optimizer)

    return train_bpd, val_bpd


def save_bpd_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train bpd')
    plt.plot(val_curve, label='validation bpd')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('bpd')
    plt.tight_layout()
    plt.savefig(filename)


def create_sample_grid(model, epoch):
    with torch.no_grad():
        img_samples = model.sample(ARGS.num_rows ** 2)
        img_shape = int(np.sqrt(img_samples.shape[-1]))
        img_samples = img_samples.reshape(-1, 1, img_shape, img_shape)
        sampled_grid = make_grid(img_samples, nrow=ARGS.num_rows, normalize=True)
        path = "./images_nfs/"
        os.makedirs(path, exist_ok=True)
        save_image(sampled_grid, path + f"train_{epoch}.png")


def main():
    data = mnist()[:2]  # ignore test split

    model = Model(shape=[784])

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('images_nfs', exist_ok=True)
    create_sample_grid(model, 0)

    train_curve, val_curve = [], []
    for epoch in range(1, ARGS.epochs+1):
        bpds = run_epoch(model, data, optimizer)
        train_bpd, val_bpd = bpds
        train_curve.append(train_bpd)
        val_curve.append(val_bpd)
        print("[Epoch {epoch}] train bpd: {train_bpd} val_bpd: {val_bpd}".format(
            epoch=epoch, train_bpd=train_bpd, val_bpd=val_bpd))

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functionality that is already imported.
        #  Save grid to images_nfs/

        if epoch % 5 == 0 or epoch == 1:
            torch.save(model.state_dict(), "NF_model.pt")
            create_sample_grid(model, epoch)

        # --------------------------------------------------------------------

    save_bpd_plot(train_curve, val_curve, 'nfs_bpd.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--num_rows', default=5, type=int,
                        help='number of rows in sample image')

    ARGS = parser.parse_args()

    main()
