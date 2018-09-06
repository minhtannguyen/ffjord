import matplotlib
matplotlib.use("Agg")

import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image

import integrate

import lib.layers as layers
import lib.layers.diffeq_layers as diffeq_layers
from lib.layers.diffeq_layers import GatedConv, GatedConvTranspose, GatedLinear
import lib.utils as utils

# go fast boi!!
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("Continuous Normalizing Flow VAE")
parser.add_argument("--data", choices=["mnist", "omniglot_static", "omniglot_dynamic"], type=str, default="omniglot_dynamic")
parser.add_argument("--dims", type=str, default="256,256")
parser.add_argument("--hidden_dim", type=int, default=64)

parser.add_argument("--layer_type", type=str, default="concat", choices=["ignore", "concat", "hyper", "blend"])
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "gated"]
)

parser.add_argument("--time_length", type=float, default=1.0)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-5)

parser.add_argument("--adjoint", type=eval, default=True, choices=[True, False])
parser.add_argument("--vanilla_vae", type=eval, default=False, choices=[True, False])

parser.add_argument("--evaluate", type=eval, default=False, choices=[True, False])

# Regularizations
parser.add_argument("--l2_coeff", type=float, default=0, help="L2 on dynamics.")
parser.add_argument("--dl2_coeff", type=float, default=0, help="Directional L2 on dynamics.")
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--warm_up", type=int, default=-1)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/vae")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)
parser.add_argument("--gpu", type=int, default=0)
args = parser.parse_args()
if args.evaluate:
    assert args.resume is not None, "If you are evaluating, you must give me a checkpoint dummy"


def text_to_numpy(fname):
    print(fname)
    with open(fname, 'r') as f:
        return np.array([[int(i) for i in line.split()] for line in f])


def binarized_mnist(path="./data/binary_mnist.npz"):
    datasets = np.load(path)
    return datasets["train"], datasets["valid"], datasets["test"]


def binarize(x):
    if type(x).__name__ == "ndarray":
        return (np.random.rand(*x.shape) < x).astype(np.float32)
    else:
        return (torch.rand(x.size()) < x).type(torch.float32)


def Encoder():
    return nn.Sequential(
        GatedConv(1, 32, 5, 1, 2),
        GatedConv(32, 32, 5, 2, 2),
        GatedConv(32, 64, 5, 1, 2),
        GatedConv(64, 64, 5, 2, 2),
        GatedConv(64, 64, 5, 1, 2),
        GatedConv(64, 64, 5, 1, 2),
        GatedConv(64, 256, 7, 1, 0),
    )


def Decoder():
    return nn.Sequential(
        GatedConvTranspose(64, 64, 7, 1, 0),
        GatedConvTranspose(64, 64, 5, 1, 2),
        GatedConvTranspose(64, 32, 5, 2, 2, 1),
        GatedConvTranspose(32, 32, 5, 1, 2),
        GatedConvTranspose(32, 32, 5, 2, 2, 1),
        GatedConvTranspose(32, 32, 5, 1, 2),
        GatedConvTranspose(32, 1, 1, 1, 0),
    )


class ODEnet(nn.Module):
    def __init__(self, hidden_dims, nonlinearity, layer_type, input_dim):
        super(ODEnet, self).__init__()
        self.nonlinearity = nonlinearity
        if nonlinearity == "gated":
            assert layer_type == "concat", "only concat layers are supported for gated nonlinearity"
            ls = []
            for i in range(len(hidden_dims)):
                dim_in = input_dim if i == 0 else hidden_dims[i - 1]
                dim_out = hidden_dims[i]
                cur_layer = diffeq_layers.ConcatLinear([dim_in], dim_out, layer_type=GatedLinear)
                ls.append(cur_layer)
            ls.append(diffeq_layers.ConcatLinear([hidden_dims[-1]], input_dim))
            self.layers = nn.ModuleList(ls)
        else:
            nonlinearity = {"tanh": nn.Tanh, "relu": nn.ReLU, "softplus": nn.Softplus, "elu": nn.ELU}[nonlinearity]
            self.nonlinearity = nonlinearity()
            base_layer = {
                "ignore": diffeq_layers.IgnoreLinear,
                "hyper": diffeq_layers.HyperLinear,
                "concat": diffeq_layers.ConcatLinear,
                "blend": diffeq_layers.BlendLinear,
            }[layer_type]
            ls = []
            for i in range(len(hidden_dims)):
                dim_in = input_dim if i == 0 else hidden_dims[i - 1]
                dim_out = hidden_dims[i]
                ls.append(base_layer(dim_in, dim_out))
            ls.append(base_layer(hidden_dims[-1], input_dim))
            self.layers = nn.ModuleList(ls)

    def forward(self, t, y):
        dx = y
        if self.nonlinearity == "gated":
            for layer in self.layers:
                dx = layer(t, dx)
            return dx
        else:
            # for each non-terminal layer apply layer and apply nonlinearity
            for layer in self.layers[:-1]:
                dx = layer(t, dx)
                dx = self.nonlinearity(dx)
            dx = self.layers[-1](t, dx)
            return dx


class VAE(nn.Module):
    def __init__(self, hidden_dim, flow, train_mean, alpha=.001):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.flow = flow
        dim_out = list(self.encoder.modules())[-1].out_channels
        self.hidden_dim = hidden_dim
        self.mu_layer = nn.Linear(dim_out, hidden_dim)
        self.logvar_layer = nn.Linear(dim_out, hidden_dim)
        self.nll = nn.BCEWithLogitsLoss(reduce=False)
        self.input_bias = torch.tensor(train_mean[None, :, :, :])
        train_mean = train_mean * (1. - 2. * alpha) + alpha  # push in from 0 and 1
        train_mean_logit = np.log(train_mean) - np.log(1. - train_mean)
        self.output_bias = torch.tensor(train_mean_logit[None, :, :, :])

    @staticmethod
    def _reparam(mu, logvar):
        eps = torch.randn(mu.size()).to(mu)
        std = torch.exp(logvar / 2.)
        return mu + eps * std

    def forward(self, x, return_recons=False):
        # encode
        batch_size = x.size()[0]
        h = self.encoder(x - self.input_bias.to(x)).view(batch_size, -1)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        # sample z0 ~ N(mu, logvar)
        z0 = self._reparam(mu, logvar)

        logqz0 = normal_logprob(z0, mu, logvar).sum(dim=1, keepdim=True)

        # if no flow z0 is zT
        if self.flow is None:
            zT = z0
            logqzT = logqz0
        # if using flow, integrate the flow to get zT and log q(zT)
        else:
            zT, logqzT = self.flow(z0, logqz0, reverse=False)

        # get generaitve model likelihood and decode
        logpzT = standard_normal_logprob(zT).sum(dim=1)
        x_logit = self.decoder(zT[:, :, None, None]) + self.output_bias.to(x)
        nll = self.nll(x_logit, x).view(batch_size, -1).sum(dim=1)
        logpx = -nll
        kl = logpzT - logqzT[:, 0]

        if return_recons:
            return logpx, kl, torch.sigmoid(x_logit)
        else:
            return logpx, kl

    def sample(self, num_samples=100, z=None):
        if z is None:
            z = torch.randn(num_samples, self.hidden_dim)

        x_logit = self.decoder(z[:, :, None, None]) + self.output_bias.to(x)
        return torch.sigmoid(x_logit)

    def num_evals(self):
        if self.flow is None:
            return 0
        else:
            return self.flow.num_evals()


def normal_logprob(z, mu, logvar):
    return -.5 * (math.log(2. * math.pi) + logvar) - (z - mu).pow(2) / (2. * torch.exp(logvar))


def standard_normal_logprob(z):
    zeros = torch.zeros_like(z)
    return normal_logprob(z, zeros, zeros)


def get_dataset(args):
    if args.data == "mnist":
        datasets = np.load("./data/binary_mnist.npz")
        train_set, valid_set, test_set = datasets["train"], datasets["valid"], datasets["test"]
        train_mean = train_set.mean(axis=0)
    else:
        import scipy.io as sio
        data = sio.loadmat("/Users/wgrathwohl/openai/cnf/data/omniglot.mat")
        trainval = np.transpose(data['data']).reshape([-1, 1, 28, 28])
        test_set = np.transpose(data['testdata']).reshape([-1, 1, 28, 28])
        if args.data == "omniglot_static":
            np.random.seed(5)
            trainval = binarize(trainval)
            test_set = binarize(test_set)

        permutation = np.random.RandomState(seed=123).permutation(trainval.shape[0])
        trainval = trainval[permutation]
        n_val = 1345
        train_set = trainval[:-n_val]
        valid_set = trainval[-n_val:]
        train_mean = train_set.mean(axis=0)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_set, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False)
    logger.info("==>>> total training batch number: {}".format(len(train_loader)))
    logger.info("==>>> total validation batch number: {}".format(len(valid_loader)))
    logger.info("==>>> total testing batch number: {}".format(len(test_loader)))
    return train_loader, valid_loader, test_loader, train_mean


def get_regularization(model):
    reg_loss = 0
    for layer in model.chain:
        if isinstance(layer, layers.CNF):
            reg_loss += layer.odefunc.regularization_loss
    return reg_loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # logger
    utils.makedirs(args.save)
    logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
    logger.info(args)

    # get deivce
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    # if using dynamically binarized dataset we need to binarize the data
    if args.data == "omniglot_dynamic":
        cvt = lambda x: binarize(x.type(torch.float32).to(device))
    else:
        cvt = lambda x: x.type(torch.float32).to(device)

    # load dataset
    train_loader, valid_loader, test_loader, train_mean = get_dataset(args)
    data_shape = train_mean.shape

    hidden_dims = tuple(map(int, args.dims.split(",")))
    gfunc = ODEnet(hidden_dims, args.nonlinearity, args.layer_type, args.hidden_dim)
    if args.vanilla_vae:
        flow = None
        logger.info("Training Vanilla VAE")
    else:
        flow = layers.CNF(
            odefunc=layers.ODEfunc(input_shape=data_shape, diffeq=gfunc, divergence_fn=args.divergence_fn),
            T=args.time_length,
        )

    vae = VAE(args.hidden_dim, flow, train_mean=train_mean)

    logger.info(vae)
    logger.info("Number of trainable parameters: {}".format(count_parameters(vae)))

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        vae.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    vae.to(device)

    # evaluate the model and exit
    if args.evaluate:
        with torch.no_grad():
            start = time.time()
            logger.info("testing...")
            losses = []
            for x in test_loader:
                x = cvt(x)
                logpx, kl = vae(x)
                elbo = logpx + kl
                loss = -elbo.mean()
                losses.append(loss.item())
            loss = np.mean(losses)
            logger.info(" Time {:.4f}, Elbo {:.4f}".format(time.time() - start, -loss))
        exit()

    # For visualization.
    fixed_z = cvt(torch.randn(100, args.hidden_dim))
    for valid_x in valid_loader:
        fixed_x = valid_x
        break

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)

    best_loss = float("inf")
    itr = (args.begin_epoch - 1) * len(train_loader)
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        kl_weight = 1. if args.warm_up < 0 else min(float(epoch) / args.warm_up, 1.0)
        for _, x in enumerate(train_loader):
            start = time.time()
            optimizer.zero_grad()

            # cast data and move to device
            x = cvt(x)

            # compute loss
            logpx, kl = vae(x)
            weighted_elbo = logpx + kl_weight * kl
            loss = -weighted_elbo.mean()
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            steps_meter.update(vae.num_evals())

            if itr % args.log_freq == 0:
                logger.info(
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.4f}({:.4f}) | Steps {:.0f}({:.2f}) | KL-Weight {}".
                    format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                        steps_meter.avg, kl_weight
                    )
                )

            itr += 1

        # compute test loss
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for x in valid_loader:
                    x = cvt(x)
                    logpx, kl = vae(x)
                    elbo = logpx + kl
                    loss = -elbo.mean()
                    losses.append(loss.item())
                loss = np.mean(losses)
                logger.info("Epoch {:04d} | Time {:.4f}, Elbo {:.4f}".format(epoch, time.time() - start, -loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": vae.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        with torch.no_grad():
            sample_filename = os.path.join(args.save, "samples", "{:04d}.jpg".format(epoch))
            recons_filename = os.path.join(args.save, "recons", "{:04d}.jpg".format(epoch))
            orig_filename = os.path.join(args.save, "recons", "orig.jpg".format(epoch))
            utils.makedirs(os.path.dirname(sample_filename))
            utils.makedirs(os.path.dirname(recons_filename))
            utils.makedirs(os.path.dirname(orig_filename))
            if args.data == "mnist":
                # samples
                generated_samples = vae.sample(z=fixed_z)
                save_image(generated_samples, sample_filename, nrow=10)
                # reconstructions
                x = cvt(fixed_x)
                _, _, recons = vae(x, return_recons=True)
                save_image(recons, recons_filename, nrow=10)
                save_image(fixed_x, orig_filename, nrow=10)
