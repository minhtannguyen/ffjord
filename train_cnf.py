import argparse
import os
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.spectral_norm as spectral_norm
import lib.layers as layers
import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.utils as utils
import lib.odenvp as odenvp

# go fast boi!!
torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10"], type=str, default="mnist")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type", type=str, default="ignore", choices=["ignore", "concat", "concatcoord", "hyper", "blend"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu"])
parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument("--time_length", type=float, default=None)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument(
    "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
)
parser.add_argument("--test_batch_size", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--autoencode', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])

# Regularizations
parser.add_argument("--l2_coeff", type=float, default=0, help="L2 on dynamics.")
parser.add_argument("--dl2_coeff", type=float, default=0, help="Directional L2 on dynamics.")
parser.add_argument(
    "--max_grad_norm", type=float, default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping"
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=10)

args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    return x


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / args.warmup_iters, 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def get_train_loader(train_set, epoch):
    if args.batch_size_schedule != "":
        epochs = [0] + list(map(int, args.batch_size_schedule.split("-")))
        n_passed = sum(np.array(epochs) <= epoch)
        current_batch_size = int(args.batch_size * n_passed)
    else:
        current_batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=current_batch_size, shuffle=True, drop_last=True, pin_memory=True
    )
    logger.info("===> Using batch size {}. Total {} iterations/epoch.".format(current_batch_size, len(train_loader)))
    return train_loader


def get_dataset(args):
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="./data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="./data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(root="./data", train=True, transform=trans(im_size), download=True)
        test_set = dset.CIFAR10(root="./data", train=False, transform=trans(im_size), download=True)
    elif args.dataset == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(args.imagesize),
                tforms.ToTensor(),
                add_noise,
            ])
        )
    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model, regularization_coeffs=None):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module

    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    if regularization_coeffs:
        regularization = get_regularization(model, regularization_coeffs)
    else:
        regularization = torch.tensor(0.).to(bits_per_dim)

    return bits_per_dim, regularization


def count_nfe(model):
    class AccNumEvals(object):
        def __init__(self):
            self.num_evals = 0

        def __call__(self, module):
            if isinstance(module, layers.CNF):
                self.num_evals += module.num_evals()

    accumulator = AccNumEvals()
    model.apply(accumulator)
    return accumulator.num_evals


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_regularization_fns():
    regularization_fns = []
    regularization_coeffs = []
    if args.l2_coeff != 0:
        regularization_coeffs.append(args.l2_coeff)
        regularization_fns.append(reg_lib.l2_regularzation_fn)
    if args.dl2_coeff != 0:
        regularization_coeffs.append(args.dl2_coeff)
        regularization_fns.append(reg_lib.directional_l2_regularization_fn)
    regularization_coeffs = tuple(regularization_coeffs)
    regularization_fns = tuple(regularization_fns)
    return regularization_fns, regularization_coeffs


def get_regularization(model, regularization_coeffs):
    if len(regularization_coeffs) == 0:
        return None

    acc_reg_states = tuple([0.] * len(regularization_coeffs))
    for module in model.modules():
        if isinstance(module, layers.CNF):
            acc_reg_states = tuple(acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states()))
    return sum(state * coeff for state, coeff in zip(acc_reg_states, regularization_coeffs))


def add_spectral_norm(model):
    """Applies spectral norm to all modules within the scope of a CNF."""

    def apply_spectral_norm(module):
        if 'weight' in module._parameters:
            logger.info("Adding spectral norm to {}".format(module))
            spectral_norm.inplace_spectral_norm(module, 'weight')

    def find_cnf(module):
        if isinstance(module, layers.CNF):
            module.apply(apply_spectral_norm)
        else:
            for child in module.children():
                find_cnf(child)

    find_cnf(model)


def spectral_norm_power_iteration(model, n_power_iterations=1):
    def recursive_power_iteration(module):
        if hasattr(module, spectral_norm.POWER_ITERATION_FN):
            getattr(module, spectral_norm.POWER_ITERATION_FN)(n_power_iterations)

    model.apply(recursive_power_iteration)


def set_cnf_options(model):
    def _set(module):
        if isinstance(module, layers.CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            if args.step_size is not None:
                module.solver_options['step_size'] = args.step_size

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.test_atol if args.test_atol else args.atol
            module.test_rtol = args.test_rtol if args.test_rtol else args.rtol

        if isinstance(module, layers.ODEfunc):
            module.rademacher = args.rademacher
            module.residual = args.residual

    model.apply(_set)


def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            alpha=args.alpha,
            time_length=args.time_length,
        )
    else:
        if args.autoencode:

            def build_cnf():
                autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.AutoencoderODEfunc(
                    autoencoder_diffeq=autoencoder_diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf
        else:

            def build_cnf():
                diffeq = layers.ODEnet(
                    hidden_dims=hidden_dims,
                    input_shape=data_shape,
                    strides=strides,
                    conv=args.conv,
                    layer_type=args.layer_type,
                    nonlinearity=args.nonlinearity,
                )
                odefunc = layers.ODEfunc(
                    diffeq=diffeq,
                    divergence_fn=args.divergence_fn,
                    residual=args.residual,
                    rademacher=args.rademacher,
                )
                cnf = layers.CNF(
                    odefunc=odefunc,
                    T=args.time_length,
                    regularization_fns=regularization_fns,
                    solver=args.solver,
                )
                return cnf

        chain = [layers.LogitTransform(alpha=args.alpha)] if args.alpha > 0 else [layers.ZeroMeanTransform()]
        chain = chain + [build_cnf() for _ in range(args.num_blocks)]
        if args.batch_norm:
            chain.append(layers.MovingBatchNorm2d(data_shape[0]))
        model = layers.SequentialFlow(chain)
    return model


if __name__ == "__main__":

    # get deivce
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)

    # load dataset
    train_set, test_loader, data_shape = get_dataset(args)

    # build model
    regularization_fns, regularization_coeffs = create_regularization_fns()
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    fixed_z = cvt(torch.randn(100, *data_shape))

    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97)
    steps_meter = utils.RunningAverageMeter(0.97)
    reg_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)

    best_loss = float("inf")
    itr = 0
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        model.train()
        train_loader = get_train_loader(train_set, epoch)
        if args.spectral_norm: spectral_norm_power_iteration(model, 100)
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            if args.spectral_norm: spectral_norm_power_iteration(model)

            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)

            # compute loss
            bits_per_dim, regularization = compute_bits_per_dim(x, model, regularization_coeffs)
            (bits_per_dim + regularization).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            time_meter.update(time.time() - start)
            loss_meter.update(bits_per_dim.item())
            steps_meter.update(count_nfe(model))
            reg_meter.update(regularization.item())
            grad_meter.update(grad_norm)

            if itr % args.log_freq == 0:
                logger.info(
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | "
                    "Steps {:.0f}({:.2f}) | Reg {:.4f}({:.4f}) | Grad Norm {:.4f}({:.4f})".format(
                        itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, steps_meter.val,
                        steps_meter.avg, reg_meter.val, reg_meter.avg, grad_meter.val, grad_meter.avg
                    )
                )

            itr += 1

        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                start = time.time()
                logger.info("validating...")
                losses = []
                for (x, y) in test_loader:
                    if not args.conv:
                        x = x.view(x.shape[0], -1)
                    x = cvt(x)
                    loss, _ = compute_bits_per_dim(x, model)
                    losses.append(loss)

                loss = np.mean(losses)
                logger.info("Epoch {:04d} | Time {:.4f}, Bit/dim {:.4f}".format(epoch, time.time() - start, loss))
                if loss < best_loss:
                    best_loss = loss
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    }, os.path.join(args.save, "checkpt.pth"))

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            save_image(generated_samples, fig_filename, nrow=10)
