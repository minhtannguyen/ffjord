import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import lib.layers as layers
import lib.utils as utils
import lib.multiscale_parallel as multiscale_parallel
import lib.modules as modules
import lib.thops as thops

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log

from tensorboardX import SummaryWriter

# go fast boi!!
torch.backends.cudnn.benchmark = True
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams']
parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument("--data", choices=["mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
parser.add_argument("--dims", type=str, default="8,32,32,8")
parser.add_argument("--strides", type=str, default="2,2,1,-2,-2")
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')

parser.add_argument("--conv", type=eval, default=True, choices=[True, False])
parser.add_argument(
    "--layer_type", type=str, default="ignore",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument(
    "--nonlinearity", type=str, default="softplus", choices=["tanh", "relu", "softplus", "elu", "swish"]
)

parser.add_argument("--seed", type=int, default=0)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument("--imagesize", type=int, default=None)
parser.add_argument("--alpha", type=float, default=1e-6)
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)

parser.add_argument("--num_epochs", type=int, default=1000)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument(
    "--batch_size_schedule", type=str, default="", help="Increases the batchsize at every given epoch, dash separated."
)
parser.add_argument("--test_batch_size", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup_iters", type=float, default=1000)
parser.add_argument("--weight_decay", type=float, default=0.0)
parser.add_argument("--spectral_norm_niter", type=int, default=10)
parser.add_argument("--weight_y", type=float, default=0.5)
parser.add_argument("--concat_size", type=int, default=0)
parser.add_argument("--drop_noise_ratio", type=float, default=0.0)

parser.add_argument("--add_noise", type=eval, default=True, choices=[True, False])
parser.add_argument("--batch_norm", type=eval, default=False, choices=[True, False])
parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--autoencode', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--multiscale', type=eval, default=False, choices=[True, False])
parser.add_argument('--parallel', type=eval, default=False, choices=[True, False])
parser.add_argument('--conditional', type=eval, default=False, choices=[True, False])
parser.add_argument('--controlled_tol', type=eval, default=False, choices=[True, False])
parser.add_argument("--train_mode", choices=["semisup", "sup", "unsup"], type=str, default="semisup")

# Regularizations
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument("--time_penalty", type=float, default=0, help="Regularization on the end_time.")
parser.add_argument(
    "--max_grad_norm", type=float, default=1e10,
    help="Max norm of graidents (default is just stupidly high to avoid any clipping)"
)

parser.add_argument("--begin_epoch", type=int, default=1)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--save", type=str, default="experiments/cnf")
parser.add_argument("--val_freq", type=int, default=1)
parser.add_argument("--log_freq", type=int, default=1)

args = parser.parse_args()

if args.controlled_tol:
    import lib.odenvp_conditional_tol as odenvp
else:
    import lib.odenvp_conditional as odenvp
    
# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__)) # write to log file
writer = SummaryWriter(os.path.join(args.save, 'tensorboard')) # write to tensorboard

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)


def add_noise_concat(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        x_pil = tforms.functional.to_pil_image(x)
        x_hf = tforms.functional.to_tensor(tforms.functional.hflip(x_pil))
        x_vf = tforms.functional.to_tensor(tforms.functional.vflip(x_pil))
        x = torch.cat((x, x_hf, x_vf), 0)
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
    else:
        x = torch.cat((x, torch.zeros(args.concat_size, x.shape[1], x.shape[2]).to(x)),0)
    return x


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
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
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise_concat])

    if args.data == "mnist":
        im_dim = 1 + args.concat_size
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="../data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="../data", train=False, transform=trans(im_size), download=True)
    elif args.data == "svhn":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.SVHN(root="../data", split="train", transform=trans(im_size), download=True)
        test_set = dset.SVHN(root="../data", split="test", transform=trans(im_size), download=True)
    elif args.data == "cifar10":
        im_dim = 3
        im_size = 32 if args.imagesize is None else args.imagesize
        train_set = dset.CIFAR10(
            root="../data", train=True, transform=tforms.Compose([
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise_concat,
            ]), download=True
        )
        test_set = dset.CIFAR10(root="../data", train=False, transform=trans(im_size), download=True)
    elif args.data == 'celeba':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.CelebA(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )
    elif args.data == 'lsun_church':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.LSUN(
            '../data', ['church_outdoor_train'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )
        test_set = dset.LSUN(
            '../data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )    
    elif args.data == 'imagenet_64':
        im_dim = 3
        im_size = 64 if args.imagesize is None else args.imagesize
        train_set = dset.ImageFolder(
            train=True, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.RandomHorizontalFlip(),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )
        test_set = dset.ImageFolder(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise_concat,
            ])
        )
    
    data_shape = (im_dim, im_size, im_size)
    if not args.conv:
        data_shape = (im_dim * im_size * im_size,)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=args.test_batch_size, shuffle=False, drop_last=True
    )
    return train_set, test_loader, data_shape


def compute_bits_per_dim(x, model):
    zero = torch.zeros(x.shape[0], 1).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module
    
    z, delta_logp = model(x, zero)  # run model forward

    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)

    return bits_per_dim

def compute_bits_per_dim_conditional(x, y, model):
    zero = torch.zeros(x.shape[0], 1).to(x)
    y_onehot = thops.onehot(y, num_classes=model.module.y_class).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module
    
    z, delta_logp = model(x, zero)  # run model forward
    
    # prior
    mean, logs = model.module._prior(y_onehot)

    logpz = modules.GaussianDiag.logp(mean, logs, z).view(-1,1)  # logp(z)
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    
    # compute xentropy loss
    y_logits = model.module.project_class(z)
    loss_xent = model.module.loss_class(y_logits, y.to(x.get_device()))
    y_predicted = np.argmax(y_logits.cpu().detach().numpy(), axis=1)

    return bits_per_dim, loss_xent, y_predicted

def create_model(args, data_shape, regularization_fns):
    hidden_dims = tuple(map(int, args.dims.split(",")))
    strides = tuple(map(int, args.strides.split(",")))

    if args.multiscale:
        model = odenvp.ODENVP(
            (args.batch_size, *data_shape),
            n_blocks=args.num_blocks,
            intermediate_dims=hidden_dims,
            nonlinearity=args.nonlinearity,
            alpha=args.alpha,
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns, "solver": args.solver, "atol": args.atol, "rtol": args.rtol},)
    elif args.parallel:
        model = multiscale_parallel.MultiscaleParallelCNF(
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
                    train_T=args.train_T,
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
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = create_model(args, data_shape, regularization_fns)

    if args.spectral_norm: add_spectral_norm(model, logger)
    set_cnf_options(args, model)

    logger.info(model)
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    
    writer.add_text('info', "Number of trainable parameters: {}".format(count_parameters(model)))

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # set initial iter
    itr = 1
    
    # set the meters
    time_epoch_meter = utils.RunningAverageMeter(0.97)
    time_meter = utils.RunningAverageMeter(0.97)
    loss_meter = utils.RunningAverageMeter(0.97) # track total loss
    nll_meter = utils.RunningAverageMeter(0.97) # track negative log-likelihood
    xent_meter = utils.RunningAverageMeter(0.97) # track xentropy score
    error_meter = utils.RunningAverageMeter(0.97) # track error score
    steps_meter = utils.RunningAverageMeter(0.97)
    grad_meter = utils.RunningAverageMeter(0.97)
    tt_meter = utils.RunningAverageMeter(0.97)

    # restore parameters
    if args.resume is not None:
        checkpt = torch.load(args.resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpt["state_dict"])
        if "optim_state_dict" in checkpt.keys():
            optimizer.load_state_dict(checkpt["optim_state_dict"])
            # Manually move optimizer state to device.
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = cvt(v)
        args.begin_epoch = checkpt['epoch'] + 1
        itr = checkpt['iter'] + 1
        time_epoch_meter.set(checkpt['epoch_time_avg'])
        time_meter.set(checkpt['time_train'])
        loss_meter.set(checkpt['loss_train'])
        nll_meter.set(checkpt['bits_per_dim_train'])
        xent_meter.set(checkpt['xent_train'])
        error_meter.set(checkpt['error_train'])
        steps_meter.set(checkpt['nfe_train'])
        grad_meter.set(checkpt['grad_train'])
        tt_meter.set(checkpt['total_time_train'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    if args.conditional:
        fixed_y = torch.from_numpy(np.arange(model.module.y_class)).repeat(model.module.y_class).type(torch.long).to(device, non_blocking=True)
        fixed_y_onehot = thops.onehot(fixed_y, num_classes=model.module.y_class)
        with torch.no_grad():
            mean, logs = model.module._prior(fixed_y_onehot)
            fixed_z = modules.GaussianDiag.sample(mean, logs)
    else:
        fixed_z = cvt(torch.randn(100, *data_shape))
    

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    best_loss_nll = float("inf")
    if args.conditional: best_error_score = float("inf")
    
    for epoch in range(args.begin_epoch, args.num_epochs + 1):
        start_epoch = time.time()
        model.train()
        train_loader = get_train_loader(train_set, epoch)
        for _, (x, y) in enumerate(train_loader):
            start = time.time()
            update_lr(optimizer, itr)
            optimizer.zero_grad()

            if not args.conv:
                x = x.view(x.shape[0], -1)

            # cast data and move to device
            x = cvt(x)
            
            # compute loss
            if args.conditional:
                loss_nll, loss_xent, y_predicted = compute_bits_per_dim_conditional(x, y, model)
                if args.train_mode == "semisup":
                    loss =  loss_nll + args.weight_y * loss_xent
                elif args.train_mode == "sup":
                    loss =  loss_xent
                elif args.train_mode == "unsup":
                    loss =  loss_nll
                else:
                    raise ValueError('Choose supported train_mode: semisup, sup, unsup')
                error_score = 1. - np.mean(y_predicted.astype(int) == y.numpy())   
                
            else:
                loss = compute_bits_per_dim(x, model)
                loss_nll, loss_xent, error_score = loss, 0., 0.
            
            if regularization_coeffs:
                reg_states = get_regularization(model, regularization_coeffs)
                reg_loss = sum(
                    reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
                )
                loss = loss + reg_loss
            total_time = count_total_time(model)
            loss = loss + total_time * args.time_penalty

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()

            if args.spectral_norm: spectral_norm_power_iteration(model, args.spectral_norm_niter)
            
            time_meter.update(time.time() - start)
            loss_meter.update(loss.item())
            nll_meter.update(loss_nll.item())
            if args.conditional:
                xent_meter.update(loss_xent.item())
            else:
                xent_meter.update(loss_xent)
            error_meter.update(error_score)
            steps_meter.update(count_nfe(model))
            grad_meter.update(grad_norm)
            tt_meter.update(total_time)
            
            # write to tensorboard
            writer.add_scalars('time', {'train_iter': time_meter.val}, itr)
            writer.add_scalars('loss', {'train_iter': loss_meter.val}, itr)
            writer.add_scalars('bits_per_dim', {'train_iter': nll_meter.val}, itr)
            writer.add_scalars('xent', {'train_iter': xent_meter.val}, itr)
            writer.add_scalars('error', {'train_iter': error_meter.val}, itr)
            writer.add_scalars('nfe', {'train_iter': steps_meter.val}, itr)
            writer.add_scalars('grad', {'train_iter': grad_meter.val}, itr)
            writer.add_scalars('total_time', {'train_iter': tt_meter.val}, itr)

            if itr % args.log_freq == 0:
                log_message = (
                    "Iter {:04d} | Time {:.4f}({:.4f}) | Bit/dim {:.4f}({:.4f}) | Xent {:.4f}({:.4f}) | Loss {:.4f}({:.4f}) | Error {:.4f}({:.4f}) "
                    "Steps {:.0f}({:.2f}) | Grad Norm {:.4f}({:.4f}) | Total Time {:.2f}({:.2f})".format(
                        itr, time_meter.val, time_meter.avg, nll_meter.val, nll_meter.avg, xent_meter.val, xent_meter.avg, loss_meter.val, loss_meter.avg, error_meter.val, error_meter.avg, steps_meter.val, steps_meter.avg, grad_meter.val, grad_meter.avg, tt_meter.val, tt_meter.avg
                    )
                )
                if regularization_coeffs:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
                logger.info(log_message)
                writer.add_text('info', log_message, itr)

            itr += 1
        
        # compute test loss
        model.eval()
        if epoch % args.val_freq == 0:
            with torch.no_grad():
                # write to tensorboard
                writer.add_scalars('time', {'train_epoch': time_meter.avg}, epoch)
                writer.add_scalars('loss', {'train_epoch': loss_meter.avg}, epoch)
                writer.add_scalars('bits_per_dim', {'train_epoch': nll_meter.avg}, epoch)
                writer.add_scalars('xent', {'train_epoch': xent_meter.avg}, epoch)
                writer.add_scalars('error', {'train_epoch': error_meter.avg}, epoch)
                writer.add_scalars('nfe', {'train_epoch': steps_meter.avg}, epoch)
                writer.add_scalars('grad', {'train_epoch': grad_meter.avg}, epoch)
                writer.add_scalars('total_time', {'train_epoch': tt_meter.avg}, epoch)
                
                start = time.time()
                logger.info("validating...")
                writer.add_text('info', "validating...", epoch)
                losses_nll = []; losses_xent = []; losses = []
                total_correct = 0
                
                for (x, y) in test_loader:
                    if not args.conv:
                        x = x.view(x.shape[0], -1)
                    x = cvt(x)
                    if args.conditional:
                        loss_nll, loss_xent, y_predicted = compute_bits_per_dim_conditional(x, y, model)
                        if args.train_mode == "semisup":
                            loss =  loss_nll + args.weight_y * loss_xent
                        elif args.train_mode == "sup":
                            loss =  loss_xent
                        elif args.train_mode == "unsup":
                            loss =  loss_nll
                        else:
                            raise ValueError('Choose supported train_mode: semisup, sup, unsup')
                        total_correct += np.sum(y_predicted.astype(int) == y.numpy())
                    else:
                        loss = compute_bits_per_dim(x, model)
                        loss_nll, loss_xent = loss, 0.
                    losses_nll.append(loss_nll.cpu().numpy()); losses.append(loss.cpu().numpy())
                    if args.conditional: 
                        losses_xent.append(loss_xent.cpu().numpy())
                    else:
                        losses_xent.append(loss_xent)
                
                loss_nll = np.mean(losses_nll); loss_xent = np.mean(losses_xent); loss = np.mean(losses)
                error_score =  1. - total_correct / len(test_loader.dataset)
                time_epoch_meter.update(time.time() - start_epoch)
                
                # write to tensorboard
                test_time_spent = time.time() - start
                writer.add_scalars('time', {'validation': test_time_spent}, epoch)
                writer.add_scalars('epoch_time', {'validation': time_epoch_meter.val}, epoch)
                writer.add_scalars('bits_per_dim', {'validation': loss_nll}, epoch)
                writer.add_scalars('xent', {'validation': loss_xent}, epoch)
                writer.add_scalars('loss', {'validation': loss}, epoch)
                writer.add_scalars('error', {'validation': error_score}, epoch)
                
                log_message = "Epoch {:04d} | Time {:.4f}, Epoch Time {:.4f}({:.4f}), Bit/dim {:.4f}, Xent {:.4f}, Loss {:.4f}, Error {:.4f}".format(epoch, time.time() - start, time_epoch_meter.val, time_epoch_meter.avg, loss_nll, loss_xent, loss, error_score)
                logger.info(log_message)
                writer.add_text('info', log_message, epoch)
                
                for name, param in model.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                    
                
                utils.makedirs(args.save)
                torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "iter": itr-1,
                        "error": error_score,
                        "loss": loss,
                        "xent": loss_xent,
                        "bits_per_dim": loss_nll,
                        "epoch_time": time_epoch_meter.val,
                        "epoch_time_avg": time_epoch_meter.avg,
                        "time": test_time_spent,
                        "error_train": error_meter.avg,
                        "loss_train": loss_meter.avg,
                        "xent_train": xent_meter.avg,
                        "bits_per_dim_train": nll_meter.avg,
                        "total_time_train": tt_meter.avg,
                        "time_train": time_meter.avg,
                        "nfe_train": steps_meter.avg,
                        "grad_train": grad_meter.avg,
                    }, os.path.join(args.save, "current_checkpt.pth"))
                
                if loss_nll < best_loss_nll:
                    best_loss_nll = loss_nll
                    utils.makedirs(args.save)
                    torch.save({
                        "args": args,
                        "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "iter": itr-1,
                        "error": error_score,
                        "loss": loss,
                        "xent": loss_xent,
                        "bits_per_dim": loss_nll,
                        "epoch_time": time_epoch_meter.val,
                        "epoch_time_avg": time_epoch_meter.avg,
                        "time": test_time_spent,
                        "error_train": error_meter.avg,
                        "loss_train": loss_meter.avg,
                        "xent_train": xent_meter.avg,
                        "bits_per_dim_train": nll_meter.avg,
                        "total_time_train": tt_meter.avg,
                        "time_train": time_meter.avg,
                        "nfe_train": steps_meter.avg,
                        "grad_train": grad_meter.avg,
                    }, os.path.join(args.save, "best_nll_checkpt.pth"))
                    
                if args.conditional:
                    if error_score < best_error_score:
                        best_error_score = error_score
                        utils.makedirs(args.save)
                        torch.save({
                            "args": args,
                            "state_dict": model.module.state_dict() if torch.cuda.is_available() else model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "iter": itr-1,
                            "error": error_score,
                            "loss": loss,
                            "xent": loss_xent,
                            "bits_per_dim": loss_nll,
                            "epoch_time": time_epoch_meter.val,
                            "epoch_time_avg": time_epoch_meter.avg,
                            "time": test_time_spent,
                            "error_train": error_meter.avg,
                            "loss_train": loss_meter.avg,
                            "xent_train": xent_meter.avg,
                            "bits_per_dim_train": nll_meter.avg,
                            "total_time_train": tt_meter.avg,
                            "time_train": time_meter.avg,
                            "nfe_train": steps_meter.avg,
                            "grad_train": grad_meter.avg,
                        }, os.path.join(args.save, "best_error_checkpt.pth"))
                        

        # visualize samples and density
        with torch.no_grad():
            fig_filename = os.path.join(args.save, "figs", "{:04d}.jpg".format(epoch))
            utils.makedirs(os.path.dirname(fig_filename))
            generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
            generated_samples = generated_samples[:,0:1,:,:]
            save_image(generated_samples, fig_filename, nrow=10)
            writer.add_images('generated_images', generated_samples.repeat(1,3,1,1), epoch)
