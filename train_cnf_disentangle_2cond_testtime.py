import argparse
import os
import time
import numpy as np

import torch
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as tforms
from torchvision.utils import save_image

import torch.utils.data as data
from torch.utils.data import Dataset

from PIL import Image
import os.path
import errno
import codecs

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
parser.add_argument("--data", choices=["colormnist", "mnist", "svhn", "cifar10", 'lsun_church'], type=str, default="mnist")
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
parser.add_argument("--y_class", type=int, default=10)
parser.add_argument("--y_color", type=int, default=10)

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
parser.add_argument("--condition_ratio", type=float, default=0.5)
parser.add_argument("--dropout_rate", type=float, default=0.0)


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
    import lib.odenvp_conditional_2cond as odenvp
    
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

class ColorMNIST(data.Dataset):
    """
    ColorMNIST
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            
            self.train_data = np.tile(self.train_data[:, :, :, np.newaxis], 3)
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            
            self.test_data = np.tile(self.test_data[:, :, :, np.newaxis], 3)
        
        self.pallette = [[31, 119, 180],
                         [255, 127, 14],
                         [44, 160, 44],
                         [214, 39, 40],
                         [148, 103, 189],
                         [140, 86, 75],
                         [227, 119, 194],
                         [127, 127, 127],
                         [188, 189, 34],
                         [23, 190, 207]]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index].copy(), self.train_labels[index]
        else:
            img, target = self.test_data[index].copy(), self.test_labels[index]
        
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        y_color_digit = np.random.randint(0, args.y_color)
        c_digit = self.pallette[y_color_digit]
        
        img[:, :, 0] = img[:, :, 0] / 255 * c_digit[0]
        img[:, :, 1] = img[:, :, 1] / 255 * c_digit[1]
        img[:, :, 2] = img[:, :, 2] / 255 * c_digit[2]
        
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, [target,torch.from_numpy(np.array(y_color_digit))]

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def add_noise(x):
    """
    [0, 1] -> [0, 255] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * 255 + noise
        x = x / 256
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
    trans = lambda im_size: tforms.Compose([tforms.Resize(im_size), tforms.ToTensor(), add_noise])

    if args.data == "mnist":
        im_dim = 1
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = dset.MNIST(root="../data", train=True, transform=trans(im_size), download=True)
        test_set = dset.MNIST(root="../data", train=False, transform=trans(im_size), download=True)
    elif args.data == "colormnist":
        im_dim = 3
        im_size = 28 if args.imagesize is None else args.imagesize
        train_set = ColorMNIST(root="../data", train=True, transform=trans(im_size), download=True)
        test_set = ColorMNIST(root="../data", train=False, transform=trans(im_size), download=True)
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
                add_noise,
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
                add_noise,
            ])
        )
        test_set = dset.CelebA(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
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
                add_noise,
            ])
        )
        test_set = dset.LSUN(
            '../data', ['church_outdoor_val'], transform=tforms.Compose([
                tforms.Resize(96),
                tforms.RandomCrop(64),
                tforms.Resize(im_size),
                tforms.ToTensor(),
                add_noise,
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
                add_noise,
            ])
        )
        test_set = dset.ImageFolder(
            train=False, transform=tforms.Compose([
                tforms.ToPILImage(),
                tforms.Resize(im_size),
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

def compute_bits_per_dim_conditional(x, y, y_color, model):
    zero = torch.zeros(x.shape[0], 1).to(x)
    y_onehot = thops.onehot(y, num_classes=model.module.y_class).to(x)
    y_onehot_color = thops.onehot(y_color, num_classes=model.module.y_color).to(x)

    # Don't use data parallelize if batch size is small.
    # if x.shape[0] < 200:
    #     model = model.module
    
    z, delta_logp = model(x, zero)  # run model forward
    
    dim_sup = int(args.condition_ratio * np.prod(z.size()[1:]))
    
    # prior
    mean, logs = model.module._prior(y_onehot)
    mean_color, logs_color = model.module._prior_color(y_onehot_color)

    logpz_sup = modules.GaussianDiag.logp(mean, logs, z[:, 0:dim_sup]).view(-1,1)  # logp(z)_sup
    logpz_color_sup = modules.GaussianDiag.logp(mean_color, logs_color, z[:, dim_sup:(2*dim_sup)]).view(-1,1)  # logp(z)_color_sup
    logpz_unsup = standard_normal_logprob(z[:, (2*dim_sup):]).view(z.shape[0], -1).sum(1, keepdim=True)
    logpz = logpz_sup + logpz_color_sup + logpz_unsup
    logpx = logpz - delta_logp

    logpx_per_dim = torch.sum(logpx) / x.nelement()  # averaged over batches
    bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
    
    # dropout
    if args.dropout_rate > 0:
        zsup = model.module.dropout(z[:, 0:dim_sup])
        zcolorsup = model.module.dropout_color(z[:, dim_sup:(2*dim_sup)])
    else:
        zsup = z[:, 0:dim_sup]
        zcolorsup = z[:, dim_sup:(2*dim_sup)]
    
    # compute xentropy loss
    y_logits = model.module.project_class(zsup)
    loss_xent = model.module.loss_class(y_logits, y.to(x.get_device()))
    y_predicted = np.argmax(y_logits.cpu().detach().numpy(), axis=1)
    
    y_logits_color = model.module.project_color(zcolorsup)
    loss_xent_color = model.module.loss_class(y_logits_color, y_color.to(x.get_device()))
    y_color_predicted = np.argmax(y_logits_color.cpu().detach().numpy(), axis=1)

    return bits_per_dim, loss_xent, loss_xent_color, y_predicted, y_color_predicted

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
            cnf_kwargs={"T": args.time_length, "train_T": args.train_T, "regularization_fns": regularization_fns, "solver": args.solver, "atol": args.atol, "rtol": args.rtol},
            condition_ratio=args.condition_ratio,
            dropout_rate=args.dropout_rate,
            y_class = args.y_class,
            y_color = args.y_color)
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
    xent_color_meter = utils.RunningAverageMeter(0.97) # track xentropy score
    error_meter = utils.RunningAverageMeter(0.97) # track error score
    error_color_meter = utils.RunningAverageMeter(0.97)
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
        xent_color_meter.set(checkpt['xent_train_color'])
        error_meter.set(checkpt['error_train'])
        error_color_meter.set(checkpt['error_train_color'])
        steps_meter.set(checkpt['nfe_train'])
        grad_meter.set(checkpt['grad_train'])
        tt_meter.set(checkpt['total_time_train'])

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    # For visualization.
    if args.conditional:
        # fixed_y = torch.from_numpy(np.arange(model.module.y_class)).repeat(model.module.y_class).type(torch.long).to(device, non_blocking=True)
        fixed_y = torch.zeros(model.module.y_class**2).type(torch.long).to(device, non_blocking=True)
        fixed_y_onehot = thops.onehot(fixed_y, num_classes=model.module.y_class)
        
        #fixed_y_color = torch.from_numpy(np.arange(model.module.y_color)).repeat(model.module.y_color).type(torch.long).to(device, non_blocking=True)
        fixed_y_color = torch.zeros(model.module.y_color**2).type(torch.long).to(device, non_blocking=True)
        fixed_y_onehot_color = thops.onehot(fixed_y_color, num_classes=model.module.y_color)
        
        with torch.no_grad():
            mean, logs = model.module._prior(fixed_y_onehot)
            mean_color, logs_color = model.module._prior_color(fixed_y_onehot_color)
            fixed_z_sup = modules.GaussianDiag.sample(mean, logs)
            #fixed_z_sup = mean
            fixed_z_color_sup = modules.GaussianDiag.sample(mean_color, logs_color)
            #fixed_z_color_sup = mean_color
            
            dim_unsup = np.prod(data_shape) - np.prod(fixed_z_sup.shape[1:]) - np.prod(fixed_z_color_sup.shape[1:])
            fixed_z_unsup = cvt(torch.randn(model.module.y_class**2, dim_unsup))
            #fixed_z_unsup = cvt(torch.zeros(model.module.y_class**2, dim_unsup))
            fixed_z = torch.cat((fixed_z_sup, fixed_z_color_sup, fixed_z_unsup),1)
    else:
        fixed_z = cvt(torch.randn(100, *data_shape))
    

    if args.spectral_norm and not args.resume: spectral_norm_power_iteration(model, 500)

    
    # visualize samples and density
    with torch.no_grad():
        fig_filename = os.path.join(args.save, "figs", "test_img.jpg")
        utils.makedirs(os.path.dirname(fig_filename))
        generated_samples = model(fixed_z, reverse=True).view(-1, *data_shape)
        save_image(generated_samples, fig_filename, nrow=10)
        import ipdb; ipdb.set_trace()
