import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import lib.layers.wrappers.cnf_regularization as reg_lib
import lib.spectral_norm as spectral_norm
import lib.layers as layers
from lib.layers.odefunc import divergence_bf, divergence_approx

import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from lib import modules

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular

from diagnostics.viz_toy import save_trajectory, trajectory_to_video

from tensorboardX import SummaryWriter

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)

parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)

parser.add_argument('--dims', type=str, default='20-20')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=0.5)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=10)

args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))
writer = SummaryWriter(os.path.join(args.save, 'tensorboard')) # write to tensorboard

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0

logger.info(args)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def generate_spiral2d(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a=0.,
                      b=1.,
                      savefig=True):
    """Parametric formula for 2d spiral is `r = a + b * theta`.

    Args:
      nspiral: number of spirals, i.e. batch dimension
      ntotal: total number of datapoints per spiral
      nsample: number of sampled datapoints for model fitting per spiral
      start: spiral starting theta value
      stop: spiral ending theta value
      noise_std: observation noise standard deviation
      a, b: parameters of the Archimedean spiral
      savefig: plot the ground truth for sanity check

    Returns: 
      Tuple where first element is true trajectory of size (nspiral, ntotal, 2),
      second element is noisy observations of size (nspiral, nsample, 2),
      third element is timestamps of size (ntotal,),
      and fourth element is timestamps of size (nsample,)
    """

    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal)
    samp_ts = orig_ts[:nsample]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts
    rs_cw = a + b * 50. / zs_cw
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw)
    orig_traj_cw = np.stack((xs, ys), axis=1)

    zs_cc = orig_ts
    rw_cc = a + b * zs_cc
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc)
    orig_traj_cc = np.stack((xs, ys), axis=1)

    if savefig:
        plt.figure()
        plt.plot(orig_traj_cw[:, 0], orig_traj_cw[:, 1], label='clock')
        plt.plot(orig_traj_cc[:, 0], orig_traj_cc[:, 1], label='counter clock')
        plt.legend()
        plt.savefig('./ground_truth.png', dpi=500)
        print('Saved ground truth spiral at {}'.format('./ground_truth.png'))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    samp_trajs_next = []
    for _ in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample
        t0_idx_next = t0_idx + 1

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        orig_traj = orig_traj_cc if cc else orig_traj_cw
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        # samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)
        
        samp_traj_next = orig_traj[t0_idx_next:t0_idx_next + nsample, :].copy()
        # samp_traj_next += npr.randn(*samp_traj_next.shape) * noise_std
        samp_trajs_next.append(samp_traj_next)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    samp_trajs_next = np.stack(samp_trajs_next, axis=0)

    return orig_trajs, samp_trajs, samp_trajs_next, orig_ts, samp_ts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class Encoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def l2loss(x, mean):
    return (x - mean) ** 2.


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

if __name__ == '__main__':
    latent_dim = 2
    nhidden = 20
    rnn_nhidden = 25
    obs_dim = 2
    nspiral = 1000
    start = 0.
    stop = 6 * np.pi
    noise_std = .3
    a = 0.
    b = .3
    ntotal = 500
    nsample = 100
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    # generate toy spiral data
    orig_trajs, samp_trajs, samp_trajs_next, orig_ts, samp_ts = generate_spiral2d(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        ntotal = ntotal,
        a=a, b=b
    )
    
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
#     zero_trajs = torch.zeros([samp_trajs.shape[0], samp_trajs.shape[1], 2]).to(samp_trajs)
#     samp_trajs =  torch.cat([samp_trajs, zero_trajs],dim=2)
    samp_trajs_next = torch.from_numpy(samp_trajs_next).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    next_ts_x2z = torch.from_numpy(np.array([0., 1.])).float().to(device)
    next_ts_x2z_flip = torch.from_numpy(np.array([1., 0.])).float().to(device)
    next_ts_t2t = torch.from_numpy(np.array([0., 1.])).float().to(device)
    
    batch_size = samp_trajs.shape[0]
    time_size = samp_trajs.shape[1]
    
    torch.save({
        "samp_trajs": samp_trajs,
        "orig_trajs": orig_trajs,
        "time_size": time_size,
        "samp_trajs_next": samp_trajs_next,
        "samp_ts": samp_ts,
        "batch_size": batch_size,
    }, os.path.join(args.save, "data_checkpt.pth"))
    
    # model
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    
    model_x2z = build_model_tabular(args, latent_dim*time_size, regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model_x2z)
    set_cnf_options(args, model_x2z)
    
    model_t2t = modules.LinearZeros(latent_dim*time_size, latent_dim*time_size).to(device).cuda()
    
#     model_t2t = build_model_tabular(args, latent_dim*time_size, regularization_fns).to(device)
#     if args.spectral_norm: add_spectral_norm(model_t2t)
#     set_cnf_options(args, model_t2t)

    # logger.info(model_x2z)
    # logger.info("Number of trainable parameters: {}".format(count_parameters(model_x2z)))
    
    # logger.info(model_t2t)
    # logger.info("Number of trainable parameters: {}".format(count_parameters(model_t2t)))

    # params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    params = (list(model_x2z.parameters()) + list(model_t2t.parameters()))
    optimizer = optim.Adam(params, lr=args.lr)
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            model_x2z.load_state_dict(checkpoint['x2z_state_dict'])
            model_t2t.load_state_dict(checkpoint['t2t_state_dict'])
            optimizer.load_state_dict(checkpoint['optim_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            model_x2z.train()
            model_t2t.train()
            optimizer.zero_grad()
            zero = torch.zeros(samp_trajs.shape[0], 1).to(samp_trajs)
            z_t, delta_logpzt = model_x2z(samp_trajs.view(batch_size, -1), zero)
            
            if itr < (args.niters // 3 + 1):
                # compute loss logpx
                logpzt = standard_normal_logprob(z_t).sum(1, keepdim=True)
                logpxt = logpzt - delta_logpzt
                logpxt_per_dim = torch.sum(logpxt) / samp_trajs.nelement()  # averaged over batches
                bitsxt_per_dim = -(logpxt_per_dim) / np.log(2)
                loss = torch.mean(bitsxt_per_dim)
            
            if itr >= (args.niters // 3 + 1) and itr < (2 * args.niters // 3 + 1):
                z_t_next = model_t2t(z_t.detach())
                z_t_next_samp, _ = model_x2z(samp_trajs_next.view(batch_size, -1), zero)
                lzdym = l2loss(z_t_next_samp.detach(), z_t_next).sum(-1).sum(-1) / z_t_next.nelement()
                loss = torch.mean(lzdym)
            
            if itr >= (2 * args.niters // 3 + 1):
                logpzt = standard_normal_logprob(z_t).sum(1, keepdim=True)
                logpxt = logpzt - delta_logpzt
                logpxt_per_dim = torch.sum(logpxt) / samp_trajs.nelement()  # averaged over batches
                bitsxt_per_dim = -(logpxt_per_dim) / np.log(2)
                
                z_t_next = model_t2t(z_t)
                z_t_next_samp, _ = model_x2z(samp_trajs_next.view(batch_size, -1), zero)
                lzdym = l2loss(z_t_next_samp.detach(), z_t_next).sum(-1).sum(-1) / z_t_next.nelement()
                
#                 x_t_next = model_x2z(z_t_next, reverse=True)
#                 x_t_next = x_t_next.view(batch_size, time_size, -1)
#                 x_t_next = x_t_next[:,:,:2]
#                 lxconst = l2loss(samp_trajs_next, x_t_next.detach).sum(-1).sum(-1)
                loss = torch.mean(bitsxt_per_dim + lzdym)
            
            # compute loss for temporal prediction
            # logpx = l2loss(samp_trajs_next, x_t_next).sum(-1).sum(-1)
            # noise_std_ = torch.zeros(x_t_next.size()).to(device) + noise_std
            # noise_logvar = 2. * torch.log(noise_std_).to(device)
            # logpx = log_normal_pdf(samp_trajs_next, x_t_next, noise_logvar).sum(-1).sum(-1)
            
            loss.backward()
#             if itr < (args.niters // 3 + 1):
#                 for param in model_x2z.parameters():
#                     import ipdb; ipdb.set_trace()
#                     param.grad.data.zero_()
                    
#             if itr >= (args.niters // 3 + 1) and itr < (2 * args.niters // 3 + 1):
#                 for param in model_x2z.parameters():
#                     param.grad.data.zero_()
                
            optimizer.step()
            loss_meter.update(loss.item())

            print('Iter: {}, loss: {:.4f}'.format(itr, loss_meter.avg))
            
            writer.add_scalars('loss', {'train_iter': loss.cpu()}, itr)
#             writer.add_scalars('logpx', {'train_iter': torch.mean(-logpx).cpu()}, itr)
#             writer.add_scalars('logpxt', {'train_iter': torch.mean(-logpxt).cpu()}, itr)
            
            if (itr % 50) == 0 & (itr >= (args.niters // 3 + 1)):
                torch.save({
                    "args": args,
                    "x2z_state_dict": model_x2z.state_dict(),
                    "t2t_state_dict": model_t2t.state_dict(),
                    "optim_state_dict": optimizer.state_dict(),
                    "iter": itr,
                    "loss": loss,
                }, os.path.join(args.save, "iter_%i_checkpt.pth"%itr))
                
                if args.visualize:
                    nvis = 500
                    model_x2z.eval()
                    model_t2t.eval()
                    with torch.no_grad():
                        samp_trajs_all = orig_trajs[0:1,0:time_size,:]
                        samp_trajs = samp_trajs_all[:,-time_size:samp_trajs_all.shape[1],:]
                        for i in range(nvis-100):
                            logger.info("%i/%i"%(i, nvis-100))
                            zero = torch.zeros(samp_trajs.shape[0], 1).to(samp_trajs)
#                             zero_trajs = torch.zeros([samp_trajs.shape[0], samp_trajs.shape[1], 2]).to(samp_trajs)
#                             samp_trajs =  torch.cat([samp_trajs, zero_trajs],dim=2)
                            z_t, delta_logpzt = model_x2z(samp_trajs.view(1, -1), zero)
                            z_t_next = model_t2t(z_t)
                            x_t_next = model_x2z(z_t_next, reverse=True)
                            x_t_next = x_t_next.view(1, time_size, -1)
                            x_t_next = x_t_next[:,:,:2]
                            samp_trajs_all = torch.cat([samp_trajs_all,x_t_next[:, -1:time_size, :]], dim=1)
                            samp_trajs = samp_trajs_all[:,-time_size:samp_trajs_all.shape[1],:]

                            if i % 50 == 0:
                                torch.save({
                                    "samp_trajs_all": samp_trajs_all,
                                    "orig_trajs": orig_trajs,
                                    "time_size": time_size,
                                    "iter": i,
                                    "nvis": nvis,
                                    "samp_trajs": samp_trajs,
                                }, os.path.join(args.save, "vis_current_checkpt_itertrain_%i.pth"%itr))

                                xs_pos_tem = samp_trajs_all[0]
                                xs_pos_tem = xs_pos_tem.cpu().numpy()
                                orig_traj_tem = orig_trajs[0].cpu().numpy()

                                if xs_pos_tem.shape[0] <= orig_traj_tem.shape[0]:
                                    logger.info("Error {:.4f} at time {:04d}".format(np.mean((xs_pos_tem[100:] - orig_traj_tem[100:xs_pos_tem.shape[0]])**2.),i))

                                plt.figure()
                                plt.plot(orig_traj_tem[:, 0], orig_traj_tem[:, 1],'g', label='true trajectory')
                                plt.plot(xs_pos_tem[:, 0], xs_pos_tem[:, 1], 'r', label='learned trajectory (t>0)')
                                plt.legend()
                                plt.savefig(os.path.join(args.save,'vis_current_itertrain_%i.png'%itr), dpi=200)

                        ts_pos = np.linspace(0., 2. * np.pi, num=nvis)
                        ts_pos = torch.from_numpy(ts_pos).float().to(device)

                    xs_pos = samp_trajs_all[0]
                    xs_pos = xs_pos.cpu().numpy()
                    orig_traj = orig_trajs[0].cpu().numpy()
                    # samp_traj = samp_trajs[0].cpu().numpy()

                    logger.info("Error {:.4f} at time {:04d}".format(np.mean((xs_pos[100:orig_traj.shape[0]] - orig_traj[100:])**2.),i))

                    torch.save({
                        "samp_trajs_all": samp_trajs_all,
                        "orig_trajs": orig_trajs,
                        "time_size": time_size,
                        "iter": i,
                        "nvis": nvis,
                        "samp_trajs": samp_trajs,
                    }, os.path.join(args.save, "vis_final_checkpt_itertrain_%i.pth"%itr))

                    plt.figure()
                    plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                             'g', label='true trajectory')
                    plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                             label='learned trajectory (t>0)')
            #         plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
            #                  label='learned trajectory (t<0)')
            #         plt.scatter(samp_traj[:, 0], samp_traj[
            #                     :, 1], label='sampled data', s=3)
                    plt.legend()
                    plt.savefig(os.path.join(args.save,'vis_itertrain_%i.png'%itr), dpi=500)
                
            
    except:
        print('Error')
        import ipdb; ipdb.set_trace()
    
    torch.save({
        "args": args,
        "x2z_state_dict": model_x2z.state_dict(),
        "t2t_state_dict": model_t2t.state_dict(),
        "optim_state_dict": optimizer.state_dict(),
        "iter": args.niters + 1,
        "loss": loss,}, os.path.join(args.save, "iter_final_checkpt.pth"))
    
    print('Training complete after {} iters.'.format(itr))
    
    if args.visualize:
        nvis = 2000
        model_x2z.eval()
        model_t2t.eval()
        with torch.no_grad():
            samp_trajs_all = orig_trajs[0:1,0:time_size,:]
            samp_trajs = samp_trajs_all[:,-time_size:samp_trajs_all.shape[1],:]
            for i in range(nvis-100):
                logger.info("%i/%i"%(i, nvis-100))
                zero = torch.zeros(samp_trajs.shape[0], 1).to(samp_trajs)
#                 zero_trajs = torch.zeros([samp_trajs.shape[0], samp_trajs.shape[1], 2]).to(samp_trajs)
#                 samp_trajs =  torch.cat([samp_trajs, zero_trajs],dim=2)
                z_t, delta_logpzt = model_x2z(samp_trajs.view(1, -1), zero)
                z_t_next = model_t2t(z_t)
                x_t_next = model_x2z(z_t_next, reverse=True)
                x_t_next = x_t_next.view(1, time_size, -1)
                x_t_next = x_t_next[:,:,:2]
                samp_trajs_all = torch.cat([samp_trajs_all,x_t_next[:, -1:time_size, :]], dim=1)
                samp_trajs = samp_trajs_all[:,-time_size:samp_trajs_all.shape[1],:]
                
                if i % 50 == 0:
                    torch.save({
                        "samp_trajs_all": samp_trajs_all,
                        "orig_trajs": orig_trajs,
                        "time_size": time_size,
                        "iter": i,
                        "nvis": nvis,
                        "samp_trajs": samp_trajs,
                    }, os.path.join(args.save, "vis_current_checkpt.pth"))
                    
                    xs_pos_tem = samp_trajs_all[0]
                    xs_pos_tem = xs_pos_tem.cpu().numpy()
                    orig_traj_tem = orig_trajs[0].cpu().numpy()
                    
                    if xs_pos_tem.shape[0] <= orig_traj_tem.shape[0]:
                        logger.info("Error {:.4f} at time {:04d}".format(np.mean((xs_pos_tem[100:] - orig_traj_tem[100:xs_pos_tem.shape[0]])**2.),i))
                    
                    plt.figure()
                    plt.plot(orig_traj_tem[:, 0], orig_traj_tem[:, 1],'g', label='true trajectory')
                    plt.plot(xs_pos_tem[:, 0], xs_pos_tem[:, 1], 'r', label='learned trajectory (t>0)')
                    plt.legend()
                    plt.savefig(os.path.join(args.save,'vis_current.png'), dpi=200)
            
            ts_pos = np.linspace(0., 2. * np.pi, num=nvis)
            ts_pos = torch.from_numpy(ts_pos).float().to(device)

        xs_pos = samp_trajs_all[0]
        xs_pos = xs_pos.cpu().numpy()
        orig_traj = orig_trajs[0].cpu().numpy()
        # samp_traj = samp_trajs[0].cpu().numpy()
        
        try:
            logger.info("Error {:.4f} at time {:04d}".format(np.mean((xs_pos[100:orig_traj.shape[0]] - orig_traj[100:])**2.),i))
        except:
            import ipdb; ipdb.set_trace()
        
        torch.save({
            "samp_trajs_all": samp_trajs_all,
            "orig_trajs": orig_trajs,
            "time_size": time_size,
            "iter": i,
            "nvis": nvis,
            "samp_trajs": samp_trajs,
        }, os.path.join(args.save, "vis_final_checkpt.pth"))

        plt.figure()
        plt.plot(orig_traj[:, 0], orig_traj[:, 1],
                 'g', label='true trajectory')
        plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r',
                 label='learned trajectory (t>0)')
#         plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c',
#                  label='learned trajectory (t<0)')
#         plt.scatter(samp_traj[:, 0], samp_traj[
#                     :, 1], label='sampled data', s=3)
        plt.legend()
        plt.savefig(os.path.join(args.save,'vis.png'), dpi=500)
