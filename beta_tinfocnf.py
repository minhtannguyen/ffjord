import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=False)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--nspiral', type=int, default=5000)
parser.add_argument('--nsample', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--sup', type=eval, default=False)
parser.add_argument('--softcond', type=eval, default=False)
parser.add_argument('--cond', type=eval, default=False)
parser.add_argument("--savedir", type=str, default="./results")
parser.add_argument("--save", type=str, default="vis")
parser.add_argument("--seed", type=int, default=0)

parser.add_argument("--a_range", type=str, default="0., 1.")
parser.add_argument("--b_range", type=str, default="0., 5.")
parser.add_argument('--noise_std', type=float, default=.3)
parser.add_argument("--a_range_test", type=str, default="0., 1.")
parser.add_argument("--b_range_test", type=str, default="0., 5.")
parser.add_argument('--noise_std_test', type=float, default=.3)
parser.add_argument('--w_sup', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument("--latent_dim", type=int, default=5)

args = parser.parse_args()

# set seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.savedir):
    os.makedirs(args.savedir)
    
savefile = os.path.join(args.savedir, args.save)

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint
    
class LinearZeros(nn.Linear):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)
        # init
        self.weight.data.zero_()
        self.bias.data.zero_()

    def forward(self, input):
        output = super().forward(input)
        return output
    
def generate_spiral2d_randtheta(nspiral=1000,
                      ntotal=500,
                      nsample=100,
                      start=0.,
                      stop=1,  # approximately equal to 6pi
                      noise_std=.1,
                      a_range=(0.,1.),
                      b_range=(0.,5.),
                      savefig=True, suffix='train'):
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
    a = np.random.normal(loc=a_range[0], scale=a_range[1], size=nspiral) # shape (nspiral, )
    b = np.random.normal(loc=b_range[0], scale=b_range[1], size=nspiral) # shape (nspiral, )
    theta = np.concatenate((a[:,None],b[:,None]),1) # shape (nspiral, 2)
    
    ### I STOPPED HERE ####
    
    # add 1 all timestamps to avoid division by 0
    orig_ts = np.linspace(start, stop, num=ntotal) # [ntotal,]
    samp_ts = orig_ts[:nsample] # [nsample,]

    # generate clock-wise and counter clock-wise spirals in observation space
    # with two sets of time-invariant latent dynamics
    zs_cw = stop + 1. - orig_ts # [ntotal,]
    zs_cw = np.repeat(zs_cw[:,None], nspiral, axis=1) # [ntotal, nspiral]
    rs_cw = np.repeat(a[None,:], ntotal, axis=0) + np.matmul(50. / zs_cw, np.diag(b)) # [ntotal, nspiral]
    xs, ys = rs_cw * np.cos(zs_cw) - 5., rs_cw * np.sin(zs_cw) # [ntotal, nspiral]
    orig_traj_cw = np.stack((xs, ys), axis=1) # [ntotal, 2, nspiral]

    zs_cc = orig_ts # [ntotal,]
    zs_cc = np.repeat(zs_cc[:,None], nspiral, axis=1) # [ntotal, nspiral]
    rw_cc = np.repeat(a[None,:], ntotal, axis=0) + np.matmul(zs_cc, np.diag(b)) # [ntotal, nspiral]
    xs, ys = rw_cc * np.cos(zs_cc) + 5., rw_cc * np.sin(zs_cc) # [ntotal, nspiral]
    orig_traj_cc = np.stack((xs, ys), axis=1) # [ntotal, 2, nspiral]

    if savefig:
        for i in range(3):
            plt.figure()
            plt.plot(orig_traj_cw[:, 0, i], orig_traj_cw[:, 1, i], label='clock')
            plt.plot(orig_traj_cc[:, 0, i], orig_traj_cc[:, 1, i], label='counter clock')
            plt.legend()
            plt.savefig(savefile + '_ground_truth_%i_%s.png'%(i, suffix), dpi=500)
            print('Saved ground truth spiral at {}'.format(savefile + '_ground_truth_%i_%s.png'%(i, suffix)))

    # sample starting timestamps
    orig_trajs = []
    samp_trajs = []
    dir_vec = []
    for i in range(nspiral):
        # don't sample t0 very near the start or the end
        t0_idx = npr.multinomial(
            1, [1. / (ntotal - 2. * nsample)] * (ntotal - int(2 * nsample)))
        t0_idx = np.argmax(t0_idx) + nsample

        cc = bool(npr.rand() > .5)  # uniformly select rotation
        dir_vec.append(float(cc))
        orig_traj = orig_traj_cc[:,:,i] if cc else orig_traj_cw[:,:,i]
        orig_trajs.append(orig_traj)

        samp_traj = orig_traj[t0_idx:t0_idx + nsample, :].copy()
        samp_traj += npr.randn(*samp_traj.shape) * noise_std
        samp_trajs.append(samp_traj)

    # batching for sample trajectories is good for RNN; batching for original
    # trajectories only for ease of indexing
    orig_trajs = np.stack(orig_trajs, axis=0)
    samp_trajs = np.stack(samp_trajs, axis=0)
    
    dir_vec = np.array(dir_vec)
    
    theta = np.concatenate((a[:,None],b[:,None], dir_vec[:,None]),1) # shape (nspiral, 3)

    return orig_trajs, samp_trajs, orig_ts, samp_ts, theta


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


class RecognitionRNN(nn.Module):

    def __init__(self, output_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, output_dim)

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


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


if __name__ == '__main__':
    latent_dim = args.latent_dim
    nhidden = 20
    rnn_nhidden = 25
    output_dim = (latent_dim - 1) * 2 + 1 if (args.softcond or args.sup or args.cond) else latent_dim * 2  # 9
    obs_dim = 2
    nspiral = args.nspiral
    start = 0.
    stop = 6 * np.pi
    noise_std = args.noise_std
    a_range = tuple(map(float, args.a_range.split(",")))
    b_range = tuple(map(float, args.b_range.split(",")))
    noise_std_test = args.noise_std_test
    a_range_test = tuple(map(float, args.a_range_test.split(",")))
    b_range_test = tuple(map(float, args.b_range_test.split(",")))
    pi_dir = 0.5
    ntotal = 500
    nsample = args.nsample
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')
    
    n_sys_params = 3
    cond_size = 5
    num_evals = 100

    # generate train toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts, theta = generate_spiral2d_randtheta(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std,
        a_range=a_range, b_range=b_range, suffix='train'
    )
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)
    theta = torch.tensor(theta).to(device)
    
    # generate test toy spiral data
    orig_trajs_test, samp_trajs_test, orig_ts_test, samp_ts_test, theta_test = generate_spiral2d_randtheta(
        nspiral=nspiral,
        start=start,
        stop=stop,
        noise_std=noise_std_test,
        a_range=a_range_test, b_range=b_range_test, suffix='test'
    )
    orig_trajs_test = torch.from_numpy(orig_trajs_test).float().to(device)
    samp_trajs_test = torch.from_numpy(samp_trajs_test).float().to(device)
    samp_ts_test = torch.from_numpy(samp_ts_test).float().to(device)
    theta_test = torch.tensor(theta_test).to(device)

    # model
    func = LatentODEfunc(latent_dim, nhidden).to(device)
    rec = RecognitionRNN(output_dim , obs_dim, rnn_nhidden, nspiral).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
        
    if args.cond:
        conditioner = LinearZeros(n_sys_params, cond_size).to(device)
        params = params + list(conditioner.parameters())
        
    if args.softcond:
        soft_conditioner = LinearZeros(cond_size, cond_size).to(device) # map [:,5] to [:,5]
        params = params + list(soft_conditioner.parameters())
    
    optimizer = optim.Adam(params, lr=args.lr)
    
    # loss meters
    loss_meter = RunningAverageMeter()
    loss_elbo_meter = RunningAverageMeter()
    if args.sup: 
        loss_sysid_meter = RunningAverageMeter()
        dir_meter = RunningAverageMeter()
        params_meter = RunningAverageMeter()
        
    # loss
    l2loss = nn.MSELoss()
    BCEloss = torch.nn.BCEWithLogitsLoss(reduction='none')

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            rec.load_state_dict(checkpoint['rec_state_dict'])
            dec.load_state_dict(checkpoint['dec_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            orig_trajs = checkpoint['orig_trajs']
            samp_trajs = checkpoint['samp_trajs']
            orig_ts = checkpoint['orig_ts']
            samp_ts = checkpoint['samp_ts']
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            # backward in time to infer q(z_0)
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
                
            if args.softcond or args.sup or args.cond:
                qz0_mean, qz0_logvar = out[:, :(latent_dim-3)], out[:, (latent_dim-3):-5]
                qz0_a_mean, qz0_a_logvar = out[:,-5:-4], out[:,-4:-3]
                qz0_b_mean, qz0_b_logvar = out[:,-3:-2], out[:,-2:-1]
                qz0_dir = out[:, -1:]
                
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                epsilon_a = torch.randn(qz0_a_mean.size()).to(device)
                z0_a = epsilon_a * torch.exp(.5 * qz0_a_logvar) + qz0_a_mean

                epsilon_b = torch.randn(qz0_b_mean.size()).to(device)
                z0_b = epsilon_b * torch.exp(.5 * qz0_b_logvar) + qz0_b_mean
                
                z0_dir = torch.sigmoid(qz0_dir)
                
                pred_dir = (z0_dir > .5).to(z0_dir)
                
                z0 = torch.cat((z0, z0_a, z0_b, z0_dir), dim=1) # 5
                
            else:
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
                
            if args.sup:
                loss_sysid = log_normal_pdf(theta[:,:1].to(qz0_a_mean), qz0_a_mean, qz0_a_logvar).sum(-1).sum(-1) + log_normal_pdf(theta[:,1:2].to(qz0_b_mean), qz0_b_mean, qz0_b_logvar).sum(-1).sum(-1) + torch.mean(BCEloss(qz0_dir, theta[:, 2:].to(qz0_dir)), dim=0).sum(-1)

            # forward in time and solve ode for reconstructions
            pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
            pred_x = dec(pred_z)

            # compute loss
            noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
            noise_logvar = 2. * torch.log(noise_std_).to(device)
            logpx = log_normal_pdf(
                samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
            
            if args.cond:
                pz0 = conditioner(theta.to(out))
            elif args.softcond:
                prior_vec = torch.tensor([a_range[0], a_range[1], b_range[0], b_range[1], pi_dir]).repeat(z0.size()[0],1).to(device)
                pz0 = soft_conditioner(prior_vec)
                
            if args.softcond or args.cond:
                pz0_mean = pz0_logvar = torch.zeros([z0.size()[0],  z0.size()[1] - 3]).to(device)  
                pz0_a_mean = pz0[:,:1, ...]
                pz0_a_logvar = pz0[:,1:2, ...]
                pz0_b_mean = pz0[:,2:3, ...]
                pz0_b_logvar = pz0[:,3:4, ...]
                pz0_dir = torch.sigmoid(pz0[:,4:, ...])
            elif args.sup:
                pz0_mean = pz0_logvar = torch.zeros([z0.size()[0],  z0.size()[1] - 3]).to(device)
            else:
                pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
            
            if args.softcond or args.cond:
                analytic_kl = normal_kl(qz0_mean, qz0_logvar, pz0_mean, pz0_logvar).sum(-1) + normal_kl(qz0_a_mean, qz0_a_logvar, pz0_a_mean, pz0_a_logvar).sum(-1) + normal_kl(qz0_b_mean, qz0_b_logvar, pz0_b_mean, pz0_b_logvar).sum(-1) + BCEloss(qz0_dir, pz0_dir).sum(-1)
            else:
                analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                        pz0_mean, pz0_logvar).sum(-1)
                
            loss_elbo = torch.mean(-logpx + args.beta * analytic_kl, dim=0)
            
            if args.sup: 
                loss = loss_elbo + args.w_sup * loss_sysid
            else:
                loss = loss_elbo
                
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item())
            loss_elbo_meter.update(loss_elbo.item())
            if args.sup:
                loss_sysid_meter.update(loss_sysid.item())
                params_error = l2loss(qz0_a_mean, theta[:,:1].to(qz0_a_mean)) + l2loss(qz0_b_mean, theta[:,1:2].to(qz0_b_mean))
                params_meter.update(params_error.item())
                dir_error = torch.mean((pred_dir != theta[:, 2:].to(pred_dir)).to(pred_dir), dim=0).sum(-1)
                dir_meter.update(dir_error.item())
                
            if args.sup:
                print('Iter: {}, running avg loss: {:.4f}, running avg elbo: {:.4f}, running avg sysid: {:.4f}, running avg error params: {:.4f}, running avg error dir: {:.4f}, sysid: ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}), sysid_truth: ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f})'.format(itr, loss_meter.avg, -loss_elbo_meter.avg, loss_sysid_meter.avg, params_meter.avg, dir_meter.avg, qz0_a_mean[0,0].cpu().detach().numpy(), qz0_b_mean[0,0].cpu().detach().numpy(), pred_dir[0,0].cpu().detach().numpy(), qz0_a_mean[10,0].cpu().detach().numpy(), qz0_b_mean[10,0].cpu().detach().numpy(), pred_dir[10,0].cpu().detach().numpy(), qz0_a_mean[20,0].cpu().detach().numpy(), qz0_b_mean[20,0].cpu().detach().numpy(), pred_dir[20,0].cpu().detach().numpy(), theta[0,0].cpu().numpy(), theta[0,1].cpu().numpy(), theta[0,2].cpu().numpy(), theta[10,0].cpu().numpy(), theta[10,1].cpu().numpy(), theta[10,2].cpu().numpy(), theta[20,0].cpu().numpy(), theta[20,1].cpu().numpy(), theta[20,2].cpu().numpy()))
            else:
                print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'rec_state_dict': rec.state_dict(),
                'dec_state_dict': dec.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'orig_trajs': orig_trajs,
                'samp_trajs': samp_trajs,
                'orig_ts': orig_ts,
                'samp_ts': samp_ts,
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
            
    ckpt_path = os.path.join(args.savedir, 'ckpt.pth')
    if args.cond:
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'conditioner_state_dict': conditioner.state_dict(),
            'orig_trajs': orig_trajs,
            'samp_trajs': samp_trajs,
            'orig_ts': orig_ts,
            'samp_ts': samp_ts,
        }, ckpt_path)
    elif args.softcond:
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'soft_conditioner_state_dict': soft_conditioner.state_dict(),
            'orig_trajs': orig_trajs,
            'samp_trajs': samp_trajs,
            'orig_ts': orig_ts,
            'samp_ts': samp_ts,
        }, ckpt_path)
    else:
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'orig_trajs': orig_trajs,
            'samp_trajs': samp_trajs,
            'orig_ts': orig_ts,
            'samp_ts': samp_ts,
        }, ckpt_path)
    print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))
    
    
    def eval_performance(samp_trajs, orig_ts, orig_trajs, theta, num_evals, suffix):
        with torch.no_grad():
            # sample from trajectorys' approx. posterior
            h = rec.initHidden().to(device)
            for t in reversed(range(samp_trajs.size(1))):
                obs = samp_trajs[:, t, :]
                out, h = rec.forward(obs, h)
                
            if args.softcond or args.sup or args.cond:
                qz0_mean, qz0_logvar = out[:, :(latent_dim-3)], out[:, (latent_dim-3):-5]
                qz0_a_mean, qz0_a_logvar = out[:,-5:-4], out[:,-4:-3]
                qz0_b_mean, qz0_b_logvar = out[:,-3:-2], out[:,-2:-1]
                qz0_dir = out[:, -1:]
                
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                epsilon_a = torch.randn(qz0_a_mean.size()).to(device)
                z0_a = epsilon_a * torch.exp(.5 * qz0_a_logvar) + qz0_a_mean

                epsilon_b = torch.randn(qz0_b_mean.size()).to(device)
                z0_b = epsilon_b * torch.exp(.5 * qz0_b_logvar) + qz0_b_mean
                
                z0_dir = torch.sigmoid(qz0_dir)
                
                pred_dir = (z0_dir > .5).to(z0_dir)
                
                z0 = torch.cat((z0, z0_a, z0_b, z0_dir), dim=1)
            
            else:
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
            
            if args.sup:
                loss_sysid = log_normal_pdf(theta[:,:1].to(qz0_a_mean), qz0_a_mean, qz0_a_logvar).sum(-1).sum(-1) + log_normal_pdf(theta[:,1:2].to(qz0_b_mean), qz0_b_mean, qz0_b_logvar).sum(-1).sum(-1) + torch.mean(BCEloss(qz0_dir, theta[:, 2:].to(qz0_dir)), dim=0).sum(-1)
                params_error = l2loss(qz0_a_mean, theta[:,:1].to(qz0_a_mean)) + l2loss(qz0_b_mean, theta[:,1:2].to(qz0_b_mean))
                dir_error = torch.mean((pred_dir != theta[:, 2:].to(pred_dir)).to(pred_dir), dim=0).sum(-1)
                
                print('%s:'%suffix)
                print('loss sysid: {:.4f}, error params: {:.4f}, error dir: {:.4f}, sysid: ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}), sysid_truth: ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f}) ({:.4f}, {:.4f}, {:.4f})'.format(loss_sysid, params_error, dir_error, qz0_a_mean[0,0].cpu().detach().numpy(), qz0_b_mean[0,0].cpu().detach().numpy(), pred_dir[0,0].cpu().detach().numpy(), qz0_a_mean[10,0].cpu().detach().numpy(), qz0_b_mean[10,0].cpu().detach().numpy(), pred_dir[10,0].cpu().detach().numpy(), qz0_a_mean[20,0].cpu().detach().numpy(), qz0_b_mean[20,0].cpu().detach().numpy(), pred_dir[20,0].cpu().detach().numpy(), theta[0,0].cpu().numpy(), theta[0,1].cpu().numpy(), theta[0,2].cpu().numpy(), theta[10,0].cpu().numpy(), theta[10,1].cpu().numpy(), theta[10,2].cpu().numpy(), theta[20,0].cpu().numpy(), theta[20,1].cpu().numpy(), theta[20,2].cpu().numpy()))
                
            orig_ts = torch.from_numpy(orig_ts).float().to(device)

            # take first trajectory for visualization
            for i in range(num_evals):
                ts_pos = np.linspace(0., 2. * np.pi, num=2000)
                ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
                ts_pos = torch.from_numpy(ts_pos).float().to(device)
                ts_neg = torch.from_numpy(ts_neg).float().to(device)

                zs_pos = odeint(func, z0[i], ts_pos)
                zs_neg = odeint(func, z0[i], ts_neg)

                xs_pos = dec(zs_pos)
                xs_neg = torch.flip(dec(zs_neg), dims=[0])

                xs_pos = xs_pos.cpu().numpy()
                xs_neg = xs_neg.cpu().numpy()
                orig_traj = orig_trajs[i].cpu().numpy()
                samp_traj = samp_trajs[i].cpu().numpy()

                plt.figure()
                plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory')
                plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r', label='learned trajectory (t>0)')
                plt.plot(xs_neg[:, 0], xs_neg[:, 1], 'c', label='learned trajectory (t<0)')
                plt.scatter(samp_traj[:, 0], samp_traj[:, 1], label='sampled data', s=3)
                plt.legend()
                plt.axis('equal')
                plt.savefig(savefile + '_%s_%i.png'%(suffix,i), dpi=500)
                plt.close()
                
                plt.figure()
                plt.plot(orig_traj[:, 0], orig_traj[:, 1], 'g', label='true trajectory')
                plt.plot(xs_pos[:, 0], xs_pos[:, 1], 'r', label='learned trajectory (t>0)')
                plt.scatter(samp_traj[:, 0], samp_traj[:, 1], label='sampled data', s=3)
                plt.legend()
                plt.axis('equal')
                plt.savefig(savefile + '_pos_%s_%i.png'%(suffix,i), dpi=500)
                
                print('Saved visualization figure at {}'.format(savefile + '_%s_%i.png'%(suffix,i)))

    if args.visualize:
        # eval performance on train set
        eval_performance(samp_trajs, orig_ts, orig_trajs, theta, num_evals, 'train')
        # eval performance on test set
        eval_performance(samp_trajs_test, orig_ts_test, orig_trajs_test, theta_test, num_evals, 'test')
        