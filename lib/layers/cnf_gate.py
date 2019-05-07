import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

import math
from torch.autograd import Variable
import torch.autograd as autograd

from .wrappers.cnf_regularization import RegularizedODEfunc
from .gate import FeedforwardGateI, FeedforwardGateII

__all__ = ["CNF_Gate"]

GATES = {
    'cnn1': FeedforwardGateI,
    'cnn2': FeedforwardGateII,
}

class CNF_Gate(nn.Module):
    def __init__(self, odefunc, size, T=1.0, train_T=False, regularization_fns=None, solver='dopri5', atol=1e-5, rtol=1e-5, scale=1.0, gate='cnn1'):
        super(CNF_Gate, self).__init__()
        if train_T:
            self.register_parameter("sqrt_end_time", nn.Parameter(torch.sqrt(torch.tensor(T))))
        else:
            self.register_buffer("sqrt_end_time", torch.sqrt(torch.tensor(T)))

        nreg = 0
        if regularization_fns is not None:
            odefunc = RegularizedODEfunc(odefunc, regularization_fns)
            nreg = len(regularization_fns)
        
        self.gate = gate
        self.gate_net = GATES[gate](in_channel=size, out_channel=10, pool_size=5)
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.test_solver = solver
        self.test_atol = atol
        self.test_rtol = rtol
        self.solver_options = {}
        self.scale = scale

    def forward(self, z, logpz=None, integration_times=None, reverse=False):

        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = torch.tensor([0.0, self.sqrt_end_time * self.sqrt_end_time]).to(z)
        if reverse:
            integration_times = _flip(integration_times, 0)
            
        # compute atol and rtol
        tol_val = self.gate_net(z)
        tol_val = self.scale * torch.mean(tol_val, dim=0)
        atol_val = tol_val[0]
        rtol_val = tol_val[1]
        test_atol_val = atol_val
        test_rtol_val = rtol_val

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.tensor(0).to(z) for _ in range(self.nreg))

        if self.training:
            state_t = odeint(
                self.odefunc,
                (z, _logpz) + reg_states,
                integration_times.to(z),
                atol=[atol_val, atol_val] + [1e20] * len(reg_states) if self.solver == 'dopri5' else atol_val,
                rtol=[rtol_val, rtol_val] + [1e20] * len(reg_states) if self.solver == 'dopri5' else rtol_val,
                method=self.solver,
                options=self.solver_options,
            )
        else:
            state_t = odeint(
                self.odefunc,
                (z, _logpz),
                integration_times.to(z),
                atol=test_atol_val,
                rtol=test_rtol_val,
                method=self.test_solver,
            )
        
        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)
        
        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t, atol_val, rtol_val
        else:
            return z_t, atol_val, rtol_val

    def get_regularization_states(self):
        reg_states = self.regularization_states
        self.regularization_states = None
        return reg_states

    def num_evals(self):
        return self.odefunc._num_evals.item()


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]