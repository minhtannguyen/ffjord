import torch
import torch.nn as nn

from integrate import odeint_adjoint as odeint

from .wrappers.cnf_regularization import RegularizedODEfunc

__all__ = ["CNF"]


class CNF(nn.Module):
    def __init__(self, odefunc, T=None, regularization_fns=None, solver='dopri5'):
        super(CNF, self).__init__()
        if T is None:
            self.end_time_param = nn.Parameter(torch.tensor(1.0))
            self.end_time = self.end_time_param**2
        else:
            self.end_time = T
        self.integration_times = torch.tensor([0.0, self.end_time])

        nreg = 0
        if regularization_fns is not None:
            for reg_fn in regularization_fns:
                odefunc = RegularizedODEfunc(odefunc, reg_fn)
                nreg += 1
        self.odefunc = odefunc
        self.nreg = nreg
        self.regularization_states = None
        self.solver = solver

    def forward(self, z, logpz=None, integration_times=None, reverse=False, atol=1e-6, rtol=1e-5):
        if logpz is None:
            _logpz = torch.zeros(z.shape[0], 1).to(z)
        else:
            _logpz = logpz

        if integration_times is None:
            integration_times = self.integration_times
        if reverse:
            integration_times = _flip(integration_times, 0)

        # Refresh the odefunc statistics.
        self.odefunc.before_odeint()

        # Add regularization states.
        reg_states = tuple(torch.zeros(1).to(z) for _ in range(self.nreg))

        state_t = odeint(
            self.odefunc, (z, _logpz) + reg_states, integration_times.to(z), atol=atol, rtol=rtol, method=self.solver
        )

        if len(integration_times) == 2:
            state_t = tuple(s[1] for s in state_t)

        z_t, logpz_t = state_t[:2]
        self.regularization_states = state_t[2:]

        if logpz is not None:
            return z_t, logpz_t
        else:
            return z_t

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
