import torch.nn as nn


class SequentialFlow_Gate(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow_Gate, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        atol_list = []; rtol_list = []
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                if hasattr(self.chain[i], 'atol'):
                    x, atol, rtol = self.chain[i](x, reverse=reverse)
                    atol_list.append(atol); rtol_list.append(rtol)
                else:
                    x = self.chain[i](x, reverse=reverse)
                    
            return x, atol_list, rtol_list
        else:
            for i in inds:
                if hasattr(self.chain[i], 'atol'):
                    x, logpx, atol, rtol = self.chain[i](x, logpx, reverse=reverse)
                    atol_list.append(atol); rtol_list.append(rtol)
                else:
                    x, logpx = self.chain[i](x, logpx, reverse=reverse)
                
            return x, logpx, atol_list, rtol_list
