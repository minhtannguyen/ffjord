import torch
import torch.nn as nn
import lib.layers as layers
from lib.layers.odefunc import ODEnet
import numpy as np
from . import modules
from . import thops


class ODENVP(nn.Module):
    """
    Real NVP for image data. Will downsample the input until one of the
    dimensions is less than or equal to 4.

    Args:
        input_size (tuple): 4D tuple of the input size.
        n_scale (int): Number of scales for the representation z.
        n_resblocks (int): Length of the resnet for each coupling layer.
    """

    def __init__(
        self,
        input_size,
        n_scale=float('inf'),
        n_blocks=2,
        intermediate_dims=(32,),
        nonlinearity="softplus",
        squash_input=True,
        alpha=0.05,
        cnf_kwargs=None,
        y_class=10,
        condition_ratio=1.0,
        dropout_rate=0.0,
    ):
        super(ODENVP, self).__init__()
        self.n_scale = min(n_scale, self._calc_n_scale(input_size))
        self.n_blocks = n_blocks
        self.intermediate_dims = intermediate_dims
        self.nonlinearity = nonlinearity
        self.squash_input = squash_input
        self.alpha = alpha
        self.cnf_kwargs = cnf_kwargs if cnf_kwargs else {}
        self.y_class = y_class

        if not self.n_scale > 0:
            raise ValueError('Could not compute number of scales for input of' 'size (%d,%d,%d,%d)' % input_size)

        self.transforms = self._build_net(input_size)

        self.dims = [o[1:] for o in self.calc_output_size(input_size)]
        
        # for conditional
        C = int(np.prod(input_size[1:]) * condition_ratio)
        self.project_ycond = modules.LinearZeros(self.y_class, 2 * C)
        self.project_class = modules.LinearZeros(C, self.y_class)
        self.register_parameter(
            "prior_h",
            nn.Parameter(torch.zeros([1, 2 * C])))
        
        # dropout
        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)
        

    def _build_net(self, input_size):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedCNFLayers(
                    initial_size=(c, h, w),
                    idims=self.intermediate_dims,
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=(layers.LogitTransform(self.alpha) if self.alpha > 0 else layers.ZeroMeanTransform()) if self.squash_input and i == 0 else None,
                    n_blocks=self.n_blocks,
                    cnf_kwargs=self.cnf_kwargs,
                    nonlinearity=self.nonlinearity,
                )
            )
            c, h, w = c * 2, h // 2, w // 2
        
        return nn.ModuleList(transforms)

    def get_regularization(self):
        if len(self.regularization_fns) == 0:
            return None

        acc_reg_states = tuple([0.] * len(self.regularization_fns))
        for module in self.modules():
            if isinstance(module, layers.CNF_Gate_Sep):
                acc_reg_states = tuple(
                    acc + reg for acc, reg in zip(acc_reg_states, module.get_regularization_states())
                )
        return sum(state * coeff for state, coeff in zip(acc_reg_states, self.regularization_coeffs))

    def _calc_n_scale(self, input_size):
        _, _, h, w = input_size
        n_scale = 0
        while h >= 4 and w >= 4:
            n_scale += 1
            h = h // 2
            w = w // 2
        return n_scale

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._generate(x, logpx)
        else:
            return self._logdensity(x, logpx)
        
    def _prior(self, y_onehot=None):
        # compute the mean and std of the gaussian used to compute logpz
        h = self.prior_h.detach().clone()
        
        assert torch.sum(h) == 0.0
        assert y_onehot is not None
        
        yp = self.project_ycond(y_onehot)
        h = yp + h
        
        return thops.split_feature(h, "split")
        
        
    def _logdensity(self, x, logpx=None):
        _logpx = torch.zeros(x.shape[0], 1).to(x) if logpx is None else logpx
        out = []
        atol_list = []; rtol_list = []; logpa_list = []; nfe_list = []
        for idx in range(len(self.transforms)):
            x, _logpx, atol, rtol, logp_actions, nfe = self.transforms[idx].forward(x, _logpx)
            if idx < len(self.transforms) - 1:
                d = x.size(1) // 2
                x, factor_out = x[:, :d], x[:, d:]
            else:
                # last layer, no factor out
                factor_out = x
            out.append(factor_out)
            atol_list += atol; rtol_list += rtol; logpa_list += logp_actions; nfe_list += nfe
        
        out = [o.view(o.size()[0], -1) for o in out]
        out = torch.cat(out, 1)
        
        return (out, atol_list, rtol_list, logpa_list, nfe_list) if logpx is None else (out, _logpx, atol_list, rtol_list, logpa_list, nfe_list)
    
    @staticmethod
    def loss_class(y_logits, y):
        if y_logits is None:
            return 0
        else:
            CE = torch.nn.CrossEntropyLoss()
            return CE(y_logits, y.long())

    def _generate(self, z, logpz=None):
        z = z.view(z.shape[0], -1)
        zs = []
        atol_list = []; rtol_list = []; logpa_list = []; nfe_list = []
        i = 0
        for dims in self.dims:
            s = np.prod(dims)
            zs.append(z[:, i:i + s])
            i += s
        zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]
        _logpz = torch.zeros(zs[0].shape[0], 1).to(zs[0]) if logpz is None else logpz
        z_prev, _logpz, atol, rtol, logp_actions, nfe = self.transforms[-1](zs[-1], _logpz, reverse=True)
        atol_list += atol; rtol_list += rtol; logpa_list += logp_actions; nfe_list += nfe
        for idx in range(len(self.transforms) - 2, -1, -1):
            z_prev = torch.cat((z_prev, zs[idx]), dim=1)
            z_prev, _logpz, atol, rtol, logp_actions, nfe = self.transforms[idx](z_prev, _logpz, reverse=True)
            atol_list += atol; rtol_list += rtol; logpa_list += logp_actions; nfe_list += nfe
        return (z_prev, atol_list[::-1], rtol_list[::-1], logpa_list[::-1], nfe_list[::-1]) if logpz is None else (z_prev, _logpz, atol_list[::-1], rtol_list[::-1], logpa_list[::-1], nfe_list[::-1])


class StackedCNFLayers(layers.SequentialFlow_Gate):
    def __init__(
        self,
        initial_size,
        idims=(32,),
        nonlinearity="softplus",
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        cnf_kwargs={},
    ):
        strides = tuple([1] + [1 for _ in idims])
        chain = []
        if init_layer is not None:
            chain.append(init_layer)

        def _make_odefunc(size):
            net = ODEnet(idims, size, strides, True, layer_type="concat", nonlinearity=nonlinearity)
            f = layers.ODEfunc(net)
            return f
            
        cnf_kwargs_hightol={"T": cnf_kwargs['T'], "train_T": cnf_kwargs['train_T'], "regularization_fns": cnf_kwargs['regularization_fns'], "solver": cnf_kwargs['solver'], "atol_forward": cnf_kwargs['atol_forward']*0.1, "rtol_forward": cnf_kwargs['rtol_forward']*0.1, "atol_reverse": cnf_kwargs['atol_reverse']*0.1, "rtol_reverse": cnf_kwargs['rtol_reverse']*0.1, "scale": cnf_kwargs['scale'], "gate": cnf_kwargs['gate']}
        if squeeze:
            c, h, w = initial_size
            after_squeeze_size = c * 4, h // 2, w // 2
            pre = [layers.CNF_Gate_Sep(_make_odefunc(initial_size), c, **cnf_kwargs_hightol)]
            for _ in range(n_blocks - 1):
                pre.append(layers.CNF_Gate_Sep(_make_odefunc(initial_size), c, **cnf_kwargs))
            post = [layers.CNF_Gate_Sep(_make_odefunc(after_squeeze_size), c*4, **cnf_kwargs) for _ in range(n_blocks)]
            chain += pre + [layers.SqueezeLayer(2)] + post
        else:
            c, h, w = initial_size
            chain += [layers.CNF_Gate_Sep(_make_odefunc(initial_size), c, **cnf_kwargs_hightol)]
            chain += [layers.CNF_Gate_Sep(_make_odefunc(initial_size), c, **cnf_kwargs) for _ in range(n_blocks-1)]
            

        super(StackedCNFLayers, self).__init__(chain)
