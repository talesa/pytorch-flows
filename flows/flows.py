import math
import types

from collections import defaultdict

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

EPS = 1e-7

def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output
    
    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == 'input':
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == 'output':
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 mask,
                 cond_in_features=None,
                 bias=True):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        if cond_in_features is not None:
            self.cond_linear = nn.Linear(
                cond_in_features, out_features, bias=False)

        self.register_buffer('mask', mask)

    def forward(self, inputs, cond_inputs=None):
        output = F.linear(inputs, self.linear.weight * self.mask,
                          self.linear.bias)
        if cond_inputs is not None:
            output += self.cond_linear(cond_inputs)
        return output


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 num_cond_inputs=None,
                 act='relu'):
        super(MADE, self).__init__()

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.joiner = nn.MaskedLinear(num_inputs, num_hidden, input_mask,
                                      num_cond_inputs)

        self.trunk = nn.Sequential(act_func(),
                                   nn.MaskedLinear(num_hidden, num_hidden,
                                                   hidden_mask), act_func(),
                                   nn.MaskedLinear(num_hidden, num_inputs * 2,
                                                   output_mask))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            h = self.joiner(inputs, cond_inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x, cond_inputs)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = inputs[:, i_col] * torch.exp(
                    a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)


class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            s = torch.sigmoid
            return s(inputs), torch.log(s(inputs) * (1 - s(inputs))).sum(
                -1, keepdim=True)
        else:
            return torch.log(inputs /
                             (1 - inputs)), -torch.log(inputs - inputs**2).sum(
                                 -1, keepdim=True)


class Logit(Sigmoid):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return super(Logit, self).forward(inputs, 'inverse')
        else:
            return super(Logit, self).forward(inputs, 'direct')


class BatchNormFlow(nn.Module):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)


class ActNorm(nn.Module):
    """ An implementation of a activation normalization layer
    from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(ActNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(num_inputs))
        self.bias = nn.Parameter(torch.zeros(num_inputs))
        self.initialized = False

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if self.initialized == False:
            self.weight.data.copy_(torch.log(1.0 / (inputs.std(0) + 1e-12)))
            self.bias.data.copy_(inputs.mean(0))
            self.initialized = True

        if mode == 'direct':
            return (
                inputs - self.bias) * torch.exp(self.weight), self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs * torch.exp(
                -self.weight) + self.bias, -self.weight.sum(
                    -1, keepdim=True).unsqueeze(0).repeat(inputs.size(0), 1)


class InvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(InvertibleMM, self).__init__()
        self.W = nn.Parameter(torch.Tensor(num_inputs, num_inputs))
        nn.init.orthogonal_(self.W)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs @ self.W, torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.slogdet(
                self.W)[-1].unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class LUInvertibleMM(nn.Module):
    """ An implementation of a invertible matrix multiplication
    layer from Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    """

    def __init__(self, num_inputs):
        super(LUInvertibleMM, self).__init__()
        self.W = torch.Tensor(num_inputs, num_inputs)
        nn.init.orthogonal_(self.W)
        self.L_mask = torch.tril(torch.ones(self.W.size()), -1)
        self.U_mask = self.L_mask.t().clone()

        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.U = nn.Parameter(torch.from_numpy(U))

        S = np.diag(U)
        sign_S = np.sign(S)
        log_S = np.log(abs(S))
        self.sign_S = torch.from_numpy(sign_S)
        self.log_S = nn.Parameter(torch.from_numpy(log_S))

        self.I = torch.eye(self.L.size(0))

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if str(self.L_mask.device) != str(self.L.device):
            self.L_mask = self.L_mask.to(self.L.device)
            self.U_mask = self.U_mask.to(self.L.device)
            self.I = self.I.to(self.L.device)
            self.P = self.P.to(self.L.device)
            self.sign_S = self.sign_S.to(self.L.device)

        L = self.L * self.L_mask + self.I
        U = self.U * self.U_mask + torch.diag(
            self.sign_S * torch.exp(self.log_S))
        W = self.P @ L @ U

        if mode == 'direct':
            return inputs @ W, self.log_S.sum().unsqueeze(0).unsqueeze(
                0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(
                W), -self.log_S.sum().unsqueeze(0).unsqueeze(0).repeat(
                    inputs.size(0), 1)


class Shuffle(nn.Module):
    """ An implementation of a shuffling layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Shuffle, self).__init__()
        self.perm = np.random.permutation(num_inputs)
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class Reverse(nn.Module):
    """ An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        if mode == 'direct':
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device)


class CouplingLayer(nn.Module):
    """ An implementation of a coupling layer
    from RealNVP (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self,
                 num_inputs,
                 num_hidden,
                 mask,
                 num_cond_inputs=None,
                 s_act='tanh',
                 t_act='relu'):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs
        self.mask = mask

        activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh}
        s_act_func = activations[s_act]
        t_act_func = activations[t_act]

        if num_cond_inputs is not None:
            total_inputs = num_inputs + num_cond_inputs
        else:
            total_inputs = num_inputs

        self.scale_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_hidden), s_act_func(),
            nn.Linear(num_hidden, num_inputs))
        self.translate_net = nn.Sequential(
            nn.Linear(total_inputs, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_hidden), t_act_func(),
            nn.Linear(num_hidden, num_inputs))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, cond_inputs=None, mode='direct'):
        mask = self.mask

        masked_inputs = inputs * mask
        if cond_inputs is not None:
            masked_inputs = torch.cat([masked_inputs, cond_inputs], -1)

        if mode == 'direct':
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(log_s)
            return inputs * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_inputs) * (1 - mask)
            t = self.translate_net(masked_inputs) * (1 - mask)
            s = torch.exp(-log_s)
            return (inputs - t) * s, -log_s.sum(-1, keepdim=True)


class RadialFlow(nn.Module):
    """ An implementation of the radial layer from
    Variational Inference with Normalizing Flows
    (https://arxiv.org/abs/1505.05770).
    """

    def __init__(self, num_inputs):
        super(RadialFlow, self).__init__()
        self.z0 = nn.Parameter(torch.zeros(1, num_inputs))
        self.log_a = nn.Parameter(torch.zeros(1, 1))
        # bhat is b before reparametrization
        self.bhat = nn.Parameter(torch.zeros(1, 1))

        self.num_inputs = num_inputs

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        if params is None:
            return self.Fforward(inputs, mode, self.num_inputs, self.z0, self.log_a, self.bhat)
        return self.Fforward(inputs, mode, self.num_inputs, **params)

    @staticmethod
    def Fforward(inputs, mode, num_inputs, z0, log_a, bhat):
        """ Functional version of the forward function. """
        if len(inputs.shape) > 1:
            assert inputs.shape[1] == num_inputs

        # offset to make the flow an identity flow if all parameters are zeros
        bhat = bhat + (torch.ones(1, 1).exp() - torch.ones(1, 1)).log()

        d = float(num_inputs)
        a = log_a.exp()
        # according to the Appendix in the paper
        b = -a + torch.nn.functional.softplus(bhat)
        if mode == 'direct':
            z = inputs
            z_z0 = z - z0
            r = z_z0.norm(dim=-1, keepdim=True) + EPS
            h = 1 / (a + r)
            hprime = -1. / (a + r).pow(2)
            logdet = (d-1)*(1. + b * h).log() + (1. + b * h + b * hprime * r).log()
            output = inputs + b * z_z0 * h
            return output, logdet
        else:
            y = inputs
            y_z0 = y - z0
            c = y_z0.norm(dim=-1, keepdim=True)
            B = a + b - c
            sqrt_delta = (B.pow(2) + 4 * a * c).sqrt()
            r = 0.5 * (-B + sqrt_delta) + EPS
            h = 1 / (a + r)
            zhat = y_z0 / (r * (1 + b / (a + r)))
            hprime = -1. / (a + r).pow(2)
            output = z0 + r * zhat
            inv_logdet = -((d - 1) * (1. + b * h).log() + (1. + b * h + b * hprime * r).log())
            return output, inv_logdet


class ExponentialFlow(nn.Module):
    """ An implementation of a transformation y = exp(x).
    Used to ensure the output is positive.
    """

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] > 0
        if inputs.shape[1] > 1: print('warning, I dunno if this should work')
        if mode == 'direct':
            x = inputs
            logdet = x.sum(dim=-1, keepdim=True)
            y = x.exp()
            return y, logdet
        else:
            y = inputs
            x = y.log()
            inv_logdet = -x.sum(dim=-1, keepdim=True)
            return x, inv_logdet


class SigmoidFlow(nn.Module):
    """ An implementation of a sigmoid transformation y = 1/(1+exp(-x)).
    Used to ensure the output is within the range [0, 1].
    """

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] > 0
        if inputs.shape[1] > 1: print('warning, I dunno if this should work')
        if mode == 'direct':
            x = inputs
            y = 1./(1.+(-x).exp())
            logdet = (y.log()+(1-y).log()).sum(dim=-1, keepdim=True)
            return y, logdet
        else:
            y = inputs
            x = -(1./y-1.).log()
            inv_logdet = -(y.log()+(1-y).log()).sum(dim=-1, keepdim=True)
            return x, inv_logdet


class IdentitySigmoidFlow(nn.Module):
    """ An implementation of a sigmoid transformation y=1/(1+exp(-z)), z=4.5(x-0.5).
    Used to ensure the output is within the range [0, 1], different from SigmoidFlow
    in that it applies a smaller transformation in the region close to x=0.5.
    https://www.desmos.com/calculator/z4viqqotai
    """

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] > 0
        if inputs.shape[1] > 1: print('warning, I dunno if this should work')
        if mode == 'direct':
            x = inputs
            z = 4.5*(x-0.5)
            y = 1./(1.+(-z).exp())
            logdet = (y.log()+(1-y).log()+tensor(4.5).log()).sum(dim=-1, keepdim=True)
            return y, logdet
        else:
            y = inputs
            z = -(1./y-1.).log()
            x = z/4.5+0.5
            inv_logdet = -(y.log()+(1-y).log()+tensor(4.5).log()).sum(dim=-1, keepdim=True)
            return x, inv_logdet


class LinearSigmoidFlow(nn.Module):
    """ An implementation of a linear sigmoid transformation.
    Used to ensure the output is within the range [0, 1], different from SigmoidFlow
    in that it applies only a small transformation in the region [0,1].
    For details see the desmos calculator below
    https://www.desmos.com/calculator/eaccyxzzuv
    """

    def __init__(self, t=7.):
        super(LinearSigmoidFlow, self).__init__()
        self.t = tensor(t, dtype=torch.float)
        self.i = torch.sigmoid(self.t)
        self.q = 1. - self.i

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] > 0
        if inputs.shape[1] > 1: print('warning, I dunno if this should work')

        t = self.t
        i = self.i
        q = self.q

        if mode == 'direct':
            x = inputs

            y = (1. - 2. * q) * x + q
            y = torch.where(x < 0.,
                            torch.sigmoid(x - t),
                            y)
            y = torch.where(x < 1.,
                            y,
                            torch.sigmoid(x + t - 1.))

            logdet = (1. - 2. * q).log() * x.shape[1]
            logdet2 = (y.log() + (1. - y).log()).sum(dim=-1, keepdim=True)
            logdet = torch.where(x < 0., logdet2, logdet)
            logdet = torch.where(x < 1., logdet,  logdet2)
            return y, logdet
        else:
            y = inputs

            if (y <= 0.).any():
                raise Exception('Input to inverse sigmoid smaller than 0')
            if (y >= 1.).any():
                raise Exception('Input to inverse sigmoid larger than 1')

            x = (y - q) / (1. - 2. * q)
            x = torch.where(y <= q,
                            -(1. / y - 1).log() + t,
                            x)
            x = torch.where(y <= i,
                            x,
                            -(1. / y - 1.).log() - t + 1.)

            logdet = (1. - 2. * q).log() * x.shape[1]
            logdet2 = (y.log() + (1. - y).log()).sum(dim=-1, keepdim=True)
            logdet = torch.where(x < q, logdet2, logdet)
            logdet = torch.where(x < i, logdet, logdet2)
            inv_logdet = -logdet
            return x, inv_logdet


class LocAndScaleFlow(nn.Module):
    """ A loc and scale layer.
    """

    def __init__(self, num_inputs):
        super(LocAndScaleFlow, self).__init__()
        self.loc = nn.Parameter(torch.zeros(1, num_inputs))
        self.scale = nn.Parameter(torch.zeros(1, num_inputs))
        self.num_inputs = num_inputs

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] == self.num_inputs
        if params is None:
            return self.Fforward(inputs, mode, self.num_inputs, self.loc, self.scale)
        return self.Fforward(inputs, mode, self.num_inputs, **params)

    @staticmethod
    def Fforward(inputs, mode, num_inputs, loc, scale):
        scale = scale + torch.ones(1, num_inputs)

        if mode == 'direct':
            x = inputs
            y = loc + x * scale
            logdet = scale.log().sum(dim=-1, keepdim=True)
            return y, logdet
        else:
            y = inputs
            x = (y-loc)/scale
            inv_logdet = -scale.log().sum(dim=-1, keepdim=True)
            return x, inv_logdet


class OffsetFlow(nn.Module):
    """ A flow layer adding an offset.
    """

    def __init__(self, num_inputs):
        super(OffsetFlow, self).__init__()
        self.loc = nn.Parameter(torch.zeros(1, num_inputs))
        self.num_inputs = num_inputs

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] == self.num_inputs
        if params is None:
            return self.Fforward(inputs, mode, self.num_inputs, self.loc)
        return self.Fforward(inputs, mode, self.num_inputs, **params)

    @staticmethod
    def Fforward(inputs, mode, num_inputs, loc):
        if mode == 'direct':
            x = inputs
            y = loc + x
            logdet = 0.
            return y, logdet
        else:
            y = inputs
            x = y - loc
            inv_logdet = 0.
            return x, inv_logdet


class SoftplusFlow(nn.Module):
    """ An implementation of a softplus transformation y = log(1+exp(x)).
    Used to ensure the output is positive.
    """

    def forward(self, inputs, mode='direct', params=None, **kwargs):
        assert inputs.shape[1] > 0
        if inputs.shape[1] > 1: print('warning, I dunno if this should work')
        if mode == 'direct':
            x = inputs
            y = torch.where(x > 20, x, (1+x.exp()).log())
            logdet = torch.where(x > 20, torch.ones(x.shape[0], 1), (1 / (1 + (-x).exp())).sum(dim=-1, keepdim=True)).log()
            return y, logdet
        else:
            y = inputs
            x = torch.where(y > 20, y, (y.exp() - 1).log())
            inv_logdet = -torch.where(y > 20, torch.ones(x.shape[0], 1), (1 / (1 + (-x).exp())).sum(dim=-1, keepdim=True)).log()
            return x, inv_logdet


class FlowSequential(nn.Sequential):
    """ A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    Allows building hypernetworks by passing params to the forward() function
    rather than using values provided by the module's nn.Parameter objects.
    """

    def parameters_nelement(self):
        """ Returns the total number of elements of the parameters in the Module. """
        return sum(p.nelement() for p in self.parameters())

    def forward(self, inputs, cond_inputs=None, mode='direct', logdets=None, params=None):
        """ Performs a forward or backward pass for flow modules.
        Allows building hypernetworks by passing params.
        If params==None it uses values provided by the module's nn.Parameter objects.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
            params: 2dim tensor with shape [batch_size, parameters_nelement]
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        params_per_module = defaultdict(lambda: None)
        if params is not None:
            params_per_module = defaultdict(dict)
            i = 0
            for module_parameter_name, parameter in self.named_parameters():
                module_name, parameter_name = module_parameter_name.split('.')
                params_per_module[module_name][parameter_name] = \
                    params[:,i:i+parameter.nelement()].reshape(-1, *parameter.shape[1:])
                i += parameter.nelement()

        assert mode in ['direct', 'inverse']
        if mode == 'direct':
            for module_name, module_object in self._modules.items():
                inputs, logdet = module_object(inputs, mode, params=params_per_module[module_name])
                logdets += logdet
        else:
            for module_name, module_object in reversed(self._modules.items()):
                inputs, logdet = module_object(inputs, mode, params=params_per_module[module_name])
                logdets += logdet

        return inputs, logdets


class FlowDensityEstimator(torch.distributions.distribution.Distribution):
    """ A density estimator able to evaluate log_prob and generate samples.
    Requires specifying a base distribution.
    """

    def __init__(self, base_distribution, flow):
        super(FlowDensityEstimator, self).__init__()
        self.flow = flow
        self.base_distribution = base_distribution

    def log_prob(self, value, params=None):
        y = value
        x, inv_logdets = self.flow.forward(y, mode='inverse', params=params)
        log_prob = self.base_distribution.log_prob(x)
        if len(inv_logdets.shape) > 1:
            inv_logdets = inv_logdets.sum(dim=1)
        assert inv_logdets.shape == log_prob.shape
        log_prob += inv_logdets
        return torch.clamp(log_prob, -1e38, 1e38)

    def sample(self, sample_shape, params=None):
        if len(sample_shape) > 0 and sample_shape[0] % params.shape[0] == 0:
            shape = sample_shape
        else:
            shape = (params.shape[0],)
        x = self.base_distribution.sample(shape)
        y, _ = self.flow.forward(x, mode='direct', params=params)
        return y

    def rsample(self, sample_shape, params=None):
        if not self.base_distribution.has_rsample:
            raise NotImplemented()
        if len(sample_shape) > 0 and sample_shape[0] % params.shape[0] == 0:
            shape = sample_shape
        else:
            shape = (params.shape[0],)
        x = self.base_distribution.rsample(shape)
        y, _ = self.flow.forward(x, mode='direct', params=params)
        return y