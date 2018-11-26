import types

from collections import defaultdict

import numpy as np
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


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('mask', mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


nn.MaskedLinear = MaskedLinear


class MADE(nn.Module):
    """ An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self, num_inputs, num_hidden):
        super(MADE, self).__init__()

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type='input')
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type='output')

        self.main = nn.Sequential(
            nn.MaskedLinear(num_inputs, num_hidden, input_mask), nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_hidden, hidden_mask), nn.ReLU(),
            nn.MaskedLinear(num_hidden, num_inputs * 2, output_mask))

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            x = self.main(inputs)

            m, a = x.chunk(2, 1)

            u = (inputs - m) * torch.exp(a)
            return u, a.sum(-1, keepdim=True)
        else:
            # TODO:
            # Sampling with MADE is tricky.
            # We need to perform N forward passes.
            raise NotImplementedError


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

    def forward(self, inputs, mode='direct', **kwargs):
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

    def forward(self, inputs, mode='direct', **kwargs):
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

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            return inputs @ self.W, torch.log(torch.abs(torch.det(
                self.W))).unsqueeze(0).unsqueeze(0).repeat(inputs.size(0), 1)
        else:
            return inputs @ torch.inverse(self.W), -torch.log(
                torch.abs(torch.det(self.W))).unsqueeze(0).unsqueeze(0).repeat(
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

    def forward(self, inputs, mode='direct', **kwargs):
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

    def forward(self, inputs, mode='direct', **kwargs):
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

    def __init__(self, num_inputs, num_hidden=64):
        super(CouplingLayer, self).__init__()

        self.num_inputs = num_inputs

        self.main = nn.Sequential(
            nn.Linear(num_inputs // 2, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, num_hidden), nn.ReLU(),
            nn.Linear(num_hidden, 2 * (self.num_inputs - num_inputs // 2)))

        def init(m):
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, inputs, mode='direct', **kwargs):
        if mode == 'direct':
            x_a, x_b = inputs.chunk(2, dim=-1)
            log_s, t = self.main(x_b).chunk(2, dim=-1)
            s = torch.exp(log_s)

            y_a = x_a * s + t
            y_b = x_b
            return torch.cat([y_a, y_b], dim=-1), log_s.sum(-1, keepdim=True)
        else:
            y_a, y_b = inputs.chunk(2, dim=-1)
            log_s, t = self.main(y_b).chunk(2, dim=-1)
            s = torch.exp(-log_s)
            x_a = (y_a - t) * s
            x_b = y_b
            return torch.cat([x_a, x_b], dim=-1), -log_s.sum(-1, keepdim=True)


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
    """ An implementation of the radial layer from
    Variational Inference with Normalizing Flows
    (https://arxiv.org/abs/1505.05770).
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

    def forward(self, inputs, mode='direct', logdets=None, params=None):
        """ Performs a forward or backward pass for flow modules.
        Allows building hypernetworks by passing params.
        If params==None it uses values provided by the module's nn.Parameter objects.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
            params: 2dim tensor with shape [batch_size, parameters_nelement]
        """
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