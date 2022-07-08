import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg

DEFAULT_DTYPE = torch.float32


def l2_normalize(x, eps=1e-12):
    return x / (x.pow(2).sum() + eps).sqrt()


def power_iteration(w, u_init, num_iter=1):
    # w: (F_out, F_in)
    # u_init: (F_out, )
    u = u_init
    with torch.no_grad():
        for _ in range(num_iter - 1):
            v = l2_normalize(w.T.mv(u))
            u = l2_normalize(w.mv(v))
        v = l2_normalize(w.T.mv(u))
    wv = w.mv(v)  # note that this node allows gradient flow
    u = l2_normalize(wv.detach())
    sigma = u.dot(wv)
    return sigma, u


class SNLinear(nn.Module):
    dtype = DEFAULT_DTYPE

    def __init__(self, in_ft, out_ft, bias=True, use_gamma=True, pow_iter=1, lip_const=1):
        super(SNLinear, self).__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.use_gamma = use_gamma
        self.pow_iter = pow_iter
        self.lip_const = lip_const
        self.weight = nn.Parameter(torch.empty((out_ft, in_ft), dtype=self.dtype))
        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.zeros((out_ft, ), dtype=self.dtype)))
        else:
            self.register_buffer("bias", None)
        self.lip_const = lip_const
        self.register_buffer("u", torch.randn((out_ft, ), dtype=self.dtype))
        if use_gamma:
            self.register_parameter(
                "gamma", nn.Parameter(torch.ones((1, ), dtype=self.dtype)))
        else:
            self.register_paramter("gamma", None)

        # initialize the parameters
        nn.init.kaiming_normal_(self.weight, mode="fan_in")

    def _init_gamma(self):
        if self.use_gamma:
            nn.init.constant_(
                self.gamma, scipy.linalg.svd(self.weight.data, compute_uv=False)[0])

    @property
    def weight_bar(self):
        sigma, u = power_iteration(self.weight, self.u, self.pow_iter)
        if self.training:
            self.u = u
        weight_bar = self.lip_const * self.weight / sigma
        if self.use_gamma:
            weight_bar = self.gamma * weight_bar
        return weight_bar

    def forward(self, x):
        return F.linear(x, self.weight_bar, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_ft, self.out_ft, self.bias is not None
        )


if __name__ == "__main__":
    layer = SNLinear(100, 200)
    x = torch.randn(64, 100)
    print(layer(x).shape)
