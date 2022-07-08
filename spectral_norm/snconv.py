import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg

DEFAULT_DTYPE = torch.float32


def l2_normalize(x, eps=1e-12):
    return x / (x.pow(2).sum() + eps).sqrt()


""" OLD VERSION
def power_iteration(w, u_init, num_iter=1):
    # w: (c_out, C_in, K, K)
    # u_init: (c_out, )
    u = u_init
    with torch.no_grad():
        for _ in range(num_iter - 1):
            v = l2_normalize(torch.einsum("i,ijkl->jkl", u, w))
            u = l2_normalize(torch.einsum("ijkl,jkl->i", w, v))
        v = l2_normalize(torch.einsum("i,ijkl->jkl", u, w))
    wv = torch.einsum("ijkl,jkl->i", w, v)  # note that this node allows gradient flow
    u = l2_normalize(wv.detach())
    sigma = (u*wv).sum()
    return sigma, u
"""


def power_iteration(w, u_init, num_iter=1):
    # w: (c_out, C_in, K, K)
    # u_init: (c_out, )
    u = u_init
    w_flat = w.flatten(start_dim=1)
    with torch.no_grad():
        for _ in range(num_iter - 1):
            v = l2_normalize(w_flat.T.mv(u))
            u = l2_normalize(w_flat.mv(v))
        v = l2_normalize(w_flat.T.mv(u))
    wv = w_flat.mv(v)  # note that this node allows gradient flow
    u = l2_normalize(wv.detach())
    sigma = u.dot(wv)
    return sigma, u


class SNConv2d(nn.Module):
    dtype = DEFAULT_DTYPE

    def __init__(self, in_ch, out_ch, ksz, stride, pad, bias=True, use_gamma=True, pow_iter=1, lip_const=1):
        super(SNConv2d, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ksz = ksz
        self.stride = stride
        self.pad = pad
        self.use_gamma = use_gamma
        self.pow_iter = pow_iter
        self.lip_const = lip_const
        self.weight = nn.Parameter(
            torch.empty((out_ch, in_ch, ksz, ksz), dtype=self.dtype))
        if bias:
            self.register_parameter(
                "bias", nn.Parameter(torch.zeros((out_ch,), dtype=self.dtype)))
        else:
            self.register_buffer("bias", None)
        self.lip_const = lip_const
        self.register_buffer("u", torch.randn((out_ch,), dtype=self.dtype))
        if use_gamma:
            self.register_parameter(
                "gamma", nn.Parameter(torch.ones((1,), dtype=self.dtype)))
        else:
            self.register_buffer("gamma", None)

        # initialize the parameters
        nn.init.kaiming_normal_(self.weight, mode="fan_in")

    def _init_gamma(self):
        if self.use_gamma:
            nn.init.constant_(
                self.gamma, scipy.linalg.svd(
                    self.weight.data.reshape(self.out_ch, -1), compute_uv=False)[0])

    def forward(self, x):
        return F.conv2d(x, self.weight_bar, self.bias, stride=self.stride, padding=self.pad)

    @property
    def weight_bar(self):
        sigma, u = power_iteration(self.weight, self.u, self.pow_iter)
        if self.training:
            self.u = u
        weight_bar = self.lip_const * self.weight / sigma
        if self.use_gamma:
            weight_bar = self.gamma * weight_bar
        return weight_bar

    def extra_repr(self):
        s = ('{in_ch}, {out_ch}, kernel_size={ksz}'
             ', stride={stride}')
        if self.pad != 0:
            s += ', padding={pad}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


if __name__ == "__main__":
    layer = SNConv2d(100, 200, 3, 1, 1)
    x = torch.randn(64, 100, 3, 3)
    print(layer(x).shape)
