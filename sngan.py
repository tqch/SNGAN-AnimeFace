from snresnet import get_block
from snconv import SNConv2d
from snlinear import SNLinear
import torch
import torch.nn as nn

normalize_G = nn.BatchNorm2d
activation_D = nn.ReLU(inplace=True)
activation_G = nn.ReLU(inplace=True)
Block_D = get_block("D")
Block_G = get_block("G")


class NetD(nn.Module):

    def __init__(self, in_ch, base_ch=64, num_blocks=[2, 2, 2, 2]):
        super(NetD, self).__init__()
        self.in_block = Block_D(in_ch, base_ch, downsample=True)
        self.layer1 = self._make_layer(base_ch, 2 * base_ch, num_blocks[0])
        self.layer2 = self._make_layer(2 * base_ch, 4 * base_ch, num_blocks[1])
        self.layer3 = self._make_layer(4 * base_ch, 8 * base_ch, num_blocks[2])
        self.layer4 = self._make_layer(8 * base_ch, 16 * base_ch, num_blocks[3])
        self.out_fc = nn.Sequential(
            activation_D,
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(start_dim=1),
            SNLinear(16 * base_ch, 1)
        )

        # re-initialization
        for m in self.modules():
            if isinstance(m, (SNConv2d, SNLinear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="leaky_relu")
                if m.use_gamma:
                    m._init_gamma()

        # turn off inplace relu preceding the first layer
        self.in_block.act1 = nn.Identity()

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks):
        return nn.Sequential(
            Block_D(in_ch, out_ch, downsample=True),
            *[Block_D(out_ch, out_ch) for _ in range(num_blocks - 1)])

    def forward(self, x):
        out = self.in_block(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.out_fc(out)


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(self.shape)

    def extra_repr(self):
        return f"shape={self.shape}"


class NetG(nn.Module):

    def __init__(self, in_ch, base_ch=64, latent_dim=128, num_blocks=[2, 2, 2, 2]):
        super(NetG, self).__init__()
        self.in_fc = nn.Sequential(
            nn.Linear(latent_dim, 16 * base_ch * 4 ** 2, bias=False), Reshape((-1, 16 * base_ch, 4, 4)))
        self.layer1 = self._make_layer(16 * base_ch, 8 * base_ch, num_blocks[0])
        self.layer2 = self._make_layer(8 * base_ch, 4 * base_ch, num_blocks[1])
        self.layer3 = self._make_layer(4 * base_ch, 2 * base_ch, num_blocks[2])
        self.layer4 = self._make_layer(2 * base_ch, base_ch, num_blocks[3])
        self.out_conv = nn.Sequential(
            normalize_G(base_ch),
            activation_G,
            nn.Conv2d(base_ch, in_ch, 1, 1, 0),  # aggregate base_ch into RGB
            nn.Tanh()
        )

        # re-initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.0, mode="fan_in", nonlinearity="leaky_relu")

        self.latent_dim = latent_dim

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            Block_G(in_ch, out_ch), *[Block_G(out_ch, out_ch) for _ in range(num_blocks - 1)])

    def forward(self, x):
        out = self.in_fc(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.out_conv(out)

    def sample(self, n, noise=None):
        device = next(self.parameters()).device
        if noise is None:
            noise = torch.randn((n, self.latent_dim))
        return self.forward(noise.to(device))


if __name__ == "__main__":
    in_ch = 3
    base_ch = 64
    latent_dim = 128
    netD = NetD(in_ch, base_ch, num_blocks=[1, 1, 1, 1])
    netG = NetG(in_ch, base_ch, latent_dim, num_blocks=[1, 1, 1, 1])
    print(netD)
    print(netG)
    x_true = torch.randn(16, 3, 64, 64)
    x_fake = netG.sample(16)
    print(x_fake.shape)
    print(netD(x_true).shape, netD(x_fake).shape)
