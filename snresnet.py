import torch.nn as nn
from snconv import SNConv2d
from snlinear import SNLinear

normalize = nn.BatchNorm2d


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=False, activation=nn.ReLU, use_sn=True, use_bias=False, use_norm=True, **act_kwargs):
        super(ResidualBlock, self).__init__()
        Conv2d = SNConv2d if use_sn else nn.Conv2d
        norm = normalize if use_norm else lambda _: nn.Identity()
        self.norm1 = norm(in_ch)
        self.act1 = activation(**act_kwargs)
        self.conv1 = Conv2d(in_ch, out_ch, 3, 1+downsample, 1, bias=use_bias)
        self.norm2 = norm(out_ch)
        self.act2 = activation(**act_kwargs)
        self.conv2 = Conv2d(out_ch, out_ch, 3, 1, 1, bias=use_bias)
        if in_ch != out_ch or downsample:
            skip = [nn.AvgPool2d(2)] if downsample else []
            skip.append(Conv2d(in_ch, out_ch, 1, 1, 0, bias=use_bias))
            self.skip = nn.Sequential(*skip)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))
        out += self.skip(x)
        return out


def get_block(type="D"):
    if type == "D":
        def Block(in_ch, out_ch, downsample=False):
            return ResidualBlock(
                in_ch, out_ch, downsample, activation=nn.ReLU, use_sn=True, use_bias=True, use_norm=False, inplace=True)
    elif type == "G":
        def Block(in_ch, out_ch):
            return ResidualBlock(
                in_ch, out_ch, downsample=False, activation=nn.ReLU, use_sn=False, use_bias=False, use_norm=True, inplace=True)
    else:
        raise NotImplementedError(type)
    return Block


class SNResNet(nn.Module):

    def __init__(self, in_ch, base_ch=64, num_blocks=[2, 2, 2, 2]):
        super(SNResNet, self).__init__()
        self.in_conv = nn.Sequential(
            SNConv2d(in_ch, base_ch, 3, 1, 1), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(base_ch, 2 * base_ch, num_blocks[0])
        self.layer2 = self._make_layer(2 * base_ch, 4 * base_ch, num_blocks[1])
        self.layer3 = self._make_layer(4 * base_ch, 8 * base_ch, num_blocks[2])
        self.layer4 = self._make_layer(8 * base_ch, 16 * base_ch, num_blocks[3])
        self.out_fc = nn.Sequential(
            normalize(16 * base_ch),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(4),
            nn.Flatten(start_dim=1),
            SNLinear(16 * base_ch, 1)
        )

        # re-initialization
        for m in self.modules():
            if isinstance(m, (SNConv2d, SNLinear)):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
                if m.use_gamma:
                    m._init_gamma()

    @staticmethod
    def _make_layer(in_ch, out_ch, num_blocks):
        return nn.Sequential(
            ResidualBlock(in_ch, out_ch, downsample=True),
            *[ResidualBlock(out_ch, out_ch) for _ in range(num_blocks-1)])

    def forward(self, x):
        out = self.in_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return self.out_fc(out)


if __name__ == "__main__":
    import torch
    x = torch.randn((64, 3, 64, 64))
    model = SNResNet(3)
    print(model(x).shape)
