"""CycleGAN networks: ResNet generator + 70x70 PatchGAN discriminator."""
import torch.nn as nn


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """in_ch -> out_ch image translator, output in [-1, 1] (tanh)."""

    def __init__(self, in_ch, out_ch, ngf=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]
        mult = 1
        for _ in range(2):  # downsample x2, twice
            layers += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2
        layers += [ResnetBlock(ngf * mult) for _ in range(n_blocks)]
        for _ in range(2):  # upsample x2, twice
            layers += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, 3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (n_layers=3). Outputs a map of real/fake scores."""

    def __init__(self, in_ch, ndf=64, n_layers=3):
        super().__init__()
        k, p = 4, 1
        layers = [nn.Conv2d(in_ch, ndf, k, 2, p), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            layers += [
                nn.Conv2d(ndf * prev, ndf * mult, k, 2, p, bias=False),
                nn.InstanceNorm2d(ndf * mult),
                nn.LeakyReLU(0.2, True),
            ]
        prev, mult = mult, min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(ndf * prev, ndf * mult, k, 1, p, bias=False),
            nn.InstanceNorm2d(ndf * mult),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * mult, 1, k, 1, p),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def init_weights(net, std=0.02):
    def f(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.InstanceNorm2d) and m.weight is not None:
            nn.init.normal_(m.weight, 1.0, std)
            nn.init.zeros_(m.bias)

    net.apply(f)
    return net
