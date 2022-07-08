####################################################
# example code of visualizing the truncation effects
####################################################

import sys
import math
import torch
import numpy as np
from load_model import load_model
from torchvision.utils import make_grid
from PIL import Image

chkpt_path = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG, ema = load_model(chkpt_path, device)

torch.manual_seed(71)
padding = 2
std_min, std_max = math.pow(0.3, 1 / 3), math.pow(3, 1 / 3)

frames = 12
fps = 1

imgs = []
with ema:
    for _ in range(frames):
        x_gen = []
        gen_noise = torch.randn((8, 128))
        for std in torch.linspace(std_min, std_max, 8).pow(2):
            with torch.no_grad():
                x_gen.append(netG.sample(8, noise=std * gen_noise).cpu())
        x_gen = torch.cat(x_gen, dim=0)
        gen_imgs = make_grid(
            x_gen, nrow=8, normalize=True, value_range=(-1, 1), pad_value=0, padding=padding
        ).numpy().transpose(1, 2, 0)
        size = gen_imgs.shape[0]
        inds = torch.cat([torch.arange(padding, padding + 64) + (64 + padding) * i for i in range(8)])
        inds = torch.cat([torch.arange(padding), inds, torch.arange(size - padding, size)])
        gen_imgs = gen_imgs[:, inds, :]
        imgs.append(gen_imgs)
outs = [Image.fromarray((im * 255).astype(np.uint8), mode="RGB") for im in imgs]
duration = np.ones(144) * 1000 / fps
outs[0].save("./truncation-effect.gif", save_all=True, append_images=outs[1:], optimize=True, duration=list(duration),
             loop=0)
