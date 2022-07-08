#####################################################################
# example code of generating an animated 8x8 image grid interpolation
#####################################################################

import torch.nn.functional as F

import sys
import math
import torch
from load_model import load_model
from torchvision.utils import make_grid
import numpy as np
from PIL import Image

chkpt_path = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG, ema = load_model(chkpt_path, device)

latent_dim = 128
key_frames = 4
fps = 24
torch.manual_seed(1324)
noise = torch.fmod(torch.randn((4 * key_frames, 128), device=device), 2)
noise = noise.reshape(key_frames, 2, 2, -1).permute(0, 3, 1, 2)


def normalize(x, mode=latent_dim-2):
    return x / x.pow(2).sum(dim=-1, keepdim=True).sqrt() * math.sqrt(mode)


imgs = []
intp = torch.linspace(0, 1, 25)[:-1]
intp = torch.stack([1 - intp, intp], dim=0).to(device)
batch_size = 128
with ema:
    x_int = F.interpolate(
        noise, size=(8, 8), mode="bicubic", align_corners=True
    ).permute(0, 2, 3, 1).reshape(key_frames, 64, latent_dim)
    for i in range(key_frames):
        x = x_int[[i%key_frames, (i+1)%key_frames]].permute(1, 2, 0) @ intp
        x = x.permute(2, 0, 1).reshape(-1, latent_dim)
        n = x.shape[0]
        for j in range(0, n, batch_size):
            imgs.append(
                netG.sample(batch_size, noise=normalize(x[j:j+batch_size])).detach().cpu())
imgs = torch.cat(imgs, dim=0)
N = imgs.shape[0]
imgs = [make_grid(
    imgs[i: i+64], nrow=8, normalize=True, value_range=(-1, 1)
).numpy().transpose(1, 2, 0) for i in range(0, N, 64)]

outs = [Image.fromarray((img * 255).round().astype(np.uint8), mode="RGB") for img in imgs]
outs[-1].save("interpolation.gif", save_all=True, append_images=outs[1:], optimize=True, duration=1000 / 24, loop=0)
