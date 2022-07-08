########################################################################
# example code of generating an animation of walking in the latent space
########################################################################

import sys
import torch
from load_model import load_model
import numpy as np
from PIL import Image

chkpt_path = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG, ema = load_model(chkpt_path, device)

n_samp = 10
torch.manual_seed(1234)
noise = torch.fmod(torch.randn((n_samp, 128), device=device), 2)

intp = torch.linspace(0, 1, 25)[:-1]
intp = torch.stack([1 - intp, intp], dim=1).to(device)
xs = []

with torch.no_grad():
    with ema:
        for i in range(n_samp):
            xs.append(netG(intp @ noise[[i % n_samp, (i + 1) % n_samp]]).cpu())
xs = torch.cat(xs, dim=0).numpy().transpose(0, 2, 3, 1)
xs = ((xs + 1) / 2 * 255).round().astype(np.uint8)

size = xs.shape[1:3]
to_size = (size[0]*4, size[1]*4)
outs = [Image.fromarray(x, mode="RGB").resize(to_size) for x in xs]
outs[-1].save("latent-walk.gif", save_all=True, append_images=outs[1:], optimize=True, duration=1000 / 24, loop=0)
