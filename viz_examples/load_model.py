#########################################
# example code of loading a trained model
#########################################

def load_model(chkpt_path, device):
    import torch
    try:
        from ..sngan import NetG
        from ..utils import EMA
    except (ImportError, ModuleNotFoundError):
        import sys
        from pathlib import Path
        parent = str(Path(__file__).resolve().parents[1])
        sys.path.insert(0, parent)
        from sngan import NetG
        from utils import EMA

    latent_dim = 128
    num_blocks = [1, 1, 1, 1]
    netG = NetG(3, latent_dim=latent_dim, num_blocks=num_blocks)

    chkpt = torch.load(chkpt_path, map_location=device)
    netG.load_state_dict(chkpt["netG"])
    netG.eval()
    netG.to(device)
    ema = EMA(netG)
    ema.__dict__.update(chkpt["ema"])
    del chkpt
    return netG, ema
