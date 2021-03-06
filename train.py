if __name__ == "__main__":
    import os
    import math
    import torch
    import torch.nn as nn
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from torchvision import transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm
    from dataset import AnimeFace
    from sngan import NetD, NetG
    import torch.cuda.amp as amp
    from utils import BatchSizeTuner, EMA
    from contextlib import nullcontext

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--root", type=str, default="~/datasets")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--base-ch", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--num-blocks", type=str, default="1,1,1,1")
    parser.add_argument("--img-dir", type=str, default="./imgs")
    parser.add_argument("--nimgs", type=int, default=64)
    parser.add_argument("--chkpt-dir", type=str, default="./chkpts")
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--ttur", type=float, default=1)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--loss", choices=["vanilla", "hinge"], default="hinge")
    parser.add_argument("--d-freq", type=int, default=1)
    parser.add_argument("--d-iters", type=int, default=1)
    parser.add_argument("--d-minloss", type=float, default=None)
    parser.add_argument("--g-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chkpt-intv", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--auto-bsz", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    args = parser.parse_args()

    # train generative models on whole dataset
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(0.5),  # random horizontal flipping
        transforms.ToTensor(),  # 0..255 RGB to [0, 1] (C, H, W) Tensor
        transforms.Normalize((0.5,) * 3, (0.5,) * 3)  # rescale to [-1, 1]
    ])

    # turn on cudnn benchmarking for performance boost when available
    # it will select the fastest convolution algorithm
    cudnnbm = False
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = cudnnbm = True
        print(f"cuDNN benchmark: ON")

    # build dataloader
    root = os.path.expanduser(args.root)
    download = args.download
    traindata = AnimeFace(root=root, download=download, split="all", transform=transform)
    batch_size = args.batch_size

    # if seeing an error, change num_workers to 0
    trainloader = DataLoader(
        traindata, shuffle=True, batch_size=batch_size, drop_last=cudnnbm, pin_memory=True, num_workers=os.cpu_count())

    # training hyparameters
    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2
    print(f"Learning rate: {lr}\nBeta1: {beta1}\nBeta2: {beta2}")

    # training on gpu is preferred
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device.type}")

    # model hyparameters
    in_ch = 3
    base_ch = args.base_ch
    latent_dim = args.latent_dim
    print(f"Base channels: {base_ch}\nLatent dimensions: {latent_dim}")

    # build discriminator and generator
    print("Building discriminator and generator...", end="")
    num_blocks = [int(k) for k in args.num_blocks.split(",")]
    netD = NetD(in_ch, base_ch, num_blocks)  # resnet-10
    netG = NetG(in_ch, base_ch, latent_dim, num_blocks[-1::-1])
    print("success!")

    # send models to device
    netD.to(device)
    netG.to(device)

    # build exponential moving average manager
    use_ema = args.use_ema
    if use_ema:
        ema_decay = args.ema_decay
        print(f"Creating EMA manager (decay = {ema_decay})...", end="")
        ema = EMA(netG, decay=ema_decay)
        print("success!")
    else:
        ema = nullcontext()

    # build optimizers
    # apply Two Time-scale Update Rule (TTUR) [2]
    # [2] Heusel, Martin, et al. "Gans trained by a two time-scale update rule converge to a local nash equilibrium."
    # Advances in neural information processing systems 30 (2017).
    ttur = args.ttur
    optD = Adam(netD.parameters(), lr=ttur * lr, betas=(beta1, beta2))
    optG = Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    loss_type = args.loss
    if loss_type == "vanilla":
        # binary cross entropy loss (raw logits as inputs) from vanilla gan
        loss_fn_D = nn.BCEWithLogitsLoss(reduction="mean")
        loss_fn_G = loss_fn_D
    elif loss_type == "hinge":
        # hinge loss proposed by [1]
        # [1] Lim, Jae Hyun, and Jong Chul Ye. "Geometric gan." arXiv preprint arXiv:1705.02894 (2017).
        loss_fn_D = lambda x, y: torch.clamp(
            1 - torch.where(y == 1, x, -x), min=0).mean()
        loss_fn_G = lambda x, _: x.mean().neg()
        netD.out_fc[-1].bias = None  # disable discriminator output bias for hinge loss
    else:
        raise NotImplementedError(loss_type)
    print(f"Loss type: {loss_type}")

    # total epochs to train
    total_epochs = args.epochs
    print(f"Total epochs: {total_epochs}")

    # frequency of discriminator steps relative to generator steps
    # i.e. number of discriminator steps per generator step
    dis_freq = args.d_freq
    print(f"D_freq : G_freq = {dis_freq} : 1")

    # max number of discriminator updates per mini-batch
    # not to be confused with number of discriminator updates per generator update
    dis_iters = args.d_iters
    dis_minloss = args.d_minloss or -999
    print(f"Maximum D inner iterations: {dis_iters}\nMinimum D loss: {dis_minloss}")

    # fake samples (in terms of batches) to generate in generator step
    gen_batches = args.g_batches
    gen_samps = batch_size * gen_batches
    print(f"Fake samples for G step: {gen_samps} (={gen_batches} batches)")

    # fixed latent code for image generation over training period
    torch.manual_seed(args.seed)
    nimgs = args.nimgs
    fixed_noise = torch.randn((nimgs, latent_dim))

    chkpt_intv = args.chkpt_intv  # save a checkpoint every 5 epochs
    chkpt_dir = args.chkpt_dir
    chkpt_path = os.path.join(chkpt_dir, "anime-sngan.pt")
    if not os.path.exists(chkpt_dir):
        os.makedirs(chkpt_dir)
    print(f"Checkpoint will be saved to {os.path.abspath(chkpt_path)}", end=" ")
    print(f"every {chkpt_intv} epochs")

    img_dir = args.img_dir
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    print(f"Generated images (x{nimgs}) will be saved to {os.path.abspath(img_dir)}")

    # enable automatic mixed precision on compatible cuda device
    use_amp = args.use_amp
    scalerG = amp.GradScaler(enabled=use_amp)
    scalerD = amp.GradScaler(enabled=use_amp)
    print(f"Automatic mixed precision is {'enabled' if use_amp else 'disabled'}!")

    start_epoch = 0
    if args.resume:
        print("Resuming from checkpoint...", end="")
        chkpt = torch.load(chkpt_path, map_location=device)
        netD.load_state_dict(chkpt["netD"])
        netG.load_state_dict(chkpt["netG"])
        optD.load_state_dict(chkpt["optD"])
        optG.load_state_dict(chkpt["optG"])
        try:
            scalerD.load_state_dict(chkpt["scalerD"])
            scalerG.load_state_dict(chkpt["scalerG"])
            ema.__dict__.update(chkpt["ema"])
        except KeyError:
            pass
        start_epoch = chkpt["epoch"]
        del chkpt
        print("success!")


    def train_step(i, x_true, use_amp):
        # ==================
        # Discriminator step
        # ==================
        cnt = x_true.shape[0]

        # inner loop
        temp_dis_loss = 0
        j = 0
        while j < dis_iters:
            netD.zero_grad(set_to_none=True)
            with amp.autocast(enabled=use_amp):
                dis_loss = loss_fn_D(
                    netD(x_true.to(device)), torch.ones((cnt, 1), device=device))
                with torch.no_grad():
                    x_fake = netG.sample(cnt)
                dis_loss += loss_fn_D(
                    netD(x_fake), torch.zeros((cnt, 1), device=device))
            scalerD.scale(dis_loss).backward()
            scalerD.step(optD)
            scalerD.update()
            netD.zero_grad(set_to_none=True)
            temp_dis_loss += dis_loss.item()
            j += 1
            if dis_loss.item() < dis_minloss:
                break
        temp_dis_loss /= j

        # ==============
        # Generator step
        # ==============
        if i % dis_freq == 0:
            temp_gen_loss = 0
            for _ in range(gen_batches):
                with amp.autocast(enabled=use_amp):
                    x_fake = netG.sample(cnt)
                    gen_loss = loss_fn_G(
                        netD(x_fake), torch.ones((cnt * gen_batches, 1), device=device)
                    ) / gen_batches
                    scalerG.scale(gen_loss).backward()
                    temp_gen_loss += gen_loss.item()
            scalerG.step(optG)
            scalerG.update()
            if use_ema:
                ema.update()
            netG.zero_grad(set_to_none=True)
            return temp_dis_loss, temp_gen_loss, cnt
        else:
            return temp_dis_loss, 0, cnt

    # automatically select maximum batch size
    auto_bsz = args.auto_bsz
    if auto_bsz:
        print("Automatically selecting the optimal batch size...", flush=True)
        # temporarily disable cudnn benchmark is disabled for tuning purpose
        torch.backends.cudnn.benchmark = False
        # override both dis_iter and dis_freq for tuning purpose
        dis_iters_, dis_freq_ = dis_iters, dis_freq
        dis_iters = dis_freq = 1
        bsztuner = BatchSizeTuner(trainloader, device=device, start_bsz=batch_size)
        for x in bsztuner:
            passed = True
            try:
                train_step(0, x, False)  # disable amp by default
            except RuntimeError as e:
                passed = False
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            finally:
                bsztuner.update(passed)
        batch_size = bsztuner.max_bsz or batch_size
        if batch_size != trainloader.batch_size:
            # create a new trainloader with updated batch size
            trainloader = DataLoader(
                traindata, shuffle=True, batch_size=batch_size, drop_last=True, pin_memory=True,
                num_workers=os.cpu_count())
        # back to normal setting
        dis_iters, dis_freq = dis_iters_, dis_freq_
        torch.backends.cudnn.benchmark = cudnnbm
        del bsztuner

    print("Training starts...", flush=True)
    for e in range(start_epoch, total_epochs):
        total_dis_loss = 0
        total_gen_loss = 0
        total_dis_cnt = 0
        total_gen_cnt = 0
        with tqdm(trainloader, desc=f"{e + 1}/{total_epochs} epochs") as t:
            netG.train()
            for i, x_true in enumerate(t):
                # perform single train step
                dis_loss, gen_loss, cnt = train_step(i, x_true, use_amp)

                # update running statistics
                total_dis_loss += dis_loss * cnt
                total_dis_cnt += cnt
                total_gen_loss += gen_loss * cnt * gen_batches
                total_gen_cnt += cnt * gen_batches

                # set progress bar postfix information
                t.set_postfix({
                    "dis_loss": total_dis_loss / total_dis_cnt,
                    "gen_loss": total_gen_loss / total_gen_cnt
                })

                # generate images from a fixed noise at the end of every epoch
                if i == len(trainloader) - 1:
                    netG.eval()
                    with torch.no_grad():
                        with ema:
                            gen_imgs = netG.sample(nimgs, fixed_noise).cpu()
                    gen_imgs = make_grid(
                        gen_imgs, nrow=math.floor(math.sqrt(nimgs)), normalize=True,
                        value_range=(-1, 1)).numpy().transpose(1, 2, 0)
                    plt.imsave(os.path.join(img_dir, f"anime-face-{e + 1}.jpg"), gen_imgs)

                    # save checkpoint
                    _ema = ema.state_dict() if use_ema else dict()
                    if (e + 1) % chkpt_intv == 0:
                        torch.save(
                            {
                                "netD": netD.state_dict(),
                                "netG": netG.state_dict(),
                                "optD": optD.state_dict(),
                                "optG": optG.state_dict(),
                                "scalerD": scalerD.state_dict(),
                                "scalerG": scalerG.state_dict(),
                                "ema": _ema,
                                "epoch": e + 1
                            }, chkpt_path)
