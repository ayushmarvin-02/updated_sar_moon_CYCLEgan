"""Unpaired SAR -> optical CycleGAN training.

Domain A = SAR tiles (selected channels, e.g. vv,vh)   Domain B = optical tiles.
G_AB: SAR -> optical     G_BA: optical -> SAR
"""
import argparse
import csv
import itertools
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UnpairedDataset, parse_channels
from losses import GANLoss, ImagePool, edge_consistency, polarity_hinge_loss
from models import PatchDiscriminator, ResnetGenerator, init_weights
from utils import save_grid, set_seed, to_img


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder made by prepare_data.py")
    p.add_argument("--out", default="runs/dualpol")
    p.add_argument("--channels", default="vv,vh",
                   help="SAR input channels: any of vv,vh,ratio (e.g. 'vv' for the single-pol ablation)")
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100, help="epochs at constant LR")
    p.add_argument("--epochs_decay", type=int, default=100, help="epochs of linear LR decay")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--ngf", type=int, default=64)
    p.add_argument("--ndf", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=9)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--lambda_edge", type=float, default=0.0,
                   help="experimental SAR/optical edge-consistency weight (0 = off)")
    p.add_argument("--lambda_pol", type=float, default=0.0,
                   help="anti-inversion polarity hinge weight on primary SAR channel vs optical (0 = off)")
    p.add_argument("--pool_size", type=int, default=50)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--sample_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(args.out, "samples"), exist_ok=True)

    ds = UnpairedDataset(args.data, "train", args.channels, args.crop, augment=True)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=dev.type == "cuda")
    in_ch, out_ch = ds.sar_channels, ds.opt_channels
    print(f"device={dev}  tiles={len(ds)}  SAR channels={parse_channels(args.channels)}  "
          f"optical channels={out_ch}")

    G_AB = init_weights(ResnetGenerator(in_ch, out_ch, args.ngf, args.n_blocks)).to(dev)
    G_BA = init_weights(ResnetGenerator(out_ch, in_ch, args.ngf, args.n_blocks)).to(dev)
    D_A = init_weights(PatchDiscriminator(in_ch, args.ndf)).to(dev)
    D_B = init_weights(PatchDiscriminator(out_ch, args.ndf)).to(dev)

    opt_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(itertools.chain(D_A.parameters(), D_B.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    total = args.epochs + args.epochs_decay

    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - args.epochs) / float(args.epochs_decay + 1)

    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        for net, k in ((G_AB, "G_AB"), (G_BA, "G_BA"), (D_A, "D_A"), (D_B, "D_B")):
            net.load_state_dict(ck[k])
        opt_G.load_state_dict(ck["opt_G"])
        opt_D.load_state_dict(ck["opt_D"])
        start = ck["epoch"] + 1
        for _ in range(start):
            sched_G.step()
            sched_D.step()
        print(f"resumed from epoch {start}")

    gan, l1 = GANLoss().to(dev), nn.L1Loss()
    pool_A, pool_B = ImagePool(args.pool_size), ImagePool(args.pool_size)
    cfg = {k: (v if not isinstance(v, (list, tuple)) else list(v)) for k, v in vars(args).items()}
    cfg.update(in_ch=in_ch, out_ch=out_ch)

    log_path = os.path.join(args.out, "losses.csv")
    new_log = not os.path.exists(log_path)
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["epoch", "G_gan", "G_cycle", "G_edge", "D_A", "D_B", "seconds"])

    for epoch in range(start, total):
        t0 = time.time()
        sums = dict(g_gan=0., g_cyc=0., g_edge=0., d_a=0., d_b=0.)
        for i, batch in enumerate(loader):
            real_A, real_B = batch["A"].to(dev), batch["B"].to(dev)

            # ---- generators ----
            for p in itertools.chain(D_A.parameters(), D_B.parameters()):
                p.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            fake_B = G_AB(real_A)
            rec_A = G_BA(fake_B)
            fake_A = G_BA(real_B)
            rec_B = G_AB(fake_A)
            loss_gan = gan(D_B(fake_B), True) + gan(D_A(fake_A), True)
            loss_cyc = (l1(rec_A, real_A) + l1(rec_B, real_B)) * args.lambda_cycle
            loss_edge = (edge_consistency(real_A, fake_B) * args.lambda_edge
                         if args.lambda_edge > 0 else torch.zeros((), device=dev))
            loss_pol = (polarity_hinge_loss(real_A, fake_B) * args.lambda_pol
                        if args.lambda_pol > 0 else torch.zeros((), device=dev))
            (loss_gan + loss_cyc + loss_edge + loss_pol).backward()
            opt_G.step()

            # ---- discriminators ----
            for p in itertools.chain(D_A.parameters(), D_B.parameters()):
                p.requires_grad_(True)
            opt_D.zero_grad(set_to_none=True)
            fb = pool_B.query(fake_B)
            fa = pool_A.query(fake_A)
            loss_DB = 0.5 * (gan(D_B(real_B), True) + gan(D_B(fb.detach()), False))
            loss_DA = 0.5 * (gan(D_A(real_A), True) + gan(D_A(fa.detach()), False))
            (loss_DA + loss_DB).backward()
            opt_D.step()

            sums["g_gan"] += loss_gan.item()
            sums["g_cyc"] += loss_cyc.item()
            sums["g_edge"] += loss_edge.item()
            sums["d_a"] += loss_DA.item()
            sums["d_b"] += loss_DB.item()
            if (i + 1) % args.log_every == 0:
                print(f"epoch {epoch + 1}/{total} iter {i + 1}/{len(loader)}  "
                      f"G_gan {loss_gan.item():.3f}  cyc {loss_cyc.item():.3f}  "
                      f"D_A {loss_DA.item():.3f}  D_B {loss_DB.item():.3f}", flush=True)

        n = max(1, len(loader))
        log.writerow([epoch + 1] + [round(sums[k] / n, 5) for k in
                                    ("g_gan", "g_cyc", "g_edge", "d_a", "d_b")] +
                     [round(time.time() - t0, 1)])
        log_f.flush()
        sched_G.step()
        sched_D.step()

        if (epoch + 1) % args.sample_every == 0:
            k = min(4, real_A.shape[0])
            rows = [[to_img(real_A[j], True), to_img(fake_B[j]), to_img(real_B[j]),
                     to_img(rec_A[j], True)] for j in range(k)]
            save_grid(rows, os.path.join(args.out, "samples", f"epoch_{epoch + 1:03d}.png"))

        ck = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(),
              "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
              "opt_G": opt_G.state_dict(), "opt_D": opt_D.state_dict(),
              "epoch": epoch, "args": cfg}
        torch.save(ck, os.path.join(args.out, "last.pt"))
        if (epoch + 1) % args.save_every == 0:
            torch.save(ck, os.path.join(args.out, f"epoch_{epoch + 1:03d}.pt"))
        print(f"epoch {epoch + 1}/{total} done in {time.time() - t0:.0f}s", flush=True)

    log_f.close()


if __name__ == "__main__":
    main()
