from __future__ import annotations

import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from mps.utils import set_seed, AverageMeter, accuracy_top1, save_json
from mps.supernet import convert_to_supernet, ChoiceConfig
from mps.aihwkit_backend import AnalogConfig
from mps.pareto import Obj, pareto_ranks

from models import resnet8_cifar


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "resnet8":
        return resnet8_cifar(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def cifar10_loaders(data_dir: str, batch_size: int, num_workers: int, val_frac: float = 0.1):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_test = transforms.Compose([transforms.ToTensor()])

    ds_train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf_train)
    ds_test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_test)

    n_val = int(len(ds_train_full) * val_frac)
    n_train = len(ds_train_full) - n_val
    ds_train, ds_val = random_split(ds_train_full, [n_train, n_val])

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def fairness_sample_opnames(choice) -> str:
    # Uniform across available operators to ensure fairness
    return random.choice(choice.op_names)


@torch.no_grad()
def eval_one_epoch(model: nn.Module, loader: DataLoader, device: str, amp: bool = True) -> float:
    model.eval()
    accm = AverageMeter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(amp and "cuda" in device)):
            logits = model(x)
        accm.update(accuracy_top1(logits, y), n=x.size(0))
    return accm.avg


def train_one_epoch_supernet(
    supernet,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer,
    amp: bool = True,
    sampled_mapping: Dict[str, str] | None = None,
):
    supernet.train()
    supernet.set_all_sampled(sampled_mapping)

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and "cuda" in device))

    lossm = AverageMeter()
    accm = AverageMeter()

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and "cuda" in device)):
            logits = supernet(x)
            loss = loss_fn(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        lossm.update(loss.item(), n=x.size(0))
        accm.update(accuracy_top1(logits.detach(), y), n=x.size(0))

    return lossm.avg, accm.avg


def progressive_noise_update(supernet, base_noise: float, max_noise: float, step: float):
    # Simple progressive schedule: increase analog out_noise for all analog-capable choices
    # (You can make this per-layer using sensitivity metrics if desired)
    for _, c in supernet.named_choices():
        if "analog" in c.op_names:
            current = getattr(c, "_cur_noise", base_noise)
            new = min(max_noise, current + step)
            c._cur_noise = new
            c.set_analog_out_noise(new)


def sample_subnet_mapping(supernet) -> Dict[str, str]:
    mapping = {}
    for name, c in supernet.named_choices():
        mapping[name] = fairness_sample_opnames(c)
    return mapping


def estimate_objectives(supernet, val_loader, device: str, mapping: Dict[str, str]) -> Obj:
    # Accuracy estimate on validation subset
    supernet.set_all_sampled(mapping)
    acc = eval_one_epoch(supernet, val_loader, device=device, amp=True)

    # Hardware proxies:
    # analog_macs = fraction of layers mapped to analog (proxy for analog MAC ratio)
    # dig_w_mb = proxy = count digital selections * constant weight size proxy
    total = 0
    analog = 0
    dig = 0
    for _, c in supernet.named_choices():
        total += 1
        sel = mapping[c.name]
        if sel == "analog":
            analog += 1
        if sel in ("fp16", "int8", "skip"):
            dig += 1
    analog_ratio = analog / max(1, total)
    dig_w_mb = float(dig)  # proxy; replace with parameter/MiB calculation if needed
    return Obj(acc=acc, analog_macs=analog_ratio, dig_w_mb=dig_w_mb)


def rank_preserving_alpha_step(supernet, val_loader, device: str, n_samples: int, lr_alpha: float, tau: float):
    # Phase 3: freeze weights; update only alphas to preserve Pareto ranks
    for p in supernet.parameters():
        p.requires_grad = False
    for _, c in supernet.named_choices():
        c.alpha.requires_grad = True

    opt = torch.optim.Adam([c.alpha for _, c in supernet.named_choices()], lr=lr_alpha)

    # Sample subnetworks and compute ranks
    mappings = [sample_subnet_mapping(supernet) for _ in range(n_samples)]
    objs = [estimate_objectives(supernet, val_loader, device, m) for m in mappings]
    ranks = pareto_ranks(objs)

    # Build pairwise ranking loss over gumbel-relaxed scores
    # We approximate "better" subnets (lower rank) should have higher log-prob under alphas.
    supernet.set_all_sampled(None)
    supernet.set_gumbel_all(True, tau=tau)

    # compute log-prob of each mapping under current alphas (product of per-layer probs)
    def logprob(mapping):
        lp = 0.0
        for _, c in supernet.named_choices():
            probs = torch.softmax(c.alpha, dim=0)
            idx = c.op_names.index(mapping[c.name])
            lp = lp + torch.log(probs[idx] + 1e-12)
        return lp

    # Pairwise hinge: if i better than j => lp_i >= lp_j + margin
    margin = 0.1
    loss = 0.0
    count = 0
    for i in range(n_samples):
        for j in range(n_samples):
            if ranks[i] < ranks[j]:
                li = logprob(mappings[i])
                lj = logprob(mappings[j])
                loss = loss + torch.relu(margin - (li - lj))
                count += 1
    loss = loss / max(1, count)

    opt.zero_grad()
    loss.backward()
    opt.step()

    supernet.set_gumbel_all(False, tau=tau)

    # restore requires_grad for later if needed
    for p in supernet.parameters():
        p.requires_grad = True

    return float(loss.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet8")
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--out-dir", default="./runs/mps")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)

    ap.add_argument("--epochs-phase1", type=int, default=10)
    ap.add_argument("--epochs-phase2", type=int, default=10)
    ap.add_argument("--epochs-phase3", type=int, default=5)

    # progressive analog noise scaling
    ap.add_argument("--analog-noise-base", type=float, default=0.0)
    ap.add_argument("--analog-noise-max", type=float, default=0.05)
    ap.add_argument("--analog-noise-step", type=float, default=0.005)

    # phase3
    ap.add_argument("--phase3-samples", type=int, default=16)
    ap.add_argument("--alpha-lr", type=float, default=3e-3)
    ap.add_argument("--gumbel-tau", type=float, default=1.0)

    args = ap.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    train_loader, val_loader, test_loader = cifar10_loaders(
        args.data_dir, args.batch_size, args.num_workers
    )

    model = get_model(args.model).to(args.device)

    choice_cfg = ChoiceConfig(enable_fp16=True, enable_int8=True, enable_analog=True, enable_skip=True)
    analog_cfg = AnalogConfig(out_noise=args.analog_noise_base, is_cuda=("cuda" in args.device))

    supernet = convert_to_supernet(
        model=model,
        choice_cfg=choice_cfg,
        analog_cfg=analog_cfg,
        device=args.device,
        input_shape=(1, 3, 32, 32),
    ).to(args.device)

    optimizer = torch.optim.SGD(
        supernet.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    # -----------------------
    # Phase 1: Fairness train
    # -----------------------
    for epoch in range(args.epochs_phase1):
        mapping = sample_subnet_mapping(supernet)
        loss, acc = train_one_epoch_supernet(supernet, train_loader, args.device, optimizer, amp=True, sampled_mapping=mapping)
        val_acc = eval_one_epoch(supernet, val_loader, args.device, amp=True)
        print(f"[P1][{epoch+1}/{args.epochs_phase1}] loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")

    # -----------------------------------------
    # Phase 2: Progressive analog noise + QAT
    # -----------------------------------------
    for epoch in range(args.epochs_phase2):
        progressive_noise_update(supernet, args.analog_noise_base, args.analog_noise_max, args.analog_noise_step)
        mapping = sample_subnet_mapping(supernet)
        loss, acc = train_one_epoch_supernet(supernet, train_loader, args.device, optimizer, amp=True, sampled_mapping=mapping)
        val_acc = eval_one_epoch(supernet, val_loader, args.device, amp=True)
        print(f"[P2][{epoch+1}/{args.epochs_phase2}] loss={loss:.4f} train_acc={acc:.4f} val_acc={val_acc:.4f}")

    # -----------------------------------------
    # Phase 3: Rank-preserving alpha fine-tune
    # -----------------------------------------
    for epoch in range(args.epochs_phase3):
        rp_loss = rank_preserving_alpha_step(
            supernet, val_loader, args.device,
            n_samples=args.phase3_samples, lr_alpha=args.alpha_lr, tau=args.gumbel_tau
        )
        mapping = supernet.extract_mapping()
        supernet.set_all_sampled(mapping)
        val_acc = eval_one_epoch(supernet, val_loader, args.device, amp=True)
        print(f"[P3][{epoch+1}/{args.epochs_phase3}] rank_loss={rp_loss:.4f} val_acc={val_acc:.4f}")

    # Extract mapping and final test
    mapping = supernet.extract_mapping()
    save_json(mapping, os.path.join(args.out_dir, "mapping.json"))
    supernet.set_all_sampled(mapping)
    test_acc = eval_one_epoch(supernet, test_loader, args.device, amp=True)
    print(f"[FINAL] mapping saved. test_acc={test_acc:.4f}")

    ckpt = {
        "model": args.model,
        "state_dict": supernet.state_dict(),
        "mapping": mapping,
        "choice_cfg": choice_cfg.__dict__,
        "analog_cfg": analog_cfg.__dict__,
        "args": vars(args),
    }
    torch.save(ckpt, os.path.join(args.out_dir, "supernet.pt"))
    print(f"Saved checkpoint to {os.path.join(args.out_dir, 'supernet.pt')}")


if __name__ == "__main__":
    main()
