from __future__ import annotations

import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import resnet8_cifar
from mps.supernet import convert_to_supernet, ChoiceConfig
from mps.aihwkit_backend import AnalogConfig
from mps.utils import accuracy_top1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = resnet8_cifar(num_classes=10)

    choice_cfg = ChoiceConfig(**ckpt["choice_cfg"])
    analog_cfg = AnalogConfig(**ckpt["analog_cfg"])

    supernet = convert_to_supernet(
        model=model,
        choice_cfg=choice_cfg,
        analog_cfg=analog_cfg,
        device=args.device,
        input_shape=(1, 3, 32, 32),
    ).to(args.device)

    supernet.load_state_dict(ckpt["state_dict"], strict=True)
    mapping = ckpt["mapping"]
    supernet.set_all_sampled(mapping)
    supernet.eval()

    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(args.device), y.to(args.device)
        logits = supernet(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()

    print(f"Test accuracy: {correct/total:.4f}")
    print("Mapping:")
    for k, v in list(mapping.items())[:30]:
        print(f"  {k}: {v}")
    if len(mapping) > 30:
        print(f"  ... ({len(mapping)} layers total)")


if __name__ == "__main__":
    main()
