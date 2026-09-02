#!/usr/bin/env python3
"""Paper-protocol source training for SicTTA Table 2 Fundus reproduction."""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from robustbench.losses import DiceLoss
from robustbench.seg_net.unet import UNet


ROOT = Path("/home/zhaoruijin/MOURUI/sictta_reproduction_20260827")
MANIFESTS = ROOT / "manifests"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FundusDataset(Dataset):
    def __init__(self, manifest: Path, size: int = 320):
        with manifest.open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.size = size
        for row in self.rows:
            if not Path(row["image"]).is_file() or not Path(row["label"]).is_file():
                raise FileNotFoundError(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
        image = np.transpose(image, (2, 0, 1))
        # Match the public SicTTA Fundus loader.
        image = (image / 255.0) * 2.0 - 1.0

        mask_rgb = np.asarray(Image.open(row["label"]).convert("RGB"), dtype=np.uint8)
        raw = mask_rgb[..., 0]
        unique = set(np.unique(raw).tolist())
        if not unique.issubset({0, 128, 255}):
            raise ValueError(f"Unexpected mask values {unique} in {row['label']}")
        label = np.zeros(raw.shape, dtype=np.int64)
        label[raw == 0] = 2
        label[raw == 128] = 1
        label[raw == 255] = 0
        label = label[None]

        _, h, w = image.shape
        zoom = [1.0, self.size / h, self.size / w]
        image = ndimage.zoom(image, zoom, order=2).astype(np.float32, copy=False)
        label = ndimage.zoom(label, zoom, order=0).astype(np.int64, copy=False)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "label": torch.from_numpy(np.ascontiguousarray(label)),
            "domain": row["domain"],
            "name": Path(row["image"]).name,
        }


def balanced_epoch_indices(rows, seed):
    by_domain = {"A": [], "B": []}
    for idx, row in enumerate(rows):
        by_domain[row["domain"]].append(idx)
    rng = random.Random(seed)
    target = max(len(by_domain["A"]), len(by_domain["B"]))
    result = []
    for domain in ("A", "B"):
        source = list(by_domain[domain])
        expanded = []
        while len(expanded) < target:
            rng.shuffle(source)
            expanded.extend(source)
        result.extend(expanded[:target])
    rng.shuffle(result)
    return result


def build_model(device):
    params = {
        "in_chns": 3,
        "ft_chns": [16, 32, 64, 128, 256],
        "dropout_p": [0, 0, 0.3, 0.4, 0.5],
        "n_classes": 3,
        "bilinear": True,
        "deep_supervise": False,
        "lr": 1e-3,
        "up_mode": "upsample",
    }
    model = UNet(params)
    model.initialize()
    return model.to(device)


def per_image_dice(pred, target, klass):
    pred_k = pred == klass
    target_k = target == klass
    inter = (pred_k & target_k).sum(dim=(-2, -1)).float()
    denom = pred_k.sum(dim=(-2, -1)).float() + target_k.sum(dim=(-2, -1)).float()
    return (2.0 * inter + 1e-5) / (denom + 1e-5)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    values = {"A": {"od": [], "oc": []}, "B": {"od": [], "oc": []}}
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)[:, 0]
        pred = model(image).argmax(1)
        od = per_image_dice(pred, label, 1).cpu().tolist()
        oc = per_image_dice(pred, label, 2).cpu().tolist()
        for i, domain in enumerate(batch["domain"]):
            values[domain]["od"].append(od[i])
            values[domain]["oc"].append(oc[i])
    metrics = {}
    domain_macros = []
    for domain in ("A", "B"):
        od = float(np.mean(values[domain]["od"]))
        oc = float(np.mean(values[domain]["oc"]))
        macro = (od + oc) / 2.0
        metrics[f"val_{domain}_od"] = od
        metrics[f"val_{domain}_oc"] = oc
        metrics[f"val_{domain}_macro"] = macro
        domain_macros.append(macro)
    metrics["val_balanced_macro"] = float(np.mean(domain_macros))
    return metrics


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "source_train_seed1")
    args = parser.parse_args()

    if not str(args.output.resolve()).startswith(str(ROOT.resolve()) + os.sep):
        raise RuntimeError(f"Output must stay inside {ROOT}: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device(f"cuda:{args.gpu}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    train_manifest = MANIFESTS / "source_train.csv"
    valid_manifest = MANIFESTS / "source_valid.csv"
    train_ds = FundusDataset(train_manifest)
    valid_ds = FundusDataset(valid_manifest)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=args.num_workers > 0)

    config = vars(args).copy()
    config["output"] = str(args.output)
    config.update({
        "train_manifest": str(train_manifest),
        "valid_manifest": str(valid_manifest),
        "train_manifest_sha256": sha256(train_manifest),
        "valid_manifest_sha256": sha256(valid_manifest),
        "preprocessing": "RGB to [-1,1], scipy resize 320 order2; mask nearest order0",
        "classes": {"0": "background", "1": "optic_disc", "2": "optic_cup"},
        "optimizer": "Adam(lr=1e-3, betas=(0.9,0.999))",
        "loss": "foreground Dice loss over OD and OC",
        "source_balancing": "exact equal A/B exposure per epoch by oversampling",
    })
    (args.output / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    model = build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
    criterion = DiceLoss(3).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"PREFLIGHT device={device} torch={torch.__version__} params={parameter_count}", flush=True)
    first = train_ds[0]
    print(f"PREFLIGHT_SAMPLE image={tuple(first['image'].shape)} label={tuple(first['label'].shape)} "
          f"range=({first['image'].min().item():.4f},{first['image'].max().item():.4f}) "
          f"labels={torch.unique(first['label']).tolist()} domain={first['domain']}", flush=True)

    start_epoch, best_metric = 0, -1.0
    last_path = args.output / "last.pth"
    if last_path.is_file():
        checkpoint = torch.load(last_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        print(f"RESUME epoch={start_epoch} best={best_metric:.6f}", flush=True)

    metrics_path = args.output / "metrics.jsonl"
    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        indices = balanced_epoch_indices(train_ds.rows, args.seed + epoch)
        train_loader = DataLoader(Subset(train_ds, indices), batch_size=args.batch_size,
                                  shuffle=False, num_workers=args.num_workers,
                                  pin_memory=True, persistent_workers=args.num_workers > 0)
        model.train()
        loss_sum, sample_count = 0.0, 0
        epoch_start = time.time()
        for step, batch in enumerate(train_loader):
            image = batch["image"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = criterion(logits, label, softmax=True, one_hot=True)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}: {loss.item()}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * image.size(0)
            sample_count += image.size(0)
            if step == 0:
                print(f"FIRST_BATCH epoch={epoch+1}/{args.epochs} shape={tuple(image.shape)} "
                      f"loss={loss.item():.6f} gpu_mem_mb={torch.cuda.max_memory_allocated(device)/1048576:.1f}", flush=True)

        val = validate(model, valid_loader, device)
        train_loss = loss_sum / sample_count
        elapsed = time.time() - epoch_start
        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **val,
            "seconds": elapsed,
            "balanced_samples": sample_count,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

        improved = val["val_balanced_macro"] > best_metric
        if improved:
            best_metric = val["val_balanced_macro"]
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": config,
            "record": record,
        }
        atomic_torch_save(state, last_path)
        if improved:
            atomic_torch_save(state, args.output / "best.pth")

        completed = epoch - start_epoch + 1
        eta = (time.time() - run_start) / completed * (args.epochs - epoch - 1)
        print(f"EPOCH {epoch+1:03d}/{args.epochs} loss={train_loss:.6f} "
              f"valA={val['val_A_macro']:.4f} valB={val['val_B_macro']:.4f} "
              f"balanced={val['val_balanced_macro']:.4f} best={best_metric:.4f} "
              f"sec={elapsed:.1f} eta_min={eta/60:.1f}", flush=True)

    print(f"TRAINING_COMPLETE epochs={args.epochs} best={best_metric:.6f} output={args.output}", flush=True)


if __name__ == "__main__":
    main()
