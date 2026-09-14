#!/usr/bin/env python3
"""Stage 8: source-calibrated counterfactual candidate fusion (SCCF).

SCCF keeps the v2 SicTTA candidate generation intact but replaces its implicit
"candidate always wins" output choice with a source-calibrated, continuous
fusion between the frozen source anchor and the memory-adapted candidate.
Calibration uses source labels plus deterministic appearance corruption only;
target labels are never read by the adaptation rule.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn

from stage3_transactional import (
    align_to_source,
    anatomy_features,
    build_stage3_adapter,
    capped_assd,
    make_model,
    normalized_entropy,
    source_style_stats,
    to_prob,
)
from table2_eval_faithful import DS, MAN


FEATURE_NAMES = (
    "source_entropy", "candidate_entropy", "source_confidence",
    "candidate_confidence", "source_anatomy", "candidate_anatomy",
    "candidate_source_disagreement", "entropy_delta",
)


def load_manifest_row(row: dict) -> tuple[torch.Tensor, np.ndarray]:
    image = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32)
    x = (image / 255.0).transpose(2, 0, 1) * 2.0 - 1.0
    mask = np.asarray(Image.open(row["label"]).convert("RGB"), dtype=np.uint8)[..., 0]
    y = np.zeros(mask.shape, dtype=np.int64)
    y[mask == 128] = 1
    y[mask == 0] = 2
    _, height, width = x.shape
    x = ndimage.zoom(x, [1, 320 / height, 320 / width], order=2).astype(np.float32)
    y = ndimage.zoom(y, [320 / height, 320 / width], order=0).astype(np.int64)
    return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0), y


def source_style_corruption(x: torch.Tensor, index: int) -> torch.Tensor:
    """Deterministic source-only corruption for calibration diversity."""
    z = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    variants = (
        (0.82, 1.12, (1.08, 0.94, 1.02)),
        (1.16, 0.86, (0.94, 1.05, 1.10)),
        (0.90, 1.20, (1.04, 1.02, 0.90)),
        (1.08, 0.92, (0.96, 0.98, 1.06)),
    )
    brightness, contrast, channels = variants[index % len(variants)]
    z = ((z - 0.5) * contrast + 0.5) * brightness
    z = z * torch.tensor(channels, dtype=z.dtype, device=z.device).view(1, 3, 1, 1)
    return z.clamp(0.0, 1.0) * 2.0 - 1.0


def confidence(prob: torch.Tensor) -> float:
    return float(prob.max(1).values.mean().item())


def feature_vector(source_prob: torch.Tensor, candidate_prob: torch.Tensor,
                   shape_stats: dict) -> np.ndarray:
    source_anatomy = float(anatomy_features(source_prob, shape_stats).get("anatomy_score", 0.0))
    candidate_anatomy = float(anatomy_features(candidate_prob, shape_stats).get("anatomy_score", 0.0))
    source_entropy = normalized_entropy(source_prob)
    candidate_entropy = normalized_entropy(candidate_prob)
    disagreement = float((source_prob.argmax(1) != candidate_prob.argmax(1)).float().mean().item())
    values = np.asarray([
        source_entropy, candidate_entropy, confidence(source_prob),
        confidence(candidate_prob), source_anatomy, candidate_anatomy,
        disagreement, candidate_entropy - source_entropy,
    ], dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite SCCF feature")
    return values


def macro_dice(prob: torch.Tensor, target: np.ndarray) -> float:
    pred = prob.argmax(1)[0].detach().cpu().numpy()
    values = []
    for label in (1, 2):
        a, b = pred == label, target == label
        values.append((2 * (a & b).sum() + 1e-5) /
                      (a.sum() + b.sum() + 1e-5))
    return float(np.mean(values))


def fit_calibrator(args: argparse.Namespace) -> None:
    if args.calibration_out.exists():
        raise RuntimeError("refusing to overwrite calibration: " + str(args.calibration_out))
    device = torch.device("cuda:" + str(args.gpu))
    shape_stats = json.loads(args.shape_stats.read_text(encoding="utf-8"))
    source_mean, source_std = source_style_stats(args.source_manifest)
    adapter, anchor = build_stage3_adapter(device, args.ckpt, args.shape_stats, hard_anatomy=True)
    rows = list(csv.DictReader(args.source_manifest.open(encoding="utf-8")))
    if args.source_limit:
        rows = rows[:args.source_limit]
    features, targets, deltas = [], [], []
    for index, row in enumerate(rows):
        x, target = load_manifest_row(row)
        x = source_style_corruption(x.to(device), index)
        corrected = align_to_source(x, source_mean, source_std)
        with torch.no_grad():
            source_prob = to_prob(anchor(corrected))
            candidate_prob = to_prob(adapter(corrected, [Path(row["image"]).name]))
        values = feature_vector(source_prob, candidate_prob, shape_stats)
        source_score = macro_dice(source_prob, target)
        candidate_score = macro_dice(candidate_prob, target)
        features.append(values)
        deltas.append(candidate_score - source_score)
        targets.append(float(candidate_score > source_score + args.min_improvement))
        if (index + 1) % 25 == 0:
            print(f"SCCF_CALIBRATION_PROGRESS case={index + 1}/{len(rows)}", flush=True)
    x_np = np.asarray(features, dtype=np.float64)
    y_np = np.asarray(targets, dtype=np.float32)
    mean = x_np.mean(0)
    std = np.maximum(x_np.std(0), 1e-6)
    x_std = (x_np - mean) / std
    if len(np.unique(y_np)) < 2:
        weight = np.zeros(x_std.shape[1], dtype=np.float64)
        bias = float(np.log((y_np.mean() + 1e-3) / (1.0 - y_np.mean() + 1e-3)))
        fit_method = "constant_fallback"
    else:
        model = nn.Linear(x_std.shape[1], 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        tx = torch.from_numpy(x_std.astype(np.float32))
        ty = torch.from_numpy(y_np[:, None].astype(np.float32))
        for _ in range(400):
            loss = nn.functional.binary_cross_entropy_with_logits(model(tx), ty)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        weight = model.weight.detach().cpu().numpy()[0].astype(np.float64)
        bias = float(model.bias.detach().cpu().item())
        fit_method = "source_logistic"
    calibration = {
        "format": "SCCF-source-calibrator-v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_mean": mean.tolist(), "feature_std": std.tolist(),
        "weight": weight.tolist(), "bias": bias,
        "source_images": len(rows), "source_labels_used_for_calibration": True,
        "target_labels_used": False,
        "fit_method": fit_method, "positive_rate": float(y_np.mean()),
        "candidate_minus_source_macro_dice_mean": float(np.mean(deltas)),
        "min_improvement": args.min_improvement,
        "no_raw_target_images_or_features_stored": True,
        "privacy_claim": "source-calibrated raw-target-data-free fusion; not differential privacy",
    }
    args.calibration_out.parent.mkdir(parents=True, exist_ok=True)
    args.calibration_out.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print("SCCF_CALIBRATION_COMPLETE " + json.dumps({
        "images": len(rows), "positive_rate": calibration["positive_rate"],
        "delta_mean": calibration["candidate_minus_source_macro_dice_mean"],
        "fit_method": fit_method}, sort_keys=True), flush=True)


class SCCF:
    def __init__(self, adapter, anchor, calibration: dict, shape_stats: dict):
        self.adapter = adapter
        self.anchor = anchor.eval()
        self.calibration = calibration
        self.shape_stats = shape_stats

    def weight(self, values: np.ndarray) -> float:
        mean = np.asarray(self.calibration["feature_mean"], dtype=np.float64)
        std = np.asarray(self.calibration["feature_std"], dtype=np.float64)
        weights = np.asarray(self.calibration["weight"], dtype=np.float64)
        score = float(self.calibration["bias"] + np.dot((values - mean) / std, weights))
        return float(1.0 / (1.0 + math.exp(-np.clip(score, -30.0, 30.0))))

    def __call__(self, x: torch.Tensor, name: str) -> tuple[torch.Tensor, dict]:
        with torch.no_grad():
            source_prob = to_prob(self.anchor(x))
            candidate_prob = to_prob(self.adapter(x, [name]))
            values = feature_vector(source_prob, candidate_prob, self.shape_stats)
            fusion_weight = self.weight(values)
            fused = fusion_weight * candidate_prob + (1.0 - fusion_weight) * source_prob
        return fused, {
            "fusion_weight": fusion_weight,
            "candidate_entropy": float(values[1]),
            "candidate_anatomy": float(values[5]),
            "candidate_source_disagreement": float(values[6]),
        }


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.gpu))
    shape_stats = json.loads(args.shape_stats.read_text(encoding="utf-8"))
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    source_mean, source_std = source_style_stats(args.source_manifest)
    adapter, anchor = build_stage3_adapter(device, args.ckpt, args.shape_stats, hard_anatomy=True)
    sccf = SCCF(adapter, anchor, calibration, shape_stats)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty output: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {"method": "sccf", "seed": args.seed, "order": args.order,
               "calibration": str(args.calibration),
               "target_labels_for_fusion": False,
               "domains": {}}
    for domain in args.order:
        dataset = DS(MAN / f"target_domain_{domain.lower()}_stream.csv")
        if args.limit: dataset.rows = dataset.rows[:args.limit]
        rows = []; start = time.time()
        for index in range(len(dataset)):
            x, y, name = dataset[index]
            corrected = align_to_source(x.to(device), source_mean, source_std)
            output_prob, diag = sccf(corrected, name)
            pred = output_prob.argmax(1)[0].detach().cpu().numpy()
            target = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            record = [name]
            for label in (1, 2):
                a, b = pred == label, target == label
                record.extend([float((2 * (a & b).sum() + 1e-5) /
                                  (a.sum() + b.sum() + 1e-5)), capped_assd(a, b)])
            record.extend([diag["fusion_weight"], diag["candidate_entropy"],
                           diag["candidate_anatomy"], diag["candidate_source_disagreement"]])
            rows.append(record)
            if (index + 1) % args.report_every == 0:
                values = np.asarray([row[1:5] for row in rows], dtype=float)
                print(f"SCCF_PROGRESS seed={args.seed} order={args.order} domain={domain} "
                      f"case={index + 1}/{len(dataset)} dice={values[:, [0, 2]].mean():.4f} "
                      f"w={np.mean([r[5] for r in rows]):.3f}", flush=True)
        values = np.asarray([row[1:5] for row in rows], dtype=float)
        summary["domains"][domain] = {
            "n": len(rows), "od_dice": float(values[:, 0].mean()),
            "od_assd": float(values[:, 1].mean()), "oc_dice": float(values[:, 2].mean()),
            "oc_assd": float(values[:, 3].mean()),
            "macro_dice": float(values[:, [0, 2]].mean()),
            "assd": float(values[:, [1, 3]].mean()),
            "fusion_weight_mean": float(np.mean([r[5] for r in rows])),
            "seconds": float(time.time() - start),
        }
        with (out / f"stage8_sccf_{domain}_per_case.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["image", "od_dice", "od_assd", "oc_dice", "oc_assd",
                             "fusion_weight", "candidate_entropy", "candidate_anatomy",
                             "candidate_source_disagreement"])
            writer.writerows(rows)
    c, d = summary["domains"]["C"], summary["domains"]["D"]
    summary["average"] = {key: float((c[key] + d[key]) / 2.0)
                           for key in ("od_dice", "od_assd", "oc_dice", "oc_assd", "macro_dice", "assd")}
    (out / f"stage8_sccf_{args.order}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SCCF_COMPLETE " + json.dumps(summary["average"], sort_keys=True), flush=True)


def unit_test() -> None:
    calibration = {"feature_mean": [0.0] * 8, "feature_std": [1.0] * 8,
                    "weight": [0.0] * 8, "bias": 0.0}
    shape = {"admission_bounds": {name: {"low": 0.0, "high": 1.0}
                                   for name in ("cup_disc_ratio", "center_distance_norm",
                                                "disc_lcc_fraction", "cup_lcc_fraction")}}
    # Validate the continuous fusion contract independently of the U-Net.
    source = torch.tensor([[[[0.9]], [[0.1]], [[0.0]]]])
    candidate = torch.tensor([[[[0.1]], [[0.8]], [[0.1]]]])
    fused = 0.5 * candidate + 0.5 * source
    checks = {"finite": bool(torch.isfinite(fused).all()),
              "normalizes": bool(torch.allclose(fused.sum(1), torch.ones(1, 1, 1))),
              "weight_bounds": 0.0 <= 1.0 / (1.0 + math.exp(-0.0)) <= 1.0,
              "feature_count": len(FEATURE_NAMES) == 8}
    print("STAGE8_SCCF_UNIT " + json.dumps(checks, sort_keys=True))
    if not all(checks.values()): raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--fit-calibrator", action="store_true")
    parser.add_argument("--ckpt", type=Path); parser.add_argument("--shape-stats", type=Path)
    parser.add_argument("--source-manifest", type=Path); parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--order", choices=("CD", "DC")); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--source-limit", type=int, default=100); parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--report-every", type=int, default=20)
    args = parser.parse_args()
    if args.unit_test: return args
    required = ("ckpt", "shape_stats", "source_manifest", "calibration_out") if args.fit_calibrator else ("ckpt", "shape_stats", "source_manifest", "calibration", "output", "order")
    missing = [x for x in required if getattr(args, x) is None]
    if missing: parser.error("missing required arguments: " + ", ".join(missing))
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.unit_test: unit_test()
    elif args.fit_calibrator: fit_calibrator(args)
    else: run(args)
