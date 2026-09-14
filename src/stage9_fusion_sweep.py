#!/usr/bin/env python3
"""Stage 9B/C: SCCF fusion and source-calibrator sensitivity sweep."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from stage3_transactional import (
    align_to_source,
    build_stage3_adapter,
    capped_assd,
    source_style_stats,
)
from stage8_sccf import SCCF
from table2_eval_faithful import DS, MAN


def fixed_calibration(base: dict, variant: str) -> dict:
    out = dict(base)
    out["feature_mean"] = [0.0] * 8
    out["feature_std"] = [1.0] * 8
    out["weight"] = [0.0] * 8
    if variant.startswith("fixed"):
        w = float(variant.replace("fixed", "")) / 100.0
        out["bias"] = math.log(w / (1.0 - w))
    elif variant == "entropy":
        out["weight"][1] = -4.0; out["bias"] = 0.0
    elif variant == "anatomy":
        out["weight"][5] = 4.0; out["bias"] = 0.0
    elif variant == "disagreement":
        out["weight"][6] = -4.0; out["bias"] = 0.0
    else:
        return base
    out["variant"] = variant
    return out


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.gpu))
    shape_stats = json.loads(args.shape_stats.read_text(encoding="utf-8"))
    base = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = fixed_calibration(base, args.variant)
    source_mean, source_std = source_style_stats(args.source_manifest)
    adapter, anchor = build_stage3_adapter(device, args.ckpt, args.shape_stats, hard_anatomy=True)
    sccf = SCCF(adapter, anchor, calibration, shape_stats)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty output: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": args.variant, "seed": args.seed, "order": args.order,
        "calibration": str(args.calibration), "target_labels_for_fusion": False,
        "domains": {},
    }
    for domain in args.order:
        dataset = DS(MAN / f"target_domain_{domain.lower()}_stream.csv")
        if args.limit: dataset.rows = dataset.rows[:args.limit]
        rows = []; start = time.time()
        for index in range(len(dataset)):
            x, y, name = dataset[index]
            corrected = align_to_source(x.to(device), source_mean, source_std)
            prob, diag = sccf(corrected, name)
            pred = prob.argmax(1)[0].detach().cpu().numpy()
            target = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            record = [name]
            for label in (1, 2):
                a, b = pred == label, target == label
                record.extend([
                    float((2 * (a & b).sum() + 1e-5) / (a.sum() + b.sum() + 1e-5)),
                    capped_assd(a, b),
                ])
            record.extend([
                diag["fusion_weight"], diag["candidate_entropy"],
                diag["candidate_anatomy"], diag["candidate_source_disagreement"],
            ])
            rows.append(record)
            if (index + 1) % args.report_every == 0:
                values = np.asarray([r[1:5] for r in rows], dtype=float)
                print(
                    f"FUSION_PROGRESS variant={args.variant} seed={args.seed} "
                    f"order={args.order} domain={domain} case={index + 1}/{len(dataset)} "
                    f"dice={values[:, [0, 2]].mean():.4f} "
                    f"w={np.mean([r[5] for r in rows]):.3f}", flush=True)
        values = np.asarray([r[1:5] for r in rows], dtype=float)
        summary["domains"][domain] = {
            "n": len(rows), "od_dice": float(values[:, 0].mean()),
            "od_assd": float(values[:, 1].mean()), "oc_dice": float(values[:, 2].mean()),
            "oc_assd": float(values[:, 3].mean()),
            "macro_dice": float(values[:, [0, 2]].mean()),
            "assd": float(values[:, [1, 3]].mean()),
            "seconds": float(time.time() - start),
            "fusion_weight_mean": float(np.mean([r[5] for r in rows])),
        }
        with (out / f"stage9_{args.variant}_{domain}_per_case.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image", "od_dice", "od_assd", "oc_dice", "oc_assd",
                "fusion_weight", "candidate_entropy", "candidate_anatomy",
                "candidate_source_disagreement",
            ])
            writer.writerows(rows)
    summary["average"] = {
        key: float(np.mean([summary["domains"][d][key] for d in args.order]))
        for key in ("od_dice", "od_assd", "oc_dice", "oc_assd", "macro_dice", "assd")
    }
    (out / f"stage9_{args.variant}_{args.order}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("FUSION_COMPLETE " + json.dumps(summary["average"], sort_keys=True), flush=True)


def unit_test():
    base = {"feature_mean": [0.0] * 8, "feature_std": [1.0] * 8,
            "weight": [0.0] * 8, "bias": 0.0}
    assert fixed_calibration(base, "fixed50")["bias"] == 0.0
    assert fixed_calibration(base, "entropy")["weight"][1] < 0
    anatomy = fixed_calibration(base, "anatomy")
    assert anatomy["weight"] == [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0]
    assert anatomy["bias"] == 0.0
    print('STAGE9_FUSION_UNIT {"fixed": true, "feature_modes": true, '
          '"anatomy_weight": 4.0}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--ckpt", type=Path); parser.add_argument("--shape-stats", type=Path)
    parser.add_argument("--source-manifest", type=Path); parser.add_argument("--calibration", type=Path)
    parser.add_argument("--variant"); parser.add_argument("--output", type=Path)
    parser.add_argument("--order", choices=("CD", "DC")); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=20)
    args = parser.parse_args()
    if args.unit_test:
        unit_test()
    else:
        missing = [x for x in ("ckpt", "shape_stats", "source_manifest", "calibration", "output", "order")
                   if getattr(args, x) is None]
        if missing: parser.error("missing: " + ",".join(missing))
        run(args)
