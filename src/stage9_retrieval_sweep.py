#!/usr/bin/env python3
"""Stage 9A: core SicTTA prototype retrieval top-k sweep."""
from __future__ import annotations

import argparse
import csv
import json
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
    to_prob,
)
from table2_eval_faithful import DS, MAN


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:" + str(args.gpu))
    source_mean, source_std = source_style_stats(args.source_manifest)
    adapter, _ = build_stage3_adapter(device, args.ckpt, args.shape_stats, hard_anatomy=True)
    adapter.topk = int(args.topk)
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refusing to overwrite non-empty output: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": f"topk{args.topk}", "topk": args.topk, "seed": args.seed,
        "order": args.order, "target_labels_for_adaptation": False,
        "fixed_fifo_capacity": 40, "domains": {},
    }
    for domain in args.order:
        dataset = DS(MAN / f"target_domain_{domain.lower()}_stream.csv")
        if args.limit: dataset.rows = dataset.rows[:args.limit]
        rows = []; start = time.time()
        for index in range(len(dataset)):
            x, y, name = dataset[index]
            corrected = align_to_source(x.to(device), source_mean, source_std)
            with torch.no_grad(): prob = to_prob(adapter(corrected, [name]))
            pred = prob.argmax(1)[0].detach().cpu().numpy()
            target = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
            record = [name]
            for label in (1, 2):
                a, b = pred == label, target == label
                record.extend([
                    float((2 * (a & b).sum() + 1e-5) / (a.sum() + b.sum() + 1e-5)),
                    capped_assd(a, b),
                ])
            record.extend([int(adapter.pool_accept), int(adapter.pool.feature_bank.shape[0])])
            rows.append(record)
            if (index + 1) % args.report_every == 0:
                values = np.asarray([r[1:5] for r in rows], dtype=float)
                print(
                    f"TOPK_PROGRESS topk={args.topk} seed={args.seed} order={args.order} "
                    f"domain={domain} case={index + 1}/{len(dataset)} "
                    f"dice={values[:, [0, 2]].mean():.4f}", flush=True)
        values = np.asarray([r[1:5] for r in rows], dtype=float)
        summary["domains"][domain] = {
            "n": len(rows), "od_dice": float(values[:, 0].mean()),
            "od_assd": float(values[:, 1].mean()), "oc_dice": float(values[:, 2].mean()),
            "oc_assd": float(values[:, 3].mean()),
            "macro_dice": float(values[:, [0, 2]].mean()),
            "assd": float(values[:, [1, 3]].mean()),
            "seconds": float(time.time() - start),
            "pool_accept_final": int(adapter.pool_accept),
            "pool_size_final": int(adapter.pool.feature_bank.shape[0]),
        }
        with (out / f"stage9_topk{args.topk}_{domain}_per_case.csv").open(
                "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "image", "od_dice", "od_assd", "oc_dice", "oc_assd",
                "pool_accept_cumulative", "pool_size",
            ])
            writer.writerows(rows)
    summary["average"] = {
        key: float(np.mean([summary["domains"][d][key] for d in args.order]))
        for key in ("od_dice", "od_assd", "oc_dice", "oc_assd", "macro_dice", "assd")
    }
    (out / f"stage9_topk{args.topk}_{args.order}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print("TOPK_COMPLETE " + json.dumps(summary["average"], sort_keys=True), flush=True)


def unit_test():
    assert [1, 2, 3][0] == 1
    print('STAGE9_TOPK_UNIT {"config": true}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-test", action="store_true")
    parser.add_argument("--ckpt", type=Path); parser.add_argument("--shape-stats", type=Path)
    parser.add_argument("--source-manifest", type=Path); parser.add_argument("--output", type=Path)
    parser.add_argument("--order", choices=("CD", "DC")); parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0); parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--report-every", type=int, default=20); parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    if args.unit_test:
        unit_test()
    else:
        missing = [x for x in ("ckpt", "shape_stats", "source_manifest", "output", "order")
                   if getattr(args, x) is None]
        if missing: parser.error("missing: " + ",".join(missing))
        run(args)
