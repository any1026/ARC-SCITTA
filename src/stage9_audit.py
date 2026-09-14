#!/usr/bin/env python3
"""Strict audit for Stage9 mechanism sweep against completed Stage8 v2."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def read_summary(path: Path, limit: int):
    summary = json.loads(path.read_text(encoding="utf-8"))
    order = summary.get("order")
    domains = summary.get("domains", {})
    if order not in ("CD", "DC") or set(domains) != set(order):
        raise RuntimeError(f"metadata {path}")
    for domain in order:
        d = domains[domain]
        if int(d.get("n", -1)) != limit:
            raise RuntimeError(f"count {path} {domain}")
        if not all(finite(d.get(k)) for k in
                   ("macro_dice", "assd", "od_dice", "oc_dice", "od_assd", "oc_assd")):
            raise RuntimeError(f"nonfinite {path} {domain}")
        csvs = list(path.parent.glob(f"*_{domain}_per_case.csv"))
        if len(csvs) != 1:
            raise RuntimeError(f"per-case {path} {domain}")
        with csvs[0].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != limit:
            raise RuntimeError(f"row count {csvs[0]}")
        for row in rows:
            if not all(finite(row.get(k)) for k in
                       ("od_dice", "oc_dice", "od_assd", "oc_assd")):
                raise RuntimeError(f"nonfinite row {csvs[0]}")
    return summary


def grouped(method: str, orders: dict[str, dict]):
    output = []
    for order in ("ALL", "CD", "DC"):
        domains = []
        for actual in ("CD", "DC"):
            if order != "ALL" and order != actual:
                continue
            domains.extend(orders[actual]["domains"][d] for d in actual)
        dice = np.asarray([d["macro_dice"] for d in domains], dtype=float)
        assd = np.asarray([d["assd"] for d in domains], dtype=float)
        output.append({
            "method": method, "order": order, "n_domain_runs": len(domains),
            "macro_dice_mean": float(dice.mean()),
            "macro_dice_sd_sample": float(dice.std(ddof=1)),
            "assd_mean": float(assd.mean()),
            "assd_sd_sample": float(assd.std(ddof=1)),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    paths = sorted(args.root.glob("**/*summary.json"))
    if not paths:
        raise RuntimeError("no Stage9 summaries")
    by_method = {}
    for path in paths:
        summary = read_summary(path, args.limit)
        by_method.setdefault(summary["method"], {})[summary["order"]] = summary
    v2_orders = {}
    for order in ("CD", "DC"):
        candidates = sorted((args.v2_root / f"v2_adain_{order}").glob("*summary.json"))
        if len(candidates) != 1:
            raise RuntimeError(f"v2 summary missing {order}")
        v2_orders[order] = read_summary(candidates[0], args.limit)
    methods = sorted(by_method)
    rows = []
    for method in methods:
        if set(by_method[method]) != {"CD", "DC"}:
            raise RuntimeError(f"missing order for {method}")
        rows.extend(grouped(method, by_method[method]))
    rows.extend(grouped("v2_adain", v2_orders))
    lookup = {(r["method"], r["order"]): r for r in rows}
    best = max(methods, key=lambda m: lookup[(m, "ALL")]["macro_dice_mean"])
    deltas = {
        order: {
            "macro_dice": lookup[(best, order)]["macro_dice_mean"] - lookup[("v2_adain", order)]["macro_dice_mean"],
            "assd": lookup[(best, order)]["assd_mean"] - lookup[("v2_adain", order)]["assd_mean"],
        }
        for order in ("ALL", "CD", "DC")
    }
    promote = bool(deltas["CD"]["macro_dice"] >= -0.003 and
                   deltas["DC"]["macro_dice"] >= -0.003 and
                   deltas["ALL"]["assd"] <= 0.15)
    report = {
        "protocol": f"Fundus-doFE; seed1; {args.limit} images/domain; CD/DC",
        "methods": methods, "grouped": rows, "best_candidate": best,
        "best_minus_v2": deltas, "promote_multiseed100": promote,
        "decision_rule": "both direction Dice deltas >= -0.003 and overall ASSD delta <= +0.15",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "grouped.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"best": best, "deltas": deltas, "promote": promote}, sort_keys=True))
    print("STAGE9_AUDIT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
