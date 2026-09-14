#!/usr/bin/env python3
"""Validate and aggregate the frozen Stage9 paper-table runs.

The aggregator never computes metrics from target labels beyond values already
written by the evaluators. It checks the expected 72 domain-level runs,
400 per-case rows, finite Dice/ASSD fields, and emits a long table plus
mean/sample-SD summaries across seeds and directions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("source_raw", "official_raw", "source_adain", "official_adain",
           "v2_adain", "stage9_anatomy")
ORDERS = ("CD", "DC")
SEEDS = (1, 2, 3)
DOMAINS = ("C", "D")
METRIC_KEYS = ("od_dice", "oc_dice", "macro_dice", "od_assd", "oc_assd", "assd")


def finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_summary(run_dir: Path, method: str, seed: int, order: str):
    candidates = sorted(run_dir.glob("*summary.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"expected one summary in {run_dir}, found {len(candidates)}")
    summary = json.loads(candidates[0].read_text(encoding="utf-8"))
    if summary.get("order") != order:
        raise RuntimeError(f"wrong order in {candidates[0]}")
    if int(summary.get("seed", -1)) != seed:
        raise RuntimeError(f"wrong seed in {candidates[0]}")
    actual = str(summary.get("method", ""))
    if method == "source_raw" and actual != "source_only":
        raise RuntimeError(f"unexpected source method {actual} in {candidates[0]}")
    if method == "official_raw" and actual not in ("official_fixed", "official"):
        raise RuntimeError(f"unexpected official method {actual} in {candidates[0]}")
    if method == "source_adain" and not actual.endswith("source_only"):
        raise RuntimeError(f"unexpected source AdaIN method {actual} in {candidates[0]}")
    if method == "official_adain" and not actual.endswith("official_fixed"):
        raise RuntimeError(f"unexpected official AdaIN method {actual} in {candidates[0]}")
    if method == "v2_adain" and not actual.endswith("anatomy"):
        raise RuntimeError(f"unexpected v2 method {actual} in {candidates[0]}")
    if method == "stage9_anatomy" and actual != "anatomy":
        raise RuntimeError(f"unexpected Stage9 method {actual} in {candidates[0]}")
    domains = summary.get("domains", {})
    if set(domains) != set(DOMAINS):
        raise RuntimeError(f"domain set mismatch in {candidates[0]}")
    per_case = {}
    for domain in DOMAINS:
        values = domains[domain]
        if int(values.get("n", -1)) != 400:
            raise RuntimeError(f"expected n=400 in {candidates[0]} {domain}")
        if not all(finite(values.get(key)) for key in METRIC_KEYS):
            raise RuntimeError(f"nonfinite summary in {candidates[0]} {domain}")
        csv_candidates = sorted(run_dir.glob(f"*_{domain}_per_case.csv"))
        if len(csv_candidates) != 1:
            raise RuntimeError(f"expected one per-case CSV in {run_dir} {domain}")
        with csv_candidates[0].open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != 400:
            raise RuntimeError(f"expected 400 rows in {csv_candidates[0]}, found {len(rows)}")
        for row in rows:
            if not all(finite(row.get(key)) for key in ("od_dice", "oc_dice", "od_assd", "oc_assd")):
                raise RuntimeError(f"nonfinite case in {csv_candidates[0]}")
        names = [row.get("image") for row in rows]
        if any(not name for name in names) or len(set(names)) != 400:
            raise RuntimeError(f"missing or duplicate image names in {csv_candidates[0]}")
        case_macro = np.asarray([(float(row["od_dice"]) + float(row["oc_dice"])) / 2.0
                                 for row in rows], dtype=np.float64)
        case_assd = np.asarray([(float(row["od_assd"]) + float(row["oc_assd"])) / 2.0
                                for row in rows], dtype=np.float64)
        if not math.isclose(float(case_macro.mean()), float(values["macro_dice"]),
                            rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"summary/case Dice mismatch in {csv_candidates[0]}")
        if not math.isclose(float(case_assd.mean()), float(values["assd"]),
                            rel_tol=0.0, abs_tol=1e-8):
            raise RuntimeError(f"summary/case ASSD mismatch in {csv_candidates[0]}")
        per_case[domain] = {
            row["image"]: {
                "macro_dice": (float(row["od_dice"]) + float(row["oc_dice"])) / 2.0,
                "assd": (float(row["od_assd"]) + float(row["oc_assd"])) / 2.0,
            }
            for row in rows
        }
    return candidates[0], per_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.out}")
    records = []
    checksums = []
    cases = {}
    for method in METHODS:
        for seed in SEEDS:
            for order in ORDERS:
                run_dir = args.root / method / f"seed{seed}_{order}"
                summary_path, per_case = locate_summary(run_dir, method, seed, order)
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                checksums.append({"path": str(summary_path), "sha256": sha256(summary_path)})
                for domain in DOMAINS:
                    cases[(method, seed, order, domain)] = per_case[domain]
                    values = summary["domains"][domain]
                    records.append({
                        "method": method, "seed": seed, "order": order,
                        "domain": domain, **{key: float(values[key]) for key in METRIC_KEYS},
                    })
    args.out.mkdir(parents=True, exist_ok=True)
    fieldnames = ["method", "seed", "order", "domain", *METRIC_KEYS]
    with (args.out / "domain_level_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(records)
    aggregate = []
    for method in METHODS:
        for group_name, predicate in (
            ("ALL", lambda row: True),
            ("CD", lambda row: row["order"] == "CD"),
            ("DC", lambda row: row["order"] == "DC"),
        ):
            selected = [row for row in records if row["method"] == method and predicate(row)]
            row = {"method": method, "group": group_name, "n_seeds": len(SEEDS),
                   "n_domain_runs": len(selected)}
            for key in METRIC_KEYS:
                seed_values = np.asarray([
                    np.mean([entry[key] for entry in selected if entry["seed"] == seed])
                    for seed in SEEDS
                ], dtype=np.float64)
                row[f"{key}_mean"] = float(seed_values.mean())
                row[f"{key}_sd_across_seeds"] = float(seed_values.std(ddof=1))
            aggregate.append(row)
    with (args.out / "paper_table_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    safety = []
    for method in METHODS:
        for group_name, orders in (("ALL", ORDERS), ("CD", ("CD",)), ("DC", ("DC",))):
            dice, assd, gains, assd_changes = [], [], [], []
            for seed in SEEDS:
                for order in orders:
                    for domain in DOMAINS:
                        current = cases[(method, seed, order, domain)]
                        source = cases[("source_raw", seed, order, domain)]
                        if set(current) != set(source):
                            raise RuntimeError(
                                f"case identity mismatch method={method} seed={seed} order={order} domain={domain}")
                        for name in source:
                            dice.append(current[name]["macro_dice"])
                            assd.append(current[name]["assd"])
                            gains.append(current[name]["macro_dice"] - source[name]["macro_dice"])
                            assd_changes.append(current[name]["assd"] - source[name]["assd"])
            dice_np = np.asarray(dice, dtype=np.float64)
            assd_np = np.asarray(assd, dtype=np.float64)
            gain_np = np.asarray(gains, dtype=np.float64)
            assd_change_np = np.asarray(assd_changes, dtype=np.float64)
            cutoff = max(1, int(math.ceil(0.25 * len(dice_np))))
            safety.append({
                "method": method, "group": group_name, "n_cases": len(dice_np),
                "macro_dice_gain_vs_source_mean": float(gain_np.mean()),
                "negative_transfer_rate": float(np.mean(gain_np < 0.0)),
                "negative_transfer_gt_1pp_rate": float(np.mean(gain_np < -0.01)),
                "macro_dice_q10": float(np.quantile(dice_np, 0.10)),
                "macro_dice_worst_quartile_mean": float(np.sort(dice_np)[:cutoff].mean()),
                "assd_change_vs_source_mean": float(assd_change_np.mean()),
                "assd_q90": float(np.quantile(assd_np, 0.90)),
            })
    with (args.out / "paper_safety_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(safety[0]))
        writer.writeheader(); writer.writerows(safety)
    lookup = {(row["method"], row["group"]): row for row in aggregate}
    contrasts = []
    for method in METHODS:
        cd = lookup[(method, "CD")]
        dc = lookup[(method, "DC")]
        contrasts.append({
            "method": method,
            "cd_minus_dc_macro_dice": cd["macro_dice_mean"] - dc["macro_dice_mean"],
            "cd_minus_dc_capped_2d_assd": cd["assd_mean"] - dc["assd_mean"],
        })
    with (args.out / "direction_contrast.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader(); writer.writerows(contrasts)
    report = {
        "protocol": "Fundus-doFE; 320x320; 400 images/domain; CD/DC; seeds 1,2,3",
        "methods": list(METHODS), "n_domain_level_runs": len(records),
        "metric_label": "capped 2D ASSD (project implementation; not strict mm ASSD/HD95)",
        "target_labels_used_for_adaptation": False,
        "summary_rows": aggregate, "safety_rows": safety,
        "direction_contrasts": contrasts,
        "negative_transfer_definition": "paired per-case Macro Dice lower than the matching source_raw result",
        "summary_sha256": checksums,
    }
    (args.out / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(records), "out": str(args.out)}, sort_keys=True), flush=True)
    print("STAGE9_PAPER_AGGREGATE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
