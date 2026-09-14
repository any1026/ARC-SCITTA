#!/usr/bin/env python3
"""Input-level source-style correction wrapped around frozen Anatomy-SicTTA v1.

The only changed component is an unlabeled per-image channel-wise AdaIN
correction.  The corrected v1 adapter, hard anatomy admission, CCD history,
and memory capacity are imported unchanged from the audited v1 tree.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from v2_policy_matrix import DS, MAN, capped_assd, make_model, to_prob
import sictta_anatomy_v1_corrected as anatomy_v1
from sotas import sictta as official_sictta


def source_style_stats(manifest):
    means, stds = [], []
    with manifest.open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            image = np.asarray(Image.open(row['image']).convert('RGB'), np.float32) / 255.0
            means.append(image.mean(axis=(0, 1)))
            stds.append(image.std(axis=(0, 1)))
    if not means:
        raise RuntimeError('source manifest contains no images: ' + str(manifest))
    return (torch.tensor(np.asarray(means).mean(0), dtype=torch.float32),
            torch.tensor(np.asarray(stds).mean(0), dtype=torch.float32))


def align_to_source(x, mean, std):
    """Match each image's channel moments to source-train moments only."""
    mean = mean.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = std.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    z = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    image_mean = z.mean(dim=(2, 3), keepdim=True)
    image_std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    corrected = (z - image_mean) / image_std * std + mean
    return corrected.clamp(0.0, 1.0) * 2.0 - 1.0


def save_preview(out, domain, index, name, corrected, pred, target):
    """Write compact qualitative artifacts for later contact-sheet review."""
    preview = out / 'previews' / domain
    preview.mkdir(parents=True, exist_ok=True)
    stem = f'{index:04d}_{Path(name).stem}'
    image = corrected[0].detach().cpu().numpy().transpose(1, 2, 0)
    image = (np.clip((image + 1.0) / 2.0, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(image).save(preview / f'{stem}_input.png')
    palette = np.asarray([[0, 0, 0], [255, 255, 255], [255, 0, 0]], np.uint8)
    Image.fromarray(palette[pred.astype(np.int64)]).save(preview / f'{stem}_pred.png')
    Image.fromarray(palette[target.astype(np.int64)]).save(preview / f'{stem}_gt.png')


def adapter_diagnostics(adapter):
    if adapter is None:
        return {'ccd_pass': 0, 'anatomy_pass': 0,
                'pool_accept': 0, 'pool_size': 0}
    if hasattr(adapter, 'diagnostics'):
        return adapter.diagnostics()
    # Official SicTTA has no diagnostics() method; expose only its observable
    # pool size and leave admission counters unavailable rather than guessing.
    return {'ccd_pass': None, 'anatomy_pass': None,
            'pool_accept': None,
            'pool_size': int(adapter.pool.feature_bank.shape[0])}


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    source_mean, source_std = source_style_stats(args.source_manifest)
    device = torch.device('cuda:' + str(args.gpu))
    source_model = make_model(device, args.ckpt)
    anchor = make_model(device, args.ckpt).eval()
    if args.mode == 'anatomy':
        adapter = anatomy_v1.TTA(
            anatomy_v1.configure_model(source_model), anchor,
            str(args.shape_stats))
    elif args.mode == 'official':
        adapter = official_sictta.TTA(
            official_sictta.configure_model(source_model), anchor)
    elif args.mode == 'official_fixed':
        # Official CCD without anatomy, but use the audited v1 implementation
        # so its corrected FIFO/history capacity remains exactly 40.
        adapter = anatomy_v1.TTA(
            anatomy_v1.configure_model(source_model), anchor,
            str(args.shape_stats))
        adapter.get_fine_anatomy = lambda _prediction: True
    else:
        adapter = None
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'method': f'input_style_correction_plus_{args.mode}',
        'seed': args.seed, 'order': args.order, 'limit': args.limit,
        'correction': 'per-image channel-wise AdaIN to source-train moments',
        'source_mean_rgb': [float(v) for v in source_mean],
        'source_std_rgb': [float(v) for v in source_std],
        'target_labels_used_for_admission_or_correction': False,
        'base_adapter': ('sictta_anatomy_v1_corrected.py (unchanged)'
                         if args.mode == 'anatomy' else args.mode),
        'domains': {},
    }
    for domain in args.order:
        dataset = DS(MAN / f'target_domain_{domain.lower()}_stream.csv')
        if args.limit:
            dataset.rows = dataset.rows[:args.limit]
        rows, before_moments, after_moments = [], [], []
        start = time.time()
        for index in range(len(dataset)):
            x, y, name = dataset[index]
            x = x.to(device)
            corrected = align_to_source(x, source_mean, source_std)
            before_moments.append(x.detach().cpu().numpy().mean((0, 2, 3)))
            after_moments.append(corrected.detach().cpu().numpy().mean((0, 2, 3)))
            if adapter is None:
                output = to_prob(anchor(corrected))
            else:
                output = to_prob(adapter(corrected, [name]))
            pred = output.argmax(1)[0].detach().cpu().numpy()
            y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            if args.save_previews and index < args.save_previews:
                save_preview(out, domain, index, name, corrected, pred, y_np)
            record = [name]
            for label in (1, 2):
                predicted, target = pred == label, y_np == label
                record.extend([
                    float((2 * (predicted & target).sum() + 1e-5) /
                          (predicted.sum() + target.sum() + 1e-5)),
                    capped_assd(predicted, target),
                ])
            diagnostics = adapter_diagnostics(adapter)
            record.extend([
                diagnostics['ccd_pass'], diagnostics['anatomy_pass'],
                diagnostics['pool_accept'], diagnostics['pool_size'],
            ])
            rows.append(record)
            if (index + 1) % args.report_every == 0:
                values = np.asarray([row[1:5] for row in rows], dtype=float)
                print('INPUT_CORRECTION_PROGRESS '
                      f'seed={args.seed} order={args.order} domain={domain} '
                      f'case={index + 1}/{len(dataset)} '
                      f'dice={values[:, [0, 2]].mean():.4f} '
                      f'pool={diagnostics["pool_size"]}', flush=True)
        values = np.asarray([row[1:5] for row in rows], dtype=float)
        diagnostics = adapter_diagnostics(adapter)
        summary['domains'][domain] = {
            'n': len(rows),
            'od_dice': float(values[:, 0].mean()),
            'od_assd': float(values[:, 1].mean()),
            'oc_dice': float(values[:, 2].mean()),
            'oc_assd': float(values[:, 3].mean()),
            'macro_dice': float(values[:, [0, 2]].mean()),
            'assd': float(values[:, [1, 3]].mean()),
            'seconds': float(time.time() - start),
            'diagnostics_cumulative': diagnostics,
            'input_mean_before_rgb': [float(v) for v in np.asarray(before_moments).mean(0)],
            'input_mean_after_rgb': [float(v) for v in np.asarray(after_moments).mean(0)],
        }
        with (out / f'input_correction_{domain}_per_case.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image', 'od_dice', 'od_assd', 'oc_dice', 'oc_assd',
                             'ccd_pass_cumulative', 'anatomy_pass_cumulative',
                             'pool_accept_cumulative', 'pool_size'])
            writer.writerows(rows)
    summary['average'] = {
        key: float(np.mean([summary['domains'][domain][key]
                            for domain in args.order]))
        for key in ('od_dice', 'od_assd', 'oc_dice', 'oc_assd',
                    'macro_dice', 'assd')
    }
    path = out / f'input_correction_{args.order}_summary.json'
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('INPUT_CORRECTION_COMPLETE ' +
          json.dumps(summary['average'], sort_keys=True), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=Path, required=True)
    parser.add_argument('--shape-stats', type=Path, required=True)
    parser.add_argument('--source-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', choices=['CD', 'DC'], default='CD')
    parser.add_argument('--mode', choices=['anatomy', 'official', 'official_fixed',
                                           'source_only'],
                        default='anatomy')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--report-every', type=int, default=100)
    parser.add_argument('--save-previews', type=int, default=0,
                        help='save this many qualitative examples per domain')
    run(parser.parse_args())
