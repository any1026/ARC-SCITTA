#!/usr/bin/env python3
"""Isolated v2 prototype matrix for reversible hard/soft/tri-state policies.

This file is intentionally independent of the frozen v1 tree. It wraps the
official SicTTA adapter, snapshots mutable state before each case, and only
commits an adapted state for the high-reliability tri-state branch. The soft
branch is an output-level interpolation and restores state for non-severe
cases; it is an audit prototype, not the final paper method.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from scipy import ndimage
import torch
from robustbench.seg_net.unet import UNet
from sotas import sictta as official_sictta
from table2_eval_faithful import DS, MAN


def make_model(device, ckpt):
    p = {'in_chns': 3, 'ft_chns': [16, 32, 64, 128, 256],
         'dropout_p': [0, 0, .3, .4, .5], 'n_classes': 3,
         'bilinear': True, 'deep_supervise': False, 'lr': 1e-3,
         'up_mode': 'upsample'}
    model = UNet(p).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)['model'])
    return model


def to_prob(out):
    if out.ndim != 4:
        raise ValueError('expected [B,C,H,W] output')
    sums = out.detach().sum(1, keepdim=True)
    if bool(torch.isfinite(out).all()) and bool((out >= 0).all()) and bool(
            torch.mean(torch.abs(sums - 1.0)) < 1e-3):
        return out
    return out.softmax(1)


def _bound_score(value, bounds):
    low, high = float(bounds['low']), float(bounds['high'])
    if low <= value <= high:
        return 1.0
    span = max(high - low, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return float(max(0.0, 1.0 - distance / span))


def anatomy_features(prob, shape_stats):
    """Continuous source-derived anatomy scores for Fundus output."""
    pred = prob.argmax(1)[0].detach().cpu().numpy()
    disc, cup = pred > 0, pred == 2
    if not disc.any() or not cup.any():
        return {'anatomy_score': 0.0, 'missing_structure': 1.0}

    def largest_fraction(mask):
        labels, count = ndimage.label(mask)
        if count == 0:
            return 0.0
        sizes = np.bincount(labels.ravel())[1:]
        return float(sizes.max() / max(1, mask.sum()))

    disc_area, cup_area = float(disc.sum()), float(cup.sum())
    disc_center = np.asarray(ndimage.center_of_mass(disc), dtype=np.float64)
    cup_center = np.asarray(ndimage.center_of_mass(cup), dtype=np.float64)
    disc_radius = math.sqrt(disc_area / math.pi)
    values = {
        'cup_disc_ratio': cup_area / disc_area,
        'center_distance_norm': float(np.linalg.norm(cup_center - disc_center) /
                                      max(disc_radius, 1e-6)),
        'disc_lcc_fraction': largest_fraction(disc),
        'cup_lcc_fraction': largest_fraction(cup),
    }
    scores = {name: _bound_score(value, shape_stats['admission_bounds'][name])
              for name, value in values.items()}
    return {'anatomy_score': float(min(scores.values())),
            'missing_structure': 0.0,
            **{f'anatomy_{k}': float(v) for k, v in scores.items()}}


def entropy_confidence(prob):
    c = prob.shape[1]
    ent = (-(prob.clamp_min(1e-6) * prob.clamp_min(1e-6).log()).sum(1)).mean()
    return float((1.0 - ent / math.log(c)).clamp(0.0, 1.0).item())


class FullSnapshot:
    """Snapshot model, pool, counters, mutable lists and RNG states."""
    def __init__(self, adapter):
        self.model_state = {k: v.detach().clone()
                            for k, v in adapter.model.state_dict().items()}
        pool = adapter.pool
        self.pool = {
            'feature_bank': pool.feature_bank.detach().clone(),
            'image_bank': pool.image_bank.detach().clone(),
            'mask_bank': pool.mask_bank.detach().clone(),
            'name_list': list(pool.name_list),
        }
        self.attrs = {}
        for key, value in adapter.__dict__.items():
            if key in {'_parameters', '_buffers', '_modules', 'model',
                       'model_anchor', 'pool'}:
                continue
            try:
                self.attrs[key] = copy.deepcopy(value)
            except Exception:
                pass
        self.bn_flags = [(m, m.training, m.track_running_stats)
                         for m in adapter.model.modules()
                         if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
        self.py_rng = random.getstate()
        self.np_rng = np.random.get_state()
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None)

    def restore(self, adapter):
        adapter.model.load_state_dict(self.model_state, strict=True)
        pool = adapter.pool
        pool.feature_bank = self.pool['feature_bank'].detach().clone()
        pool.image_bank = self.pool['image_bank'].detach().clone()
        pool.mask_bank = self.pool['mask_bank'].detach().clone()
        pool.name_list = list(self.pool['name_list'])
        for key, value in self.attrs.items():
            setattr(adapter, key, copy.deepcopy(value))
        for module, training, track in self.bn_flags:
            module.train(training)
            module.track_running_stats = track
        random.setstate(self.py_rng)
        np.random.set_state(self.np_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)


def state_digest(adapter):
    h = hashlib.sha256()
    for key, value in sorted(adapter.model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
    for key in ('feature_bank', 'image_bank', 'mask_bank'):
        h.update(getattr(adapter.pool, key).detach().cpu().numpy().tobytes())
    h.update(repr(list(adapter.pool.name_list)).encode())
    for key in ('entropy_list', 'ccd_pass', 'pool_accept', 'anatomy_pass'):
        if hasattr(adapter, key):
            h.update(repr(getattr(adapter, key)).encode())
    return h.hexdigest()


def build_adapter(device, ckpt):
    source = make_model(device, ckpt)
    anchor = make_model(device, ckpt).eval()
    return official_sictta.TTA(official_sictta.configure_model(source), anchor), anchor


def capped_assd(a, b):
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return 10.0
    st = ndimage.generate_binary_structure(2, 1)
    ae, be = a ^ ndimage.binary_erosion(a, st), b ^ ndimage.binary_erosion(b, st)
    da, db = ndimage.distance_transform_edt(~a), ndimage.distance_transform_edt(~b)
    return float(min(10.0, (da[be].mean() + db[ae].mean()) / 2))


def run(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:' + str(args.gpu))
    adapter, anchor = build_adapter(device, args.ckpt)
    shape_stats = json.loads(args.shape_stats.read_text(encoding='utf-8'))
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {'method': 'v2_policy_prototype', 'policy': args.policy,
               'anatomy_weight': args.anatomy_weight, 'seed': args.seed,
               'order': args.order, 'stream_repeat': args.repeat,
               'no_target_labels_for_admission': True,
               'base_adapter': 'official SicTTA CCD with full wrapper snapshot',
               'domains': {}}
    for dom in args.order:
        ds = DS(MAN / f'target_domain_{dom.lower()}_stream.csv')
        if args.repeat:
            order_rng = random.Random(args.seed * 1000 + args.repeat * 17 + ord(dom))
            order_rng.shuffle(ds.rows)
        if args.limit:
            ds.rows = ds.rows[:args.limit]
        rows, q_values, anatomy_values = [], [], []
        start = time.time(); accepted = soft = rejected = rollback = committed = 0
        for i in range(len(ds)):
            x, y, name = ds[i]; x = x.to(device)
            source_prob = to_prob(anchor(x))
            before = FullSnapshot(adapter); before_digest = state_digest(adapter)
            adapted_prob = to_prob(adapter(x, [name]))
            af = anatomy_features(adapted_prob, shape_stats)
            agreement = float((source_prob.argmax(1) == adapted_prob.argmax(1)).float().mean().item())
            ccd_proxy = entropy_confidence(adapted_prob); anat = float(af['anatomy_score'])
            q = float(np.clip((1.0 - args.anatomy_weight) * ccd_proxy +
                              args.anatomy_weight * anat, 0.0, 1.0))
            severe = bool(af.get('missing_structure', 0.0) > 0 or anat < 0.25)
            decision, committed_case, rollback_case = 'reject', 0, 0
            if args.policy == 'hard_fullrollback':
                if anat >= 0.999999:
                    decision, committed_case = 'accept', 1
                else:
                    before.restore(adapter); rollback_case = 1
            elif args.policy == 'soft_blend':
                if severe:
                    before.restore(adapter); rollback_case = 1
                else:
                    decision = 'soft'; before.restore(adapter); rollback_case = 1
            elif args.policy == 'tri_state':
                if severe:
                    before.restore(adapter); rollback_case = 1
                elif q >= 0.75:
                    decision, committed_case = 'accept', 1
                else:
                    decision = 'soft'; before.restore(adapter); rollback_case = 1
            else:
                raise ValueError(args.policy)
            if decision == 'accept':
                output_prob = adapted_prob; accepted += 1; committed += committed_case
            elif decision == 'soft':
                output_prob = q * adapted_prob + (1.0 - q) * source_prob; soft += 1
            else:
                output_prob = source_prob; rejected += 1
            rollback += rollback_case
            rollback_exact = int((decision != 'accept') and state_digest(adapter) == before_digest)
            y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            pred = output_prob.argmax(1)[0].detach().cpu().numpy(); rec = [name]
            for k in (1, 2):
                aa, bb = pred == k, y_np == k
                rec.extend([float((2 * (aa & bb).sum() + 1e-5) /
                             (aa.sum() + bb.sum() + 1e-5)), capped_assd(aa, bb)])
            rec.extend([decision, q, anat, agreement, ccd_proxy, rollback_exact])
            rows.append(rec); q_values.append(q); anatomy_values.append(anat)
            if (i + 1) % args.report_every == 0:
                vals = np.asarray([r[1:5] for r in rows], dtype=float)
                print(f'V2_POLICY_PROGRESS policy={args.policy} seed={args.seed} '
                      f'order={args.order} rep={args.repeat} domain={dom} '
                      f'case={i + 1}/{len(ds)} dice={vals[:, [0, 2]].mean():.4f} '
                      f'accepted={accepted} soft={soft} rejected={rejected} rollback={rollback}',
                      flush=True)
        arr = np.asarray([r[1:5] for r in rows], dtype=float)
        summary['domains'][dom] = {
            'n': len(rows), 'od_dice': float(arr[:, 0].mean()), 'od_assd': float(arr[:, 1].mean()),
            'oc_dice': float(arr[:, 2].mean()), 'oc_assd': float(arr[:, 3].mean()),
            'macro_dice': float(arr[:, [0, 2]].mean()), 'assd': float(arr[:, [1, 3]].mean()),
            'seconds': float(time.time() - start), 'accepted': accepted, 'soft': soft,
            'rejected': rejected, 'rollback': rollback, 'committed': committed,
            'rollback_exact_rate': float(np.mean([r[-1] for r in rows])),
            'q_mean': float(np.mean(q_values)), 'q_std': float(np.std(q_values)),
            'anatomy_mean': float(np.mean(anatomy_values)),
        }
        with (out / f'v2_{args.policy}_{dom}_per_case.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.writer(f); w.writerow(['image', 'od_dice', 'od_assd', 'oc_dice', 'oc_assd',
                                            'decision', 'q', 'anatomy_score', 'agreement',
                                            'ccd_proxy', 'rollback_exact']); w.writerows(rows)
    summary['average'] = {k: float(np.mean([summary['domains'][d][k] for d in args.order]))
                          for k in ('od_dice', 'od_assd', 'oc_dice', 'oc_assd', 'macro_dice', 'assd')}
    (out / f'v2_{args.policy}_{args.order}_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('V2_POLICY_COMPLETE', json.dumps(summary['average'], sort_keys=True), flush=True)


def unit_test():
    class Pool:
        def __init__(self):
            self.feature_bank = torch.tensor([1.0]); self.image_bank = torch.tensor([2.0])
            self.mask_bank = torch.tensor([3.0]); self.name_list = ['original']
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = torch.nn.Linear(1, 1); self.pool = Pool()
            self.entropy_list = ['e']; self.ccd_pass = 1; self.pool_accept = 2; self.extra_mutable = 0
    toy = Toy(); snap = FullSnapshot(toy)
    toy.model.weight.data.add_(4); toy.pool.feature_bank = torch.tensor([9.0]); toy.pool.name_list.append('mutated')
    toy.entropy_list.append('x'); toy.ccd_pass = 99; toy.extra_mutable = 7; snap.restore(toy)
    checks = {'model': float(toy.model.weight.detach().abs().sum()) == float(snap.model_state['weight'].abs().sum()),
              'pool': toy.pool.feature_bank.tolist() == [1.0] and toy.pool.name_list == ['original'],
              'attrs': toy.entropy_list == ['e'] and toy.ccd_pass == 1, 'extra': toy.extra_mutable == 0}
    print('V2_FULL_SNAPSHOT_UNIT', json.dumps(checks, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--unit-test', action='store_true')
    parser.add_argument('--policy', choices=['hard_fullrollback', 'soft_blend', 'tri_state'])
    parser.add_argument('--anatomy-weight', type=float, default=0.3); parser.add_argument('--ckpt', type=Path)
    parser.add_argument('--shape-stats', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--order', choices=['CD', 'DC'], default='CD'); parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=0); parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0); parser.add_argument('--report-every', type=int, default=100)
    args = parser.parse_args()
    if args.unit_test:
        unit_test()
    else:
        for required in ('ckpt', 'shape_stats', 'output'):
            if getattr(args, required) is None:
                parser.error(f'--{required.replace("_", "-")} is required')
        run(args)
