#!/usr/bin/env python3
"""Stage 3: Delta-Risk Transactional SicTTA for Fundus streams.

The proposed mechanism is a two-phase, unlabeled transaction rather than an
extra hard gate.  A candidate SicTTA memory update is proposed, evaluated with
the same image under a geometry-preserving view, and then committed only when
the candidate lowers a robust rolling risk estimate.  Borderline candidates
are returned through a soft source/adapted blend after exact rollback; harmful
candidates are rejected and rolled back.  The risk threshold is calibrated
online from the recent candidate-risk deltas (median absolute deviation), not
from target labels.
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
from PIL import Image
from robustbench.seg_net.unet import UNet
from sotas import sictta as official_sictta
from table2_eval_faithful import DS, MAN
import sictta_anatomy_v1_corrected as anatomy_v1


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
        # ``state_dict`` does not include U-Net forward caches.  Snapshot
        # simple model/pool attributes as well so a failed transaction cannot
        # leak blocks1/latent_A1/last_x or a quality bank into the next case.
        self.model_attrs = _copy_simple_attrs(adapter.model)
        self.model_optimizers = _copy_optimizer_states(adapter.model)
        self.pool_attrs = _copy_simple_attrs(adapter.pool)
        self.attrs = {key: copy.deepcopy(value)
                      for key, value in adapter.__dict__.items()
                      if key not in _SNAPSHOT_INTERNALS and
                      key not in {'model', 'model_anchor', 'pool'} and
                      _snapshot_value(value)}
        self.optimizer_states = _copy_optimizer_states(adapter)
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
        _restore_simple_attrs(adapter.model, self.model_attrs)
        _restore_optimizer_states(adapter.model, self.model_optimizers)
        _restore_simple_attrs(adapter.pool, self.pool_attrs)
        _restore_simple_attrs(adapter, self.attrs)
        _restore_optimizer_states(adapter, self.optimizer_states)
        for module, training, track in self.bn_flags:
            module.train(training)
            module.track_running_stats = track
        random.setstate(self.py_rng)
        np.random.set_state(self.np_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)


_SNAPSHOT_INTERNALS = {'_parameters', '_buffers', '_modules'}


def _snapshot_value(value):
    """Return whether a value is cheap and deterministic to snapshot.

    The upstream U-Net stores transient tensors such as ``blocks1`` and
    ``latent_A1`` directly on the model object.  They are not part of
    ``state_dict`` but can still affect a subsequent ``get_output`` call, so
    they must be included in a transaction snapshot.  Optimizers and modules
    are handled separately and are deliberately excluded here.
    """
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return True
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_snapshot_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, (str, int, float, bool)) and
                   _snapshot_value(item) for key, item in value.items())
    return False


def _copy_simple_attrs(obj):
    return {key: copy.deepcopy(value)
            for key, value in obj.__dict__.items()
            if key not in _SNAPSHOT_INTERNALS and _snapshot_value(value)}


def _copy_optimizer_states(obj):
    states = {}
    for key, value in obj.__dict__.items():
        if isinstance(value, torch.optim.Optimizer):
            states[key] = copy.deepcopy(value.state_dict())
    return states


def _restore_simple_attrs(obj, saved):
    for key, value in list(obj.__dict__.items()):
        if key in _SNAPSHOT_INTERNALS:
            continue
        if key not in saved and _snapshot_value(value):
            del obj.__dict__[key]
    for key, value in saved.items():
        obj.__dict__[key] = copy.deepcopy(value)


def _restore_optimizer_states(obj, saved):
    for key, state in saved.items():
        value = getattr(obj, key, None)
        if isinstance(value, torch.optim.Optimizer):
            value.load_state_dict(copy.deepcopy(state))


def _hash_value(hasher, value):
    """Canonical hash for tensors, numpy arrays and nested simple values."""
    if torch.is_tensor(value):
        value_cpu = value.detach().cpu().contiguous()
        hasher.update(b'tensor')
        hasher.update(str(value_cpu.dtype).encode())
        hasher.update(repr(tuple(value_cpu.shape)).encode())
        hasher.update(value_cpu.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        value_cpu = np.ascontiguousarray(value)
        hasher.update(b'ndarray')
        hasher.update(str(value_cpu.dtype).encode())
        hasher.update(repr(value_cpu.shape).encode())
        hasher.update(value_cpu.tobytes())
    elif isinstance(value, dict):
        hasher.update(b'dict')
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_value(hasher, key); _hash_value(hasher, value[key])
    elif isinstance(value, (list, tuple)):
        hasher.update(type(value).__name__.encode())
        for item in value:
            _hash_value(hasher, item)
    else:
        hasher.update(repr(value).encode())


def snapshot_digest(snapshot):
    """Hash every field represented by ``FullSnapshot``."""
    h = hashlib.sha256()
    for key in ('model_state', 'model_attrs', 'model_optimizers',
                'pool_attrs', 'attrs', 'optimizer_states', 'bn_flags',
                'py_rng', 'np_rng', 'torch_rng', 'cuda_rng'):
        h.update(key.encode())
        _hash_value(h, getattr(snapshot, key))
    return h.hexdigest()


def state_digest(adapter):
    """Digest the same complete state that a rollback snapshot stores."""
    return snapshot_digest(FullSnapshot(adapter))


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


def source_style_stats(manifest):
    """Estimate source RGB moments from source images only."""
    means, stds = [], []
    with Path(manifest).open(encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            image = np.asarray(Image.open(row['image']).convert('RGB'),
                               np.float32) / 255.0
            means.append(image.mean(axis=(0, 1)))
            stds.append(image.std(axis=(0, 1)))
    if not means:
        raise RuntimeError('source manifest contains no images: ' + str(manifest))
    return (torch.tensor(np.asarray(means).mean(0), dtype=torch.float32),
            torch.tensor(np.asarray(stds).mean(0), dtype=torch.float32))


def align_to_source(x, mean, std):
    """Unlabeled per-image RGB moment alignment used by the existing v2."""
    mean = mean.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = std.to(device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    z = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    image_mean = z.mean(dim=(2, 3), keepdim=True)
    image_std = z.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
    corrected = (z - image_mean) / image_std * std + mean
    return corrected.clamp(0.0, 1.0) * 2.0 - 1.0


def flip_back(prob):
    return torch.flip(prob, dims=[3])


def normalized_entropy(prob):
    p = prob.clamp_min(1e-6)
    return float((-(p * p.log()).sum(1).mean() /
                  math.log(max(2, prob.shape[1]))).clamp(0.0, 1.0).item())


def view_risk(prob, view_prob, shape_stats):
    """Unlabeled risk proxy: uncertainty + view inconsistency + anatomy risk."""
    agreement = float((prob.argmax(1) == view_prob.argmax(1)).float().mean().item())
    anatomy = float(anatomy_features(prob, shape_stats).get('anatomy_score', 0.0))
    risk = (0.45 * normalized_entropy(prob) +
            0.35 * (1.0 - agreement) +
            0.20 * (1.0 - anatomy))
    return float(np.clip(risk, 0.0, 1.0)), agreement, anatomy


def robust_margin(history, floor=0.012, scale=0.75):
    """Adaptive decision margin from recent candidate deltas."""
    if len(history) < 8:
        return float(floor)
    recent = np.asarray(history[-32:], dtype=np.float64)
    med = float(np.median(recent))
    mad = float(np.median(np.abs(recent - med)))
    return float(max(floor, scale * 1.4826 * mad))


def build_stage3_adapter(device, ckpt, shape_stats_path, hard_anatomy=False):
    """Build corrected CCD/FIFO adapter; stage3 uses continuous review instead
    of the old hard anatomy gate unless ``hard_anatomy`` is requested."""
    source = make_model(device, ckpt)
    anchor = make_model(device, ckpt).eval()
    source = anatomy_v1.configure_model(source)
    source.train()
    adapter = anatomy_v1.TTA(source, anchor, str(shape_stats_path))
    if not hard_anatomy:
        adapter.get_fine_anatomy = lambda _prediction: True
    return adapter, anchor


def readonly_predict(adapter, x, name):
    """Run one adapter prediction and restore *all* state immediately.

    This is used for view-risk evaluation.  It deliberately keeps the
    adapter's prediction path (so the candidate memory is exercised), but the
    view can no longer leak CCD history, pool entries or model caches into the
    live transaction.
    """
    snapshot = FullSnapshot(adapter)
    try:
        with torch.no_grad():
            return to_prob(adapter(x, [name]))
    finally:
        snapshot.restore(adapter)


def transactional_step(adapter, anchor, x, name, shape_stats, history):
    """Propose -> audit -> commit/soft-return/reject with exact rollback."""
    before = FullSnapshot(adapter)
    before_digest = snapshot_digest(before)
    ccd_before = int(getattr(adapter, 'ccd_pass', 0))
    pool_before = int(getattr(adapter, 'pool_accept', 0))
    with torch.no_grad():
        source_prob = to_prob(anchor(x))
        source_view = flip_back(to_prob(anchor(torch.flip(x, dims=[3]))))
        pre_risk, _, _ = view_risk(source_prob, source_view, shape_stats)

        # First pass proposes one real SicTTA memory update.
        candidate_prob = to_prob(adapter(x, [name]))
        candidate_ccd = int(getattr(adapter, 'ccd_pass', 0)) - ccd_before
        candidate_admitted = int(getattr(adapter, 'pool_accept', 0)) - pool_before
        candidate_state = FullSnapshot(adapter)

        # Evaluate the second view read-only from the candidate state.  The
        # helper restores candidate_state even if adapter.forward raises.
        candidate_view = flip_back(readonly_predict(
            adapter, torch.flip(x, dims=[3]), name + '::view'))
        post_risk, agreement, anatomy = view_risk(
            candidate_prob, candidate_view, shape_stats)

    delta = float(post_risk - pre_risk)
    margin = robust_margin(history)
    q = float(np.clip(0.5 - delta / max(2.0 * margin, 1e-6), 0.05, 0.95))
    severe = bool(anatomy < 0.20)
    if (not severe) and delta <= -margin:
        decision = 'commit'
        # candidate_state is already the current state; commit it explicitly
        # to make the transaction boundary obvious and testable.
        candidate_state.restore(adapter)
        # Restoring candidate_state also restores the RNG state at the end of
        # the candidate proposal, so the committed state is deterministic.
        accepted = 1; soft = rejected = rollback = 0
        output_prob = candidate_prob
    elif (not severe) and delta <= margin:
        decision = 'soft'
        before.restore(adapter)
        accepted = rejected = 0; soft = rollback = 1
        output_prob = q * candidate_prob + (1.0 - q) * source_prob
    else:
        decision = 'reject'
        before.restore(adapter)
        accepted = soft = 0; rejected = rollback = 1
        output_prob = source_prob
    history.append(delta)
    rollback_exact = int(decision != 'commit' and state_digest(adapter) == before_digest)
    return output_prob, {
        'decision': decision, 'q': q, 'pre_risk': pre_risk,
        'post_risk': post_risk, 'risk_delta': delta, 'margin': margin,
        'anatomy_score': anatomy, 'agreement': agreement,
        'accepted': accepted, 'soft': soft, 'rejected': rejected,
        'rollback': rollback, 'rollback_exact': rollback_exact,
        'candidate_ccd_pass': candidate_ccd,
        'candidate_admitted': candidate_admitted,
    }


def stage3_run(args):
    """Run one seed/order for baseline, v2 and the proposed stage3 mechanism."""
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda:' + str(args.gpu))
    shape_stats = json.loads(args.shape_stats.read_text(encoding='utf-8'))
    source_mean, source_std = source_style_stats(args.source_manifest)
    needs_adapter = args.method != 'source_only'
    hard_anatomy = args.method == 'v2_adain'
    if needs_adapter:
        adapter, anchor = build_stage3_adapter(
            device, args.ckpt, args.shape_stats, hard_anatomy=hard_anatomy)
    else:
        adapter = None
        anchor = make_model(device, args.ckpt).eval()
    out = args.output
    if out.exists() and any(out.iterdir()):
        raise RuntimeError('refusing to overwrite non-empty output: ' + str(out))
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        'method': args.method, 'seed': args.seed, 'order': args.order,
        'mechanism': ('Delta-Risk Transactional SicTTA: adaptive robust-margin '
                      'candidate audit with commit/soft/reject and exact rollback'
                      if args.method == 'stage3_transactional' else args.method),
        'correction': args.method in ('v2_adain', 'stage3_no_review',
                                      'stage3_transactional'),
        'no_target_labels_for_admission_or_risk': True,
        'fixed_ccd_history': 40, 'fixed_fifo_capacity': 40, 'domains': {},
    }
    for dom in args.order:
        ds = DS(MAN / f'target_domain_{dom.lower()}_stream.csv')
        if args.repeat:
            order_rng = random.Random(args.seed * 1000 + args.repeat * 17 + ord(dom))
            order_rng.shuffle(ds.rows)
        if args.limit:
            ds.rows = ds.rows[:args.limit]
        rows, q_values, risk_deltas = [], [], []
        history = []
        counts = {'commit': 0, 'soft': 0, 'reject': 0, 'skip': 0,
                  'accepted': 0, 'rollback': 0, 'rollback_exact': 0}
        start = time.time()
        for i in range(len(ds)):
            x, y, name = ds[i]; x = x.to(device)
            corrected = align_to_source(x, source_mean, source_std) \
                if summary['correction'] else x
            if args.method == 'source_only':
                output_prob = to_prob(anchor(corrected))
                diag = {'decision': 'source', 'q': 1.0, 'pre_risk': 0.0,
                        'post_risk': 0.0, 'risk_delta': 0.0, 'margin': 0.0,
                        'anatomy_score': 0.0, 'agreement': 1.0,
                        'accepted': 0, 'soft': 0, 'rejected': 0,
                        'rollback': 0, 'rollback_exact': 1}
            elif args.method == 'stage3_transactional':
                output_prob, diag = transactional_step(
                    adapter, anchor, corrected, name, shape_stats, history)
            else:
                pool_before = int(getattr(adapter, 'pool_accept', 0))
                ccd_before = int(getattr(adapter, 'ccd_pass', 0))
                with torch.no_grad():
                    output_prob = to_prob(adapter(corrected, [name]))
                pool_delta = int(getattr(adapter, 'pool_accept', 0)) - pool_before
                ccd_delta = int(getattr(adapter, 'ccd_pass', 0)) - ccd_before
                accepted = int(pool_delta > 0)
                decision = ('commit' if accepted else 'skip') \
                    if args.method == 'stage3_no_review' else 'baseline'
                diag = {'decision': decision, 'q': 1.0, 'pre_risk': 0.0,
                        'post_risk': 0.0, 'risk_delta': 0.0, 'margin': 0.0,
                        'anatomy_score': float(anatomy_features(
                            output_prob, shape_stats)['anatomy_score']),
                        'agreement': 0.0, 'accepted': accepted, 'soft': 0,
                        'rejected': 0, 'rollback': 0, 'rollback_exact': 0,
                        'candidate_ccd_pass': ccd_delta,
                        'candidate_admitted': pool_delta}
            decision = diag['decision']
            if decision == 'commit':
                counts['commit'] += 1
            elif decision == 'soft':
                counts['soft'] += 1
            elif decision == 'reject':
                counts['reject'] += 1
            elif decision == 'skip':
                counts['skip'] += 1
            for key in counts:
                if key in ('commit', 'soft', 'reject'):
                    continue
                counts[key] += int(diag.get(key, 0))
            y_np = y.numpy() if torch.is_tensor(y) else np.asarray(y)
            pred = output_prob.argmax(1)[0].detach().cpu().numpy()
            rec = [name]
            for k in (1, 2):
                aa, bb = pred == k, y_np == k
                rec.extend([float((2 * (aa & bb).sum() + 1e-5) /
                             (aa.sum() + bb.sum() + 1e-5)),
                            capped_assd(aa, bb)])
            rec.extend([diag[k] for k in ('decision', 'q', 'pre_risk',
                                          'post_risk', 'risk_delta', 'margin',
                                          'anatomy_score', 'agreement',
                                          'rollback_exact')])
            rows.append(rec); q_values.append(diag['q']); risk_deltas.append(diag['risk_delta'])
            if (i + 1) % args.report_every == 0:
                vals = np.asarray([r[1:5] for r in rows], dtype=float)
                print(f'STAGE3_PROGRESS method={args.method} seed={args.seed} '
                      f'order={args.order} domain={dom} case={i + 1}/{len(ds)} '
                      f'dice={vals[:, [0, 2]].mean():.4f} '
                      f'decisions={counts}', flush=True)
        arr = np.asarray([r[1:5] for r in rows], dtype=float)
        summary['domains'][dom] = {
            'n': len(rows), 'od_dice': float(arr[:, 0].mean()),
            'od_assd': float(arr[:, 1].mean()), 'oc_dice': float(arr[:, 2].mean()),
            'oc_assd': float(arr[:, 3].mean()),
            'macro_dice': float(arr[:, [0, 2]].mean()),
            'assd': float(arr[:, [1, 3]].mean()),
            'seconds': float(time.time() - start), **counts,
            'q_mean': float(np.mean(q_values)), 'q_std': float(np.std(q_values)),
            'risk_delta_mean': float(np.mean(risk_deltas)),
            'risk_delta_std': float(np.std(risk_deltas)),
            'pool_size_final': (int(adapter.pool.feature_bank.shape[0])
                                if adapter is not None else 0),
        }
        with (out / f'stage3_{args.method}_{dom}_per_case.csv').open(
                'w', newline='', encoding='utf-8') as handle:
            writer = csv.writer(handle)
            writer.writerow(['image', 'od_dice', 'od_assd', 'oc_dice', 'oc_assd',
                             'decision', 'q', 'pre_risk', 'post_risk',
                             'risk_delta', 'margin', 'anatomy_score',
                             'agreement', 'rollback_exact'])
            writer.writerows(rows)
    summary['average'] = {k: float(np.mean([summary['domains'][d][k]
                                            for d in args.order]))
                          for k in ('od_dice', 'od_assd', 'oc_dice', 'oc_assd',
                                    'macro_dice', 'assd')}
    (out / f'stage3_{args.method}_{args.order}_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')
    print('STAGE3_COMPLETE ' + json.dumps(summary['average'], sort_keys=True),
          flush=True)


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--unit-test', action='store_true')
    parser.add_argument('--method', choices=['source_only', 'official_fixed',
                                             'v2_adain', 'stage3_no_review',
                                             'stage3_transactional'])
    parser.add_argument('--ckpt', type=Path)
    parser.add_argument('--shape-stats', type=Path)
    parser.add_argument('--source-manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--order', choices=['CD', 'DC'], default='CD')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--report-every', type=int, default=100)
    args = parser.parse_args()
    if args.unit_test:
        unit_test()
    else:
        for required in ('method', 'ckpt', 'shape_stats', 'source_manifest',
                         'output'):
            if getattr(args, required) is None:
                parser.error(f'--{required.replace("_", "-")} is required')
        stage3_run(args)
