#!/usr/bin/env python3
"""Estimate robust anatomy bounds from source labels only."""
import csv, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT=Path('/home/zhaoruijin/MOURUI/sictta_reproduction_20260827')
rows=list(csv.DictReader((ROOT/'manifests/source_train.csv').open(encoding='utf-8')))
vals={k:[] for k in ('cup_disc_ratio','center_distance_norm','disc_lcc_fraction','cup_lcc_fraction')}
for r in rows:
 m=np.asarray(Image.open(r['label']).convert('RGB'),np.uint8)[...,0]
 y=np.zeros(m.shape,np.int64); y[m==128]=1; y[m==0]=2
 disc=y>0; cup=y==2
 if not disc.any() or not cup.any(): continue
 def frac(mask):
  lab,n=ndimage.label(mask)
  return float(np.bincount(lab.ravel())[1:].max()/mask.sum()) if n else 0.
 da=float(disc.sum()); ca=float(cup.sum()); dc=np.asarray(ndimage.center_of_mass(disc)); cc=np.asarray(ndimage.center_of_mass(cup))
 vals['cup_disc_ratio'].append(ca/da)
 vals['center_distance_norm'].append(float(np.linalg.norm(cc-dc)/max(math.sqrt(da/math.pi),1e-6)))
 vals['disc_lcc_fraction'].append(frac(disc)); vals['cup_lcc_fraction'].append(frac(cup))
out={'source_manifest':str(ROOT/'manifests/source_train.csv'),'n_source_rows':len(rows),'n_valid_masks':len(vals['cup_disc_ratio']),'quantiles':{},'admission_bounds':{}}
for k,v in vals.items():
 q=np.percentile(v,[1,99]); out['quantiles'][k]={'q01':float(q[0]),'q99':float(q[1]),'mean':float(np.mean(v)),'std':float(np.std(v))}
 # Wide robust bounds: avoid rejecting normal annotation variation while
 # detecting empty, fragmented, or grossly displaced predictions.
 if k=='cup_disc_ratio': low=max(0.01,float(q[0]-0.05)); high=min(0.95,float(q[1]+0.05))
 elif k=='center_distance_norm': low=0.; high=float(q[1]*1.5+0.05)
 else: low=max(0.5,float(q[0]-0.15)); high=1.0
 out['admission_bounds'][k]={'low':low,'high':high}
path=ROOT/'outputs/anatomy_stats_source_train.json'; path.write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))
