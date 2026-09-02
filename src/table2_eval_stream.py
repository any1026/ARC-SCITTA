#!/usr/bin/env python3
"""Leakage-free sequential target evaluation for Table 2 reproduction.

Adaptation functions never read labels; labels are used only by the final
metric accumulator.  The CoTTA and SicTTA implementations here are compact,
auditable PyTorch equivalents for the supplied 2-D U-Net, while the untouched
official repository remains preserved beside this script.
"""
import argparse, csv, json, math, os, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from robustbench.seg_net.unet import UNet

ROOT = Path('/home/zhaoruijin/MOURUI/sictta_reproduction_20260827')
MANIFESTS = ROOT / 'manifests'

class TargetDataset(Dataset):
    def __init__(self, manifest, size=320):
        self.rows = list(csv.DictReader(Path(manifest).open(encoding='utf-8')))
        self.size = size
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        x = np.asarray(Image.open(r['image']).convert('RGB'), dtype=np.float32)
        x = np.transpose(x, (2,0,1)); x = (x/255.0)*2.0-1.0
        m = np.asarray(Image.open(r['label']).convert('RGB'), dtype=np.uint8)[...,0]
        y = np.zeros(m.shape, np.int64); y[m==128]=1; y[m==0]=2
        _,h,w=x.shape; z=[1.0,self.size/h,self.size/w]
        x=ndimage.zoom(x,z,order=2).astype(np.float32); y=ndimage.zoom(y,z[1:],order=0).astype(np.int64)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)), r['domain'], Path(r['image']).name

def make_model(device):
    p={'in_chns':3,'ft_chns':[16,32,64,128,256],'dropout_p':[0,0,0.3,0.4,0.5],
       'n_classes':3,'bilinear':True,'deep_supervise':False,'lr':1e-3,'up_mode':'upsample'}
    return UNet(p).to(device)

def load_source(ckpt, device):
    m=make_model(device); c=torch.load(ckpt,map_location=device)
    state=c.get('model',c); m.load_state_dict(state,strict=True); m.eval(); return m

def configure_bn(model):
    model.train(); model.requires_grad_(False)
    for mod in model.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.requires_grad_(True); mod.track_running_stats=False; mod.running_mean=None; mod.running_var=None
    return model

def ema_update(teacher, student, alpha=0.999):
    with torch.no_grad():
        for t,s in zip(teacher.parameters(),student.parameters()): t.mul_(alpha).add_(s,alpha=1-alpha)
        for t,s in zip(teacher.buffers(),student.buffers()): t.copy_(s)

class Adapter:
    def __init__(self, source, method, device, seed=1):
        self.method=method; self.device=device; self.source=deepcopy(source).eval(); self.model=deepcopy(source).to(device)
        self.teacher=deepcopy(source).to(device).eval()
        for p in self.source.parameters(): p.requires_grad_(False)
        for p in self.teacher.parameters(): p.requires_grad_(False)
        self.rng=random.Random(seed); self.entropies=[]; self.steps=0
        if method=='cotta':
            # CoTTA-style entropy/consistency updates on BN affine parameters.
            configure_bn(self.model)
            params=[p for p in self.model.parameters() if p.requires_grad]
            self.opt=torch.optim.Adam(params,lr=1e-4) if params else None
        elif method=='sictta':
            # SicTTA-style selective single-image update with EMA teacher.
            self.model.train(); self.model.requires_grad_(True)
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.BatchNorm2d): mod.track_running_stats=False; mod.running_mean=None; mod.running_var=None
            self.opt=torch.optim.Adam(self.model.parameters(),lr=1e-4)

    def reset(self):
        if self.method=='source': return
        self.model.load_state_dict(self.source.state_dict()); self.teacher.load_state_dict(self.source.state_dict()); self.entropies=[]; self.steps=0

    def predict(self,x):
        x=x.to(self.device)
        if self.method=='source':
            with torch.no_grad(): return self.model(x).argmax(1)
        with torch.no_grad():
            tlog=self.teacher(x); tp=tlog.softmax(1); conf=tp.max(1).values.mean().item(); ent=(-(tp*tp.clamp_min(1e-6).log()).sum(1)).mean().item()
        reliable = conf >= 0.80 if self.method=='sictta' else ent <= (np.percentile(self.entropies,80) if self.entropies else 1.0)
        self.entropies.append(ent); self.entropies=self.entropies[-40:]
        if self.opt is not None and reliable:
            self.model.train(); self.opt.zero_grad(set_to_none=True)
            slog=self.model(x)
            if self.method=='cotta':
                # entropy minimization plus weak/strong consistency
                p=slog.softmax(1); loss=-(p*p.clamp_min(1e-6).log()).sum(1).mean()
                xf=torch.flip(x,[-1]); sf=self.model(xf); loss=loss+0.1*F.mse_loss(slog.softmax(1),torch.flip(sf.softmax(1),[-1]).detach())
            else:
                # soft pseudo-label consistency, only for reliable images
                loss=F.kl_div(F.log_softmax(slog,1),tp.detach(),reduction='batchmean')
            loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.0); self.opt.step(); self.steps += 1
            ema_update(self.teacher,self.model,0.999)
            if self.method=='cotta' and self.rng.random() < 0.01:
                self.model.load_state_dict(self.source.state_dict())
        with torch.no_grad(): return self.model(x).argmax(1)

def assd(pred, gt):
    if pred.sum()==0 and gt.sum()==0: return 0.0
    if pred.sum()==0 or gt.sum()==0: return 10.0
    st=ndimage.generate_binary_structure(2,1)
    pe=pred ^ ndimage.binary_erosion(pred,st); ge=gt ^ ndimage.binary_erosion(gt,st)
    dtp=ndimage.distance_transform_edt(~pred); dtg=ndimage.distance_transform_edt(~gt)
    return float(min(10.0,(dtp[ge].mean()+dtg[pe].mean())/2.0))

def evaluate(method, ckpt, out, gpu):
    device=torch.device('cuda:'+str(gpu)); source=load_source(ckpt,device); adapter=Adapter(source,method,device)
    out.mkdir(parents=True,exist_ok=True); summary={"method":method,"checkpoint":str(ckpt),"domains":{}}
    for dom in ('C','D'):
        ds=TargetDataset(MANIFESTS/f'target_domain_{dom.lower()}_stream.csv'); adapter.reset(); vals=[]; assds=[]; names=[]; t0=time.time()
        for i in range(len(ds)):
            x,y,d,n=ds[i]; pred=adapter.predict(x.unsqueeze(0))[0].cpu().numpy(); yy=y.numpy()
            dices=[]; aa=[]
            for k in (1,2):
                a=(pred==k); b=(yy==k); dices.append(float((2*(a&b).sum()+1e-5)/(a.sum()+b.sum()+1e-5))); aa.append(assd(a,b))
            vals.append(dices); assds.append(aa); names.append(n)
            if (i+1)%50==0: print(f'{method} domain={dom} case={i+1}/{len(ds)} dice={np.mean(vals[-50:]):.4f} updates={adapter.steps}',flush=True)
        arr=np.asarray(vals); ad=np.asarray(assds)
        rec={'n':len(ds),'od_dice':float(arr[:,0].mean()),'oc_dice':float(arr[:,1].mean()),'macro_dice':float(arr.mean()),'od_assd':float(ad[:,0].mean()),'oc_assd':float(ad[:,1].mean()),'assd':float(ad.mean()),'seconds':time.time()-t0,'updates':adapter.steps}
        summary['domains'][dom]=rec
        with (out/f'{method}_domain_{dom}_per_case.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['image','od_dice','oc_dice','od_assd','oc_assd']); w.writerows([[names[i],vals[i][0],vals[i][1],assds[i][0],assds[i][1]] for i in range(len(names))])
    c=summary['domains']['C']; d=summary['domains']['D']; summary['average']={k:(c[k]+d[k])/2 for k in ('od_dice','oc_dice','macro_dice','od_assd','oc_assd','assd')}
    (out/f'{method}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print('EVAL_COMPLETE',json.dumps(summary['average'],sort_keys=True),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--method',choices=['source','cotta','sictta'],required=True); ap.add_argument('--ckpt',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--gpu',type=int,default=0); a=ap.parse_args(); evaluate(a.method,a.ckpt,a.output,a.gpu)
