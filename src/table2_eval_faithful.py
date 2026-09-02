#!/usr/bin/env python3
"""Table 2 C->D evaluation using official SicTTA and a CoTTA segmentation port."""
import argparse, csv, json, os, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from robustbench.seg_net.unet import UNet
from sotas import sictta as official_sictta

ROOT=Path('/home/zhaoruijin/MOURUI/sictta_reproduction_20260827')
MAN=ROOT/'manifests'

class DS:
 def __init__(self,p): self.rows=list(csv.DictReader(Path(p).open(encoding='utf-8')))
 def __len__(self): return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]; x=np.asarray(Image.open(r['image']).convert('RGB'),np.float32).transpose(2,0,1); x=(x/255)*2-1
  m=np.asarray(Image.open(r['label']).convert('RGB'),np.uint8)[...,0]; y=np.zeros(m.shape,np.int64); y[m==128]=1; y[m==0]=2
  _,h,w=x.shape; x=ndimage.zoom(x,[1,320/h,320/w],order=2).astype(np.float32); y=ndimage.zoom(y,[320/h,320/w],order=0).astype(np.int64)
  return torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0),y,Path(r['image']).name

def model(device,ckpt):
 p={'in_chns':3,'ft_chns':[16,32,64,128,256],'dropout_p':[0,0,.3,.4,.5],'n_classes':3,'bilinear':True,'deep_supervise':False,'lr':1e-3,'up_mode':'upsample'}
 m=UNet(p).to(device); c=torch.load(ckpt,map_location=device); m.load_state_dict(c['model']); return m

def configure_cotta(m):
 m.train(); m.requires_grad_(True)
 for x in m.modules():
  if isinstance(x,nn.BatchNorm2d): x.track_running_stats=False; x.running_mean=None; x.running_var=None
 return m

def photometric(x):
 # Geometry-free augmentation preserves pixel correspondence for segmentation.
 z=(x+1)/2
 z=TF.adjust_brightness(z,random.uniform(.6,1.4)); z=TF.adjust_contrast(z,random.uniform(.7,1.3)); z=TF.adjust_saturation(z,random.uniform(.5,1.5)); z=TF.adjust_gamma(z.clamp(0,1),random.uniform(.7,1.3)); z=(z+torch.randn_like(z)*.005).clamp(0,1)
 return z*2-1

class CoTTA:
 def __init__(self,src,lr=1e-4,restore=0.01,ap=0.92,symmetric=False):
  self.student=configure_cotta(deepcopy(src)); self.anchor=deepcopy(self.student); self.teacher=deepcopy(self.student)
  self.anchor.requires_grad_(False); self.teacher.requires_grad_(False); self.teacher.train(); self.anchor.train()
  self.opt=torch.optim.Adam(self.student.parameters(),lr=lr); self.source_state=deepcopy(self.student.state_dict()); self.updates=0; self.restore=restore; self.ap=ap; self.symmetric=symmetric
 def __call__(self,x):
  out=self.student(x)
  with torch.no_grad():
   anchor_conf=self.anchor(x).softmax(1).max(1).values.mean()
   standard=self.teacher(x)
   if anchor_conf < self.ap: target=torch.stack([self.teacher(photometric(x)) for _ in range(32)]).mean(0)
   else: target=standard
  loss=-(target.softmax(1)*out.log_softmax(1)).sum(1).mean()
  if self.symmetric: loss=0.5*loss-0.5*(out.softmax(1)*target.log_softmax(1)).sum(1).mean()
  self.opt.zero_grad(set_to_none=True); loss.backward(); self.opt.step()
  with torch.no_grad():
   for t,s in zip(self.teacher.parameters(),self.student.parameters()): t.mul_(.999).add_(s,alpha=.001)
   current=self.student.state_dict()
   for name,p in self.student.named_parameters():
    mask=torch.rand_like(p)<self.restore; p.copy_(torch.where(mask,self.source_state[name],p))
  self.updates+=1; return target

def make_adapter(method,src,args):
 if method=='sictta':
  anchor=deepcopy(src).eval(); adapted=official_sictta.configure_model(src); return official_sictta.TTA(adapted,anchor)
 if method=='cotta': return CoTTA(src,lr=args.lr,restore=args.restore,ap=args.ap,symmetric=args.symmetric)
 raise ValueError(method)

def assd(a,b):
 if not a.any() and not b.any(): return 0.
 if not a.any() or not b.any(): return 10.
 st=ndimage.generate_binary_structure(2,1); ae=a^ndimage.binary_erosion(a,st); be=b^ndimage.binary_erosion(b,st); da=ndimage.distance_transform_edt(~a); db=ndimage.distance_transform_edt(~b)
 return float(min(10.,(da[be].mean()+db[ae].mean())/2))

def run(a):
 device=torch.device('cuda:'+str(a.gpu)); torch.manual_seed(1); random.seed(1); np.random.seed(1)
 src=model(device,a.ckpt); adapter=make_adapter(a.method,src,a); a.output.mkdir(parents=True,exist_ok=True)
 summary={'method':a.method,'protocol':'single continuous C->D stream; no reset between domains','implementation':'official repository sotas/sictta.py' if a.method=='sictta' else f'CoTTA port: EMA=.999, restore={a.restore}, AP={a.ap}, LR={a.lr}, symmetric={a.symmetric}, N=32, photometric alignment-safe augmentation','domains':{}}
 total_updates=0
 for dom in ['C','D']:
  ds=DS(MAN/f'target_domain_{dom.lower()}_stream.csv'); rows=[]; t=time.time()
  for i in range(len(ds)):
   x,y,n=ds[i]; x=x.to(device)
   with torch.no_grad() if a.method=='sictta' else torch.enable_grad(): out=adapter(x,[n]) if a.method=='sictta' else adapter(x)
   p=out.argmax(1)[0].detach().cpu().numpy(); vals=[]
   for k in [1,2]:
    aa=p==k; bb=y==k; dice=float((2*(aa&bb).sum()+1e-5)/(aa.sum()+bb.sum()+1e-5)); vals.extend([dice,assd(aa,bb)])
   rows.append([n,*vals])
   if (i+1)%50==0: print(f'{a.method} domain={dom} case={i+1}/400 dice={np.mean(np.asarray(rows)[:,-4::2].astype(float)):.4f}',flush=True)
  arr=np.asarray([r[1:] for r in rows],float); rec={'n':len(rows),'od_dice':arr[:,0].mean(),'od_assd':arr[:,1].mean(),'oc_dice':arr[:,2].mean(),'oc_assd':arr[:,3].mean(),'macro_dice':arr[:,[0,2]].mean(),'assd':arr[:,[1,3]].mean(),'seconds':time.time()-t}
  summary['domains'][dom]={k:float(v) for k,v in rec.items()};
  with (a.output/f'{a.method}_domain_{dom}_per_case.csv').open('w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(['image','od_dice','od_assd','oc_dice','oc_assd']); w.writerows(rows)
 c,d=summary['domains']['C'],summary['domains']['D']; summary['average']={k:(c[k]+d[k])/2 for k in ['od_dice','od_assd','oc_dice','oc_assd','macro_dice','assd']}; (a.output/f'{a.method}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print('EVAL_COMPLETE',json.dumps(summary['average']),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--method',choices=['cotta','sictta'],required=True); p.add_argument('--ckpt',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',type=int,required=True); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--restore',type=float,default=.01); p.add_argument('--ap',type=float,default=.92); p.add_argument('--symmetric',action='store_true'); run(p.parse_args())
