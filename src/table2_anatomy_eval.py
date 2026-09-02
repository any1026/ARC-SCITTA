#!/usr/bin/env python3
"""Evaluate official SicTTA or the full-code anatomy-gated derivative."""
import argparse, csv, json, random, time
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
from robustbench.seg_net.unet import UNet
from sotas import sictta as official
import sictta_anatomy_v1 as anatomy
from table2_eval_faithful import DS, assd, MAN

def make_model(device, ckpt):
 p={'in_chns':3,'ft_chns':[16,32,64,128,256],'dropout_p':[0,0,.3,.4,.5],'n_classes':3,'bilinear':True,'deep_supervise':False,'lr':1e-3,'up_mode':'upsample'}
 m=UNet(p).to(device); c=torch.load(ckpt,map_location=device); m.load_state_dict(c['model']); return m

def run(args):
 random.seed(1); np.random.seed(1); torch.manual_seed(1)
 dev=torch.device('cuda:'+str(args.gpu)); src=make_model(dev,args.ckpt); anchor=deepcopy(src).eval()
 if args.method=='official': adapter=official.TTA(official.configure_model(src),anchor)
 else: adapter=anatomy.TTA(anatomy.configure_model(src),anchor,args.shape_stats)
 out=args.output; out.mkdir(parents=True,exist_ok=True); domains=args.order
 summary={'method':args.method,'order':domains,'no_target_labels_for_admission':True,'domains':{}}
 for dom in domains:
  ds=DS(MAN/f'target_domain_{dom.lower()}_stream.csv'); rows=[]; t=time.time()
  for i in range(len(ds)):
   x,y,n=ds[i]
   y_np=y.numpy() if torch.is_tensor(y) else np.asarray(y)
   with torch.no_grad(): z=adapter(x.to(dev),[n])
   pred=z.argmax(1)[0].cpu().numpy(); rr=[]
   for k in (1,2):
    a=pred==k; b=y_np==k; rr.extend([float((2*(a&b).sum()+1e-5)/(a.sum()+b.sum()+1e-5)),assd(a,b)])
   rows.append([n,*rr])
   if (i+1)%50==0: print(f'{args.method} order={domains} domain={dom} case={i+1}/400 dice={np.mean(np.asarray(rows)[:,-4::2].astype(float)):.4f}',flush=True)
  arr=np.asarray([r[1:] for r in rows],float); rec={'n':len(rows),'od_dice':arr[:,0].mean(),'od_assd':arr[:,1].mean(),'oc_dice':arr[:,2].mean(),'oc_assd':arr[:,3].mean(),'macro_dice':arr[:,[0,2]].mean(),'assd':arr[:,[1,3]].mean(),'seconds':time.time()-t}
  if hasattr(adapter,'diagnostics'): rec['diagnostics']=adapter.diagnostics()
  summary['domains'][dom]={k:(float(v) if isinstance(v,(float,np.floating)) else v) for k,v in rec.items()}
  with (out/f'{args.method}_{dom}_per_case.csv').open('w',newline='',encoding='utf-8') as f: w=csv.writer(f); w.writerow(['image','od_dice','od_assd','oc_dice','oc_assd']); w.writerows(rows)
 summary['average']={k:float(np.mean([summary['domains'][d][k] for d in domains])) for k in ('od_dice','od_assd','oc_dice','oc_assd','macro_dice','assd')}
 (out/f'{args.method}_{domains}_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print('ANATOMY_EVAL_COMPLETE',json.dumps(summary['average'],sort_keys=True),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--method',choices=['official','anatomy'],required=True); p.add_argument('--order',default='CD',choices=['CD','DC']); p.add_argument('--ckpt',type=Path,required=True); p.add_argument('--shape-stats',type=Path,default=Path('/home/zhaoruijin/MOURUI/sictta_reproduction_20260827/outputs/anatomy_stats_source_train.json')); p.add_argument('--output',type=Path,required=True); p.add_argument('--gpu',type=int,default=0); a=p.parse_args(); run(a)
