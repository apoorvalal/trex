"""Create compact summaries and figures from saved pilot outputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

p=argparse.ArgumentParser()
p.add_argument('--results',type=Path,required=True)
a=p.parse_args()
r=a.results
rows=pd.DataFrame(json.loads((r/'metrics.json').read_text()))
summary=rows.groupby(['dataset','method']).agg(
    sw_mean=('sliced_wasserstein','mean'),sw_sd=('sliced_wasserstein','std'),
    fit_seconds=('fit_seconds','mean'),sample_seconds=('sample_seconds','mean'),
    selected_k=('k','mean'),nll=('test_nll','mean')).reset_index()
summary.to_json(r/'summary.json',orient='records',indent=2)
print(summary.to_string(index=False))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130})
methods=['held_out','bootstrap','split_adam','em_gmm','wgan','ptgan','diffusion','fresh_target_draws']
labels={'held_out':'Held-out target','bootstrap':'Training-row bootstrap','split_adam':'Split/refine Gaussian mixture','em_gmm':'EM Gaussian mixture','wgan':'WGAN-GP','ptgan':'PTGAN','diffusion':'Diffusion','fresh_target_draws':'Independent target draws'}
for dataset in ['ring','banana']:
    split=np.load(r/f'{dataset}-0-split.npz')
    fig,axes=plt.subplots(2,4,figsize=(14,6.5),sharex=True,sharey=True)
    for ax,method in zip(axes.flat,methods):
        z=split['test'] if method=='held_out' else np.load(r/f'{dataset}-0-{method}.npz')['rows']
        ax.scatter(z[:,0],z[:,1],s=4,alpha=.35,c='#245951',rasterized=True)
        selected=rows[(rows.dataset==dataset)&(rows.seed==0)&(rows.method==method)]
        title=labels[method]
        if len(selected): title+=f'\nSW = {selected.iloc[0].sliced_wasserstein:.3f}'
        ax.set_title(title)
        ax.grid(alpha=.15)
    combined=split['test']
    lo=np.quantile(combined,.002,axis=0)-.5
    hi=np.quantile(combined,.998,axis=0)+.5
    for ax in axes.flat: ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1])
    fig.suptitle(f'{dataset.title()}: held-out comparison, seed 0; shared axes (extreme tails may fall outside)',fontsize=13)
    fig.tight_layout(); fig.savefig(r/f'{dataset}-comparison.png'); plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    mean=split['train'].mean(0); scale=split['train'].std(0)+1e-6
    for ax,k in zip(axes.flat,[1,2,4,8,16,32]):
        state=np.load(r/f'{dataset}-0-splats-k{k}.npz')
        ax.scatter(split['test'][:,0],split['test'][:,1],s=2,c='gray',alpha=.2)
        for w,m,c in zip(state['weights'],state['means'],state['covariances']):
            vals,vecs=np.linalg.eigh(c*scale[:,None]*scale[None,:])
            angle=np.degrees(np.arctan2(vecs[1,-1],vecs[0,-1]))
            # Ellipses show two-standard-deviation component footprints,
            # not a jointly calibrated confidence region of the mixture.
            e=Ellipse(m*scale+mean,4*np.sqrt(vals[-1]),4*np.sqrt(vals[0]),angle=angle,
                      facecolor='#d46748',edgecolor='#923f28',alpha=min(.65,.13+1.5*w))
            ax.add_patch(e)
        ax.set_title(f'{k} components'); ax.set_xlim(lo[0],hi[0]); ax.set_ylim(lo[1],hi[1]); ax.grid(alpha=.15)
    fig.suptitle(f'{dataset.title()}: learned component shapes as capacity grows (seed 0)',fontsize=13)
    fig.tight_layout(); fig.savefig(r/f'{dataset}-capacity.png'); plt.close(fig)

cap=pd.DataFrame(json.loads((r/'capacity.json').read_text()))
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,dataset in zip(axes,['ring','banana']):
    for method in ['split_adam','em_gmm']:
        group=cap[(cap.dataset==dataset)&(cap.method==method)].groupby('k').test_nll.agg(['mean','std'])
        ax.errorbar(group.index,group['mean'],yerr=group['std'],marker='o',label=labels[method],capsize=3)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32])
    ax.set_title(dataset.title());ax.set_xlabel('Components');ax.set_ylabel('Test negative log likelihood');ax.grid(alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Capacity curve: mean ± seed SD; lower is better')
fig.tight_layout();fig.savefig(r/'capacity-nll.png');plt.close(fig)
