"""Held-out density-splat pilot against EM-GMM and merged Trex generators."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import torch
from model import GaussianSplats


def load_reference(path):
    # On a remote checkout with independent work, use an exact source snapshot
    # rather than switching its branch or replacing its installed Trex.
    package = types.ModuleType('_trex_pilot_reference')
    package.__path__ = [str(path)]
    sys.modules[package.__name__] = package
    for name in ['base', 'simdgp']:
        spec = importlib.util.spec_from_file_location(package.__name__ + '.' + name, path / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def toy(name, seed, n):
    rng = np.random.default_rng(seed)
    if name == 'ring':
        angle = rng.integers(8, size=n)*2*np.pi/8
        return 1.5*np.c_[np.cos(angle), np.sin(angle)] + rng.normal(0, .1, (n, 2))
    u = rng.normal(size=n)
    return np.c_[u, .7*(u*u-1)+rng.normal(0, .2, n)]


def sync(device):
    if device == 'cuda': torch.cuda.synchronize()


def main(args):
    torch.set_num_threads(2)
    ref = load_reference(args.trex_source)
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    config['reference_source_hashes'] = {name: hashlib.sha256((args.trex_source/name).read_bytes()).hexdigest() for name in ['base.py', 'simdgp.py']}
    config['torch_version'] = torch.__version__
    config['device_name'] = torch.cuda.get_device_name() if args.device == 'cuda' else 'CPU'
    (args.out/'config.json').write_text(json.dumps(config, indent=2))
    rows = []
    capacity = []
    for dataset in args.datasets:
        for seed in range(args.seeds):
            if dataset == 'lalonde':
                columns = ['t','age','education','black','hispanic','married','nodegree','re74','re75','re78']
                data = pd.read_feather(args.paper_repo/'data/original_data/exp_merged.feather')[columns].to_numpy(float)
                train, remainder = train_test_split(data, test_size=.4, random_state=seed, stratify=data[:,0])
                val, test = train_test_split(remainder, test_size=.5, random_state=seed+50, stratify=remainder[:,0])
                transformer = ref.TabularTransformer(column_names=columns, binary_columns=['t','black','hispanic','married','nodegree'], nonnegative_columns=['age','education','re74','re75','re78']).fit(train)
            else:
                train, val, test = [toy(dataset, seed+offset, n) for offset,n in [(0,2048),(100,1024),(200,1024)]]
                transformer = ref.TabularTransformer().fit(train)
            train_z, val_z, test_z = [transformer.transform(x) for x in [train,val,test]]
            np.savez_compressed(args.out/f'{dataset}-{seed}-split.npz', train=train, validation=val, test=test)

            def score(fake_raw):
                # Trex's current SW helper truncates unequal sample sizes. All
                # comparisons here intentionally have exactly len(test) rows.
                metrics = ref.distribution_metrics(test_z, transformer.transform(fake_raw), n_projections=128, seed=7919)
                if dataset == 'ring':
                    angle=np.arange(8)*2*np.pi/8
                    centers=1.5*np.c_[np.cos(angle),np.sin(angle)]
                    distances=np.linalg.norm(fake_raw[:,None]-centers,axis=2)
                    labels=distances.argmin(1)
                    counts=np.bincount(labels[distances.min(1)<.3], minlength=8)
                    metrics['mode_coverage']=int((counts >= 5).sum())
                    metrics['off_mode_fraction']=float((distances.min(1)>=.3).mean())
                if dataset == 'lalonde':
                    metrics['re78_mean_abs_error']=float(abs(fake_raw[:,9].mean()-test[:,9].mean()))
                    metrics['earnings_zero_rate_mae']=float(np.abs((fake_raw[:,7:10]==0).mean(0)-(test[:,7:10]==0).mean(0)).mean())
                    def contrast(a):
                        return np.linalg.lstsq(np.c_[np.ones(len(a)),a[:,:7]],a[:,9],rcond=None)[0][1]
                    metrics['adjusted_association_abs_error']=float(abs(contrast(fake_raw)-contrast(test)))
                return metrics

            def record(method, fake_raw, seconds, sample_seconds, **kwargs):
                row=dict(dataset=dataset,seed=seed,method=method,n_train=len(train),n_validation=len(val),n_test=len(test),fit_seconds=seconds,sample_seconds=sample_seconds,**kwargs,**score(fake_raw))
                rows.append(row)
                np.savez_compressed(args.out/f'{dataset}-{seed}-{method}.npz',rows=fake_raw)
                (args.out/'metrics.json').write_text(json.dumps(rows,indent=2))
                print(f'{dataset} seed={seed} {method}: SW={row["sliced_wasserstein"]:.4f}, fit={seconds:.2f}s', flush=True)

            rng=np.random.default_rng(seed+300)
            start=time.perf_counter()
            fake=train[rng.integers(len(train),size=len(test))]
            record('bootstrap',fake,0.,time.perf_counter()-start)
            if dataset != 'lalonde':
                record('fresh_target_draws',toy(dataset,seed+400,len(test)),0.,0.)

            # Validation chooses both capacity and the Adam checkpoint. The test
            # split is used only for reporting; no tuning to its outcome.
            def selection(z, nll):
                if dataset != 'lalonde': return float(nll)
                return float(ref.distribution_metrics(val_z, transformer.transform(transformer.inverse_transform(z)),seed=123)['sliced_wasserstein'])

            model=GaussianSplats(variance_floor=args.variance_floor,seed=seed,device=args.device,dtype=torch.float32).initialize(train_z)
            best_adam=None; best_em=None; total_adam=0.; total_em=0.
            for k in args.counts:
                if model.n_components != k:
                    while model.n_components < k: model.split()
                sync(args.device); start=time.perf_counter()
                # Mixed-data checkpoint selection uses training proxy NLL;
                # cross-K selection uses validation sample discrepancy, not a
                # spurious mixed-data continuous likelihood.
                model.refine(train_z,validation=val_z if dataset!='lalonde' else None,steps=args.steps)
                sync(args.device); seconds=time.perf_counter()-start; total_adam+=seconds
                with torch.no_grad():
                    val_nll=float(-model.log_prob(val_z).mean())
                    test_nll=float(-model.log_prob(test_z).mean()) if dataset!='lalonde' else None
                sample_z=model.sample(len(val),seed=seed+500).numpy()
                select=selection(sample_z,val_nll)
                state=model.snapshot()
                np.savez_compressed(args.out/f'{dataset}-{seed}-splats-k{k}.npz',**state)
                capacity.append(dict(dataset=dataset,seed=seed,method='split_adam',k=k,selection=select,test_nll=test_nll,cumulative_seconds=total_adam))
                if best_adam is None or select < best_adam['selection']:
                    best_adam=dict(selection=select,k=k,test_nll=test_nll,state={key:value.detach().clone() for key,value in model.state_dict().items()})
                start=time.perf_counter()
                em=GaussianMixture(k,covariance_type='full',reg_covar=args.variance_floor,n_init=3,max_iter=300,random_state=seed).fit(train_z.astype(np.float64))
                total_em+=time.perf_counter()-start
                # sklearn sample has its own deterministic random state.
                select_em=selection(em.sample(len(val))[0],-em.score(val_z))
                em_test=-float(em.score(test_z)) if dataset!='lalonde' else None
                capacity.append(dict(dataset=dataset,seed=seed,method='em_gmm',k=k,selection=select_em,test_nll=em_test,cumulative_seconds=total_em,converged=bool(em.converged_)))
                if best_em is None or select_em < best_em['selection']:
                    best_em=dict(selection=select_em,k=k,test_nll=em_test,model=em)
            chosen=GaussianSplats(variance_floor=args.variance_floor,seed=seed,device=args.device,dtype=torch.float32)
            chosen._set_components(torch.ones(best_adam['k'],device=args.device)/best_adam['k'],best_adam['state']['means'],torch.eye(train.shape[1],device=args.device).repeat(best_adam['k'],1,1))
            chosen.load_state_dict(best_adam['state'])
            sync(args.device); start=time.perf_counter()
            fake=transformer.inverse_transform(chosen.sample(len(test),seed=seed+600).numpy())
            sync(args.device); sample_s=time.perf_counter()-start
            record('split_adam',fake,total_adam,sample_s,k=best_adam['k'],test_nll=best_adam['test_nll'])
            em=best_em['model']; em.random_state=seed+600
            start=time.perf_counter(); fake=transformer.inverse_transform(em.sample(len(test))[0]); sample_s=time.perf_counter()-start
            record('em_gmm',fake,total_em,sample_s,k=best_em['k'],test_nll=best_em['test_nll'])
            (args.out/'capacity.json').write_text(json.dumps(capacity,indent=2))

            for method in args.neural:
                common=dict(hidden_dims=(64,64),batch_size=128,max_steps=args.neural_steps,seed=seed,device=args.device)
                if method=='diffusion':
                    neural=ref.TabularDiffusion(n_timesteps=1000,lr=.001,**common)
                else:
                    lower,upper=transformer.transformed_bounds()
                    extra=dict(critic_hidden_dims=(64,64),critic_steps=1,lr=.0001,betas=(0.,.9),generator_dropout=0.,binary_dims=transformer.binary_indices,lower_bounds=lower,upper_bounds=upper)
                    neural=(ref.TabularWGAN(gp_weight=5.,**extra,**common) if method=='wgan' else ref.TabularPTGAN(temperature_ratio=.9,coherency_weight=100.,gp_weight=0.,**extra,**common))
                sync(args.device); start=time.perf_counter(); neural.fit(train_z); sync(args.device); seconds=time.perf_counter()-start
                start=time.perf_counter(); fake=transformer.inverse_transform(neural.sample(len(test)).numpy()); sync(args.device); sample_s=time.perf_counter()-start
                record(method,fake,seconds,sample_s)
    print('DONE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trex-source',type=Path,default=Path('trex'))
    p.add_argument('--paper-repo',type=Path,default=Path('../dswgan-paper'))
    p.add_argument('--out',type=Path,default=Path('tmp/gaussian-tabular'))
    p.add_argument('--datasets',nargs='+',default=['ring','banana','lalonde'])
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--counts',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--neural-steps',type=int,default=1500)
    p.add_argument('--neural',nargs='*',default=['wgan','ptgan','diffusion'])
    p.add_argument('--variance-floor',type=float,default=.0025)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if args.counts != [2**j for j in range(len(args.counts))]: p.error('counts must be consecutive powers of two starting at 1')
    main(args)
