"""Recompute the published LLM vignette from recorded outputs, without models."""
from __future__ import annotations

import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trex import SafetensorsLLMInContextGenerator, distribution_metrics

LABELS={"independent_target":"Independent target draw","resample":"Empirical resampling",
        "frozen":"Frozen Qwen (ICL)","qlora":"Qwen + QLoRA"}


def load_saved(path):
    record=json.loads(gzip.decompress(Path(path).read_bytes()))
    files=record["files"]
    for name,digest in record["sha256"].items():
        assert hashlib.sha256(files[name].encode()).hexdigest()==digest,name
    arrays={name:np.loadtxt(StringIO(files[name+".csv"]),delimiter=",",skiprows=1)
            for name in ("train","test",*LABELS)}
    meta=json.loads(files["metadata.json"])
    training=json.loads(files["training.json"])
    state=json.loads(files["trainer_state.json"])
    assert arrays["train"].shape==(800,3) and arrays["test"].shape==(512,3)
    assert state["global_step"]==64
    assert training["adapter_parameters"]>0 and training["target_tokens_min"]>0
    runs={}
    parser=SafetensorsLLMInContextGenerator(model_path=meta["model"],device="cpu").fit(
        arrays["train"],column_names=["age","employed","income"])
    for name in ("frozen","qlora"):
        run=json.loads(files[name+"-run.json"])
        trace=json.loads(files[name+"-trace.json"])
        parsed=[row for attempt in trace for row in parser._parse_rows(attempt["completion"])]
        assert len(trace)==run["attempts"]
        assert len(parsed)==run["parsed_before_final_truncation"]
        allowed={",".join(f"{v:.6g}" for v in row) for row in arrays["train"]}
        for attempt in trace:
            examples=attempt["prompt"].split("Examples:\n",1)[1].split("\nGenerate exactly",1)[0].splitlines()
            assert len(examples)==8 and set(examples)<=allowed
            assert len(parser._parse_rows(attempt["completion"]))==attempt["parsed_rows"]
        assert arrays[name].shape==(256,3)
        np.testing.assert_allclose(np.array(parsed[:256]),arrays[name],rtol=5.1e-10,atol=1e-12)
        assert run["model"]["quantization"]["bnb_4bit_quant_type"]=="nf4"
        assert run["model"]["quantization"]["bnb_4bit_compute_dtype"]=="bfloat16"
        run["zero_row_attempts"]=sum(attempt["parsed_rows"]==0 for attempt in trace)
        runs[name]=run
    return arrays,meta,training,state,runs


def evaluate(arrays):
    train,test=arrays["train"],arrays["test"]
    mean,scale=train.mean(0),train.std(0)
    train_set={tuple(row) for row in train}
    output={}
    for name in LABELS:
        raw=arrays[name]
        finite=np.isfinite(raw).all(1)
        x=raw[finite]
        valid=((x[:,0]>=18)&(x[:,0]<=65)&(x[:,0]==np.floor(x[:,0]))&
               np.isin(x[:,1],[0,1])&(x[:,2]>=0)&
               np.where(x[:,1]==0,x[:,2]==0,x[:,2]>0))
        employed=x[:,1]==1
        copied=np.array([tuple(row) in train_set for row in x])
        scores=distribution_metrics((test-mean)/scale,(x-mean)/scale,n_projections=128,seed=91)
        output[name]={**scores,"rows":len(raw),"nonfinite":int((~finite).sum()),
                      "support_invalid":int((~valid).sum()),"support_invalid_pct":100*float((~valid).mean()),
                      "employment_share":float((x[:,1]==1).mean()),"income_mean":float(x[:,2].mean()),
                      "age_mean":float(x[:,0].mean()),"unique_rows":len(np.unique(x,axis=0)),
                      "train_copy_pct":100*float(copied.mean()),
                      "employed_train_copy_pct":100*float(copied[employed].mean()) if employed.any() else None}
    return output


def scores_table(scores):
    records=[]
    for name,label in LABELS.items():
        q=scores[name]
        records.append({"Generator":label,"Sliced W1":q["sliced_wasserstein"],
                        "Marginal W1":q["marginal_w1_mean"],"Correlation error":q["corr_frobenius"],
                        "Invalid support (%)":q["support_invalid_pct"],
                        "Employed-row copies (%)":q["employed_train_copy_pct"]})
    return pd.DataFrame(records).set_index("Generator").round(3)


def summary_table(arrays):
    return pd.DataFrame({label:{"Mean age":arr[:,0].mean(),"Employed (%)":100*(arr[:,1]==1).mean(),
                               "Mean income (thousands)":arr[:,2].mean()}
                         for label,arr in [("Held-out target",arrays["test"]),
                                           *[(LABELS[k],arrays[k]) for k in LABELS]]}).T.round(2)


def sample_plot(arrays):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.3),sharex=True,sharey=True)
    for ax,key,title in zip(axes,["test","frozen","qlora"],["Held-out target","Frozen Qwen","Qwen + QLoRA"]):
        x=arrays[key]
        ax.scatter(x[:,0],x[:,2],s=10,alpha=.42,c=np.where(x[:,1]==1,"#187b73","#b85244"),edgecolors="none")
        ax.set(title=title,xlabel="Age")
    axes[0].set_ylabel("Annual income (thousands)")
    fig.tight_layout()
    return fig


def income_plot(arrays):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.3))
    for name,label,color in [("test","Held-out target","#222222"),
                              ("resample","Empirical resampling","#73848c"),
                              ("frozen","Frozen Qwen","#c35a3d"),("qlora","Qwen + QLoRA","#187b73")]:
        x=arrays[name]
        income=np.sort(x[:,2])
        axes[0].step(income,np.arange(1,len(income)+1)/len(income),where="post",label=label,color=color)
        age=np.sort(x[:,0])
        axes[1].step(age,np.arange(1,len(age)+1)/len(age),where="post",label=label,color=color)
    axes[0].set(xlabel="Annual income (thousands)",ylabel="Empirical CDF")
    axes[1].set(xlabel="Age",ylabel="Empirical CDF")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def loss_plot(state):
    import matplotlib.pyplot as plt
    points=[row for row in state["log_history"] if "loss" in row]
    fig,ax=plt.subplots(figsize=(7,2.8))
    ax.plot([int(row["step"]) for row in points],[float(row["loss"]) for row in points],marker="o",color="#187b73")
    ax.set(xlabel="Optimizer step",ylabel="Training token loss",title="Completion-token training loss (logged windows)")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, default=Path(__file__).resolve().parents[2] /
                        "docs/guides/data/llm-generators/2026-09-27/executed.json.gz")
    args = parser.parse_args()
    arrays, metadata, training, state, runs = load_saved(args.record)
    scores = evaluate(arrays)
    print(scores_table(scores).to_string())
    print(summary_table(arrays).to_string())
    print("Verified all saved file hashes, 100 generation attempts, and 512 returned rows.")
