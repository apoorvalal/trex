#!/usr/bin/env python3
"""Run the actual optional Trex wrappers; save rows, prompts and training state."""
from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
os.environ.setdefault("WANDB_DISABLED","true")
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from trex import SafetensorsLLMInContextGenerator, SafetensorsQLORAGenerator

COLUMNS=["age","employed","income"]
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
REVISION="7ae557604adf67be50417f59c2c2f167def9a775"


def draw(rng,n):
    age=rng.integers(18,66,size=n)
    employed=rng.binomial(1,1/(1+np.exp(-(-.2+.035*(age-35)))))
    income=employed*np.round(np.exp(3.2+.025*(age-35)+.35*rng.normal(size=n)),2)
    return np.column_stack([age,employed,income]).astype(float)


def make_data(out):
    rng=np.random.default_rng(20260927)
    for name,n in (("train",800),("test",512),("independent_target",256)):
        np.savetxt(out/(name+".csv"),draw(rng,n),delimiter=",",header=",".join(COLUMNS),comments="",fmt="%.10g")


class AuditMixin:
    """Observe the documented API without changing prompts, loading or parsing."""
    def _prompt(self,rows_requested):
        prompt=super()._prompt(rows_requested)
        self.current_prompt=prompt
        self.current_requested=rows_requested
        return prompt

    def _parse_rows(self,text):
        rows=super()._parse_rows(text)
        self.trace.append({"prompt":self.current_prompt,"requested":self.current_requested,
                           "completion":text,"parsed_rows":len(rows)})
        save_json(self.trace_path,self.trace)
        return rows

    def _load_model(self):
        torch.cuda.synchronize()
        t0=time.perf_counter()
        model=super()._load_model()
        torch.cuda.synchronize()
        self.load_seconds=time.perf_counter()-t0
        quantization=getattr(model.config,"quantization_config",{})
        if hasattr(quantization,"to_dict"): quantization=quantization.to_dict()
        self.model_details={"class":type(model).__name__,"parameters":sum(p.numel() for p in model.parameters()),
                            "device_map":{str(k):str(v) for k,v in getattr(model,"hf_device_map",{}).items()},
                            "quantization":quantization}
        return model


class AuditedICL(AuditMixin,SafetensorsLLMInContextGenerator):
    pass


class AuditedQLoRA(AuditMixin,SafetensorsQLORAGenerator):
    pass


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def save_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def sample_and_save(model,name,out,n,seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.trace=[]
    model.trace_path=out/(name+"-trace.json")
    model.progress_path=str(out/(name+".csv"))
    clean_gpu()
    t0=time.perf_counter()
    values=model.sample(n)
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-t0
    info={"rows":len(values),"seed":seed,"attempts":len(model.trace),
          "parsed_before_final_truncation":sum(x["parsed_rows"] for x in model.trace),
          "sample_seconds_including_load":elapsed,"model_load_seconds":model.load_seconds,
          "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
          "peak_reserved_bytes":torch.cuda.max_memory_reserved(),"model":model.model_details}
    np.savetxt(out/(name+".csv"),values,delimiter=",",header=",".join(COLUMNS),comments="",fmt="%.10g")
    save_json(out/(name+"-trace.json"),model.trace)
    save_json(out/(name+"-run.json"),info)
    print("COMPLETE",name,json.dumps(info),flush=True)
    return info


def main(args):
    from huggingface_hub import snapshot_download
    assert torch.cuda.is_available(),"This execution belongs on CUDA"
    torch.set_num_threads(8)
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    make_data(out)
    train=np.loadtxt(out/"train.csv",delimiter=",",skiprows=1)
    # Pinned public checkpoint, not a placeholder path or user-supplied weights.
    path=snapshot_download(MODEL,revision=REVISION,
                           allow_patterns=["*.json","*.safetensors","merges.txt","vocab.json","LICENSE","README.md"])
    config=dict(model_path=path,examples=8,rows_per_prompt=8,max_new_tokens=384,
                temperature=.7,load_in_4bit=True,max_attempts=max(8,args.rows//2),device="cuda")
    meta={"model":MODEL,"revision":REVISION,"gpu":torch.cuda.get_device_name(),
          "python":platform.python_version(),"dependencies":{key:version(key) for key in
          ["torch","transformers","peft","accelerate","bitsandbytes","numpy","safetensors","datasets"]},
          "train_rows":800,"test_rows":512,"requested_rows":args.rows,"data_seed":20260927,
          "sampling_seed":724,"config":{k:v for k,v in config.items() if k!="model_path"},
          "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [Path(__file__),ROOT/"trex/simdgp.py"]}}
    save_json(out/"metadata.json",meta)
    if not args.skip_frozen:
        frozen=AuditedICL(**config).fit(train,column_names=COLUMNS)
        sample_and_save(frozen,"frozen",out,args.rows,724)
        del frozen
    adapted=AuditedQLoRA(**config).fit(train,column_names=COLUMNS)
    training=dict(num_train_epochs=1,learning_rate=2e-4,per_device_train_batch_size=1,
                  gradient_accumulation_steps=8,rows_per_completion=4,train_samples=args.train_samples,
                  max_length=384,lora_r=16,lora_alpha=32,lora_dropout=.05,seed=123)
    adapter=out/"adapter"
    if not args.skip_training:
        from transformers import AutoTokenizer
        from trex.simdgp import _CompletionDataset
        examples=adapted._adapter_training_examples(rows_per_completion=4,train_samples=args.train_samples,seed=123)
        tokenizer=AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
        dataset=_CompletionDataset(examples,tokenizer,max_length=384)
        target_counts=[int((item["labels"]!=-100).sum()) for item in dataset]
        assert min(target_counts)>0,"A completion was completely truncated"
        training["target_tokens_min"]=min(target_counts)
        training["target_tokens_max"]=max(target_counts)
        clean_gpu()
        t0=time.perf_counter()
        adapted.fit_adapter(output_dir=str(adapter),**{k:v for k,v in training.items() if not k.startswith("target_")})
        torch.cuda.synchronize()
        training.update(elapsed_seconds=time.perf_counter()-t0,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        checkpoints=sorted(adapter.glob("checkpoint-*/trainer_state.json"),key=lambda p:int(p.parent.name.split("-")[-1]))
        if checkpoints:
            shutil.copy2(checkpoints[-1],out/"trainer_state.json")
        from safetensors import safe_open
        with safe_open(adapter/"adapter_model.safetensors",framework="pt",device="cpu") as weights:
            training["adapter_parameters"]=sum(int(np.prod(weights.get_slice(key).get_shape())) for key in weights.keys())
        training["adapter_sha256"]=hashlib.sha256((adapter/"adapter_model.safetensors").read_bytes()).hexdigest()
        save_json(out/"training.json",training)
        print("TRAINING COMPLETE",json.dumps(training),flush=True)
    else:
        adapted.adapter_path=str(adapter)
        adapted.tokenizer_path=str(adapter)
    sample_and_save(adapted,"qlora",out,args.rows,724)
    rng=np.random.default_rng(725)
    bootstrap=train[rng.integers(len(train),size=args.rows)]
    np.savetxt(out/"resample.csv",bootstrap,delimiter=",",header=",".join(COLUMNS),comments="",fmt="%.10g")
    save_json(out/"hashes.json",{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(out.iterdir()) if p.is_file() and p.name!="hashes.json"})
    print("FINISHED",out,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"tmp/llm-guide/executed")
    parser.add_argument("--rows",type=int,default=256)
    parser.add_argument("--train-samples",type=int,default=512)
    parser.add_argument("--skip-training",action="store_true")
    parser.add_argument("--skip-frozen",action="store_true")
    main(parser.parse_args())
