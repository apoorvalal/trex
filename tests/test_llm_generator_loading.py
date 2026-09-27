"""The fitted NF4 adapter must be reloaded on the same quantized base law."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from trex import SafetensorsLLMInContextGenerator, SafetensorsQLORAGenerator


@pytest.mark.parametrize("adapted", [False, True])
def test_generation_reloads_nf4_double_quantized_base(monkeypatch, adapted):
    calls=[]
    def load(path,**kwargs):
        calls.append((path,kwargs))
        return SimpleNamespace()
    def config(**kwargs):
        return kwargs
    auto=SimpleNamespace(from_pretrained=load)
    monkeypatch.setitem(sys.modules,"transformers",SimpleNamespace(
        AutoModelForCausalLM=auto,AutoModelForImageTextToText=auto,BitsAndBytesConfig=config))
    attachments=[]
    def attach(base,path):
        attachments.append(path)
        return base
    monkeypatch.setitem(sys.modules,"peft",SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=attach)))
    cls=SafetensorsQLORAGenerator if adapted else SafetensorsLLMInContextGenerator
    model=cls("fixed-base",load_in_4bit=True,device="cuda").fit(np.ones((4,2)))
    if adapted: model.adapter_path="fitted-adapter"
    model._load_model()
    assert calls[0][0]=="fixed-base"
    quant=calls[0][1]["quantization_config"]
    assert quant["load_in_4bit"]
    assert quant["bnb_4bit_quant_type"]=="nf4"
    assert quant["bnb_4bit_use_double_quant"]
    assert quant["bnb_4bit_compute_dtype"]==torch.bfloat16
    assert attachments==(["fitted-adapter"] if adapted else [])
