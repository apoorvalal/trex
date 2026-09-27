# Executed LLM-generator vignette

This is the full GPU execution behind `docs/guides/llm-generators.qmd`,
not a simulated set of model responses. September 27, 2026: RTX 5070,
Qwen2.5-0.5B-Instruct at the revision pinned in `run.py`.

## Run training and generation

Use the shared CUDA-enabled Torch environment with Transformers, PEFT,
bitsandbytes, Accelerate and safetensors. Exact versions are in the saved
`metadata.json` (also in the guide). From the repository root:

~~~bash
python benchmarks/llm_generators/run.py --out tmp/llm-guide/executed
~~~

The runner generates three independent samples from a specified DGP
(800 training, 512 test, 256 reference), runs frozen ICL, constructs 512
training-only prompt/completion examples, trains a rank-16 adapter for
64 optimizer steps, reloads it and samples 256 rows. Both sample calls reset
NumPy/Torch seeds to 724. The adapter-example seed is 123; Hugging Face
Trainer uses its default training seed 42. The complete sequence matters
for initialization RNG state. No held-out row selects prompts or training.

The audit subclass only observes prompts, completions, loading and parsing;
it uses the production wrappers' implementations. It keeps every failed
attempt, does not enforce schema or deduplicate, and retains the first
256 parsed rows. A model's extra parsed rows are recorded before truncation.
NF4/double-quantization/BF16 loading matches adapter training; the guide
run includes the correction to the former FP4 generation defaults.

The output adapter can be reused by constructing and fitting a
`SafetensorsQLORAGenerator` on the training rows, then setting its
`adapter_path` and `tokenizer_path` attributes to the saved adapter directory.
Model and adapter weights are not in the public
replication ZIP; its training record includes the adapter SHA-256.

## Recompute the published evaluation without a GPU

~~~bash
python benchmarks/llm_generators/analyze.py
~~~

This reads the hash-checked saved execution capsule in
`docs/guides/data/llm-generators/2026-09-27/executed.json.gz`. It needs only
the ordinary Trex dependencies plus matplotlib for figures, not optional
model-loading packages. Every raw completion is reparsed and matched to the
published rows; each prompt's examples must be training rows. Metrics retain
the seven finite but support-invalid frozen samples.

Main results are descriptive for one fixed-budget run:

| Method | Standardized sliced W1 | Empty attempts | Exact employed-row copies |
|---|---:|---:|---:|
| Empirical resampling | 0.158 | n/a | 100% |
| Frozen Qwen | 0.198 | 44 / 62 | 74.6% |
| Qwen + QLoRA | 0.270 | 0 / 38 | 0.63% |

Training: 91.67 s. Sampling including load/all attempts: frozen 374.11 s,
adapted 78.72 s. Adapter training improves output formatting and correlation
structure but worsens overall/marginal fit in this run. Resampling is a
stronger joint-distribution baseline. No privacy claim follows from reduced
exact copying, especially with naturally repeated zero-income rows.
