# Opt-in DeepSpeed 0.20 compatibility preview

The historical Docker workflow and the default CLI still use the legacy engine.
To try this preview, use a separate environment with an official
[PyTorch 2.13.0 build](https://pytorch.org/get-started/locally/) for the
intended accelerator and the unmodified release of DeepSpeed:

```sh
DS_BUILD_OPS=0 python -m pip install -r requirements.txt
python finetune_cpm2.py --help
```

Use the existing launch options together with `--deepspeed-modern` and
`--deepspeed`. Do **not** copy this repository's legacy `engine.py` or
`stage1.py` into the modern DeepSpeed installation. This path uses the
public `deepspeed.initialize(config=...)` API, ZeRO stage 1, and PyTorch
AdamW, without an Apex dependency. Existing FP16 config and `--fp16`
must agree. Stage 2/3, optimizer offload, BF16, and autocast are out
of scope for this preview.

## Checkpoints

The modern path saves model weights only, one tensor shard per model-parallel
rank, and a `latest_checkpointed_iteration.txt` tracker. Loading requires
`--finetune` to make the new-run semantics explicit. It restores supported
`mp_rank_XX_model_states.pt` model shards before DeepSpeed creates FP32 master
parameters. It does not restore optimizer, scheduler, RNG, or dataloader
state. Passing optimizer, scheduler, or dataloader resume flags fails with
a clear error. Keep model-parallel size the same on load.

Loads use explicit `weights_only=True` with no fallback. Some historical
checkpoints contain Python objects and will fail to load. No historical
weights have been verified here. Use only checkpoints whose provenance
you trust; do not bypass restricted deserialization to make an old
checkpoint load. This preview is not equivalent to a full legacy
optimizer resume and does not change legacy checkpoint handling.

## Verification

```sh
python -m unittest discover -s tests -v
```

CPU verification so far: Python 3.11, ARM64, torch 2.13.0+cpu and DeepSpeed
0.20.0. The fine-tuning CLI imports and prints help. Unit tests cover
restricted weights-only loading, invalid tags, mismatched keys, rejecting
full resume, and a fresh process restoring a synthetic shard and taking
a PyTorch training step.

On the Oracle ARM64 host, the same PyTorch 2.13.0 / DeepSpeed 0.20.0
builds also passed a real two-process Gloo ZeRO-1 smoke in both model-parallel
size 1 (two data ranks) and size 2 (two model ranks). Both runs performed
forward/backward and an optimizer step, wrote model weights, then launched
fresh processes to warm-start and take another step. This used only
synthetic input, FP32, and the stock Gloo fallback for the unavailable
optional shared-memory extension. It does not establish CUDA/FP16 support,
full CLI training on CPU, numerical parity, or old-checkpoint compatibility.

Run this real DeepSpeed train/save/warm-start smoke before relying on the
preview for training. In an appropriate test environment, repeat with
two CUDA devices, once as `--model-parallel-size=2` and once as
`--model-parallel-size=1`, to cover both model and data parallel groups:

```sh
export PYTHONPATH=.
CHECKPOINT_DIR="$(mktemp -d)"
torchrun --standalone --nproc-per-node=2 tests/modern_runtime_smoke.py \
  --fp16 --model-parallel-size=2 --checkpoint-dir "$CHECKPOINT_DIR"
torchrun --standalone --nproc-per-node=2 tests/modern_runtime_smoke.py \
  --fp16 --model-parallel-size=2 --checkpoint-dir "$CHECKPOINT_DIR" --reload
```

The smoke uses generated inputs and does not require pretrained weights or
datasets. For a CPU smoke, use the CPU PyTorch build, omit `--fp16`, and
set `DS_ACCELERATOR=cpu`. The optional harness-only `--cpu-gloo-only` flag
skips the DeepSpeed CPU shared-memory extension on hosts without a compiler.
