"""Opt-in DeepSpeed 0.20 runtime for CPM-2, using weights-only checkpoints."""
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import torch


DEEPSPEED_VERSION = "0.20.0"
TORCH_VERSION = "2.13.0"


def validate_runtime(deepspeed, args, config):
    if deepspeed.__version__ != DEEPSPEED_VERSION:
        raise ValueError(
            f"--deepspeed-modern needs unmodified DeepSpeed {DEEPSPEED_VERSION}; "
            f"found {deepspeed.__version__}"
        )
    if torch.__version__.partition("+")[0] != TORCH_VERSION:
        raise ValueError(
            f"--deepspeed-modern needs PyTorch {TORCH_VERSION}; found {torch.__version__}"
        )
    if not args.deepspeed:
        raise ValueError("--deepspeed-modern also requires --deepspeed")
    zero = config.get("zero_optimization", {})
    if not isinstance(zero, Mapping) or zero.get("stage") != 1:
        raise ValueError("--deepspeed-modern currently supports only ZeRO stage 1")
    if bool(config.get("fp16", {}).get("enabled", False)) != args.fp16:
        raise ValueError("--fp16 must agree with the DeepSpeed config")
    if config.get("bf16", {}).get("enabled") or config.get("torch_autocast", {}).get("enabled"):
        raise ValueError("--deepspeed-modern does not support BF16 or DeepSpeed autocast")
    if zero.get("offload_optimizer") or args.cpu_optimizer:
        raise ValueError("--deepspeed-modern does not support optimizer offload")
    if args.load_oprimizer_states or args.load_lr_scheduler_states or args.resume_dataloader:
        raise ValueError("--deepspeed-modern cannot resume optimizer, scheduler, or dataloader state")
    if args.load is not None and not args.finetune:
        raise ValueError("--deepspeed-modern --load is weights-only; add --finetune")
    if args.prompt_tune and args.prompt_config is None:
        raise ValueError("--prompt-tune needs --prompt-config")


def initialize_engine(deepspeed, model, optimizer, lr_scheduler, mpu, config):
    # CPM overrides batch size and gradient accumulation in the in-memory config.
    # Passing args as well would make DeepSpeed reject two competing configs.
    return deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        mpu=mpu,
        dist_init_required=False,
        config=deepcopy(config),
    )


def _shard_path(root, tag, model_parallel_rank):
    return Path(root) / tag / f"mp_rank_{model_parallel_rank:02d}_model_states.pt"


def load_model_weights(model, root, model_parallel_rank, prompt_tune=False):
    tracker = Path(root) / "latest_checkpointed_iteration.txt"
    tag = tracker.read_text().strip()
    if tag != "release" and (not tag.isascii() or not tag.isdecimal() or int(tag) <= 0):
        raise ValueError(f"Invalid CPM checkpoint tag: {tag!r}")
    if tag != "release":
        tag = str(int(tag))
    path = _shard_path(root, tag, model_parallel_rank)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping) or not isinstance(state.get("module"), Mapping):
        raise ValueError(f"{path} is not a CPM model checkpoint")
    weights = state["module"]
    expected = set(model.state_dict())
    if any(not isinstance(value, torch.Tensor) for value in weights.values()):
        raise ValueError(f"{path} contains non-tensor model state")
    if any(not isinstance(key, str) for key in weights):
        raise ValueError(f"{path} contains non-string model keys")
    missing = expected - weights.keys()
    allowed_missing = {"encoder.prompt_embeds.weight"} if prompt_tune else set()
    if missing - allowed_missing or weights.keys() - expected:
        raise ValueError(f"CPM model keys differ: missing={sorted(missing)}, "
                         f"unexpected={sorted(weights.keys() - expected)}")
    model.load_state_dict(weights, strict=not bool(missing))
    return path


def save_model_weights(model, root, iteration, model_parallel_rank):
    path = _shard_path(root, str(iteration), model_parallel_rank)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"module": model.state_dict()}, tmp)
    tmp.replace(path)
    return path
