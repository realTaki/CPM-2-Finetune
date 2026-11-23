"""Synthetic forward/backward and weights-only restart with unmodified DeepSpeed."""
import argparse
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist

import modern_runtime
import mpu
from model import EncDecConfig, EncDecModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--model-parallel-size", default=1, type=int)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--cpu-gloo-only", action="store_true",
                        help="Skip optional DeepSpeed CPU shared-memory extension when no compiler is present.")
    args = parser.parse_args()
    accelerator = deepspeed.accelerator.get_accelerator()
    device = accelerator.device_name()
    if device == "cuda":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if args.cpu_gloo_only:
        if device != "cpu":
            parser.error("--cpu-gloo-only requires the CPU accelerator")
        # Test the stock fallback collectives without compiling an optional
        # Linux extension. This test-only override never applies to training.
        deepspeed.ops.__compatible_ops__["deepspeed_shm_comm"] = False
    deepspeed.init_distributed()
    mpu.initialize_model_parallel(args.model_parallel_size)
    try:
        deepspeed.checkpointing.configure(mpu, num_checkpoints=2)
        torch.manual_seed(11 if args.reload else 73)
        deepspeed.checkpointing.model_parallel_cuda_manual_seed(11 if args.reload else 73)
        config = EncDecConfig(d_model=8, d_kv=4, d_ff=16, num_layers=1,
                              num_decoder_layers=1, num_heads=2, dropout_rate=0.0)
        config.vocab_size = 16
        model = EncDecModel(config, parallel_output=True).to(device)
        if args.fp16:
            model.half()
        if args.reload:
            modern_runtime.load_model_weights(
                model, args.checkpoint_dir, mpu.get_model_parallel_rank()
            )
        seed_weights = model.lm_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
        engine, _, _, _ = modern_runtime.initialize_engine(
            deepspeed, model, optimizer, None, mpu,
            {"train_micro_batch_size_per_gpu": 1,
             "zero_optimization": {"stage": 1}, "zero_allow_untested_optimizer": True,
             "fp16": {"enabled": args.fp16, "loss_scale": 1}},
        )
        ids = torch.tensor([[1, 2]], dtype=torch.long, device=device)
        mask = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=device)
        pred = engine(enc_input_ids=ids, enc_attention_mask=mask,
                      dec_input_ids=ids, dec_attention_mask=mask,
                      cross_attention_mask=mask)["lm_logits"]
        loss = pred.float().square().mean()
        engine.backward(loss)
        engine.step()
        assert torch.isfinite(loss)
        assert not torch.equal(seed_weights, model.lm_head.weight)
        print(f"rank {dist.get_rank()} {'restart' if args.reload else 'initial'} step passed: {loss.item()}")
        if not args.reload:
            if mpu.get_data_parallel_rank() == 0:
                modern_runtime.save_model_weights(
                    model, args.checkpoint_dir, 1, mpu.get_model_parallel_rank()
                )
            dist.barrier()
            if dist.get_rank() == 0:
                (Path(args.checkpoint_dir) / "latest_checkpointed_iteration.txt").write_text("1")
            dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
