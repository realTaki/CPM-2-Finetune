import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

import modern_runtime


class ModernRuntimeTest(unittest.TestCase):
    def test_adapter_uses_one_unmodified_config(self):
        ds = Mock()
        config = {"zero_optimization": {"stage": 1}}
        model, opt, scheduler, mpu = (object() for _ in range(4))
        modern_runtime.initialize_engine(ds, model, opt, scheduler, mpu, config)
        ds.initialize.assert_called_once_with(
            model=model, optimizer=opt, lr_scheduler=scheduler, mpu=mpu,
            dist_init_required=False, config=config,
        )
        ds.initialize.call_args.kwargs["config"]["zero_optimization"]["stage"] = 2
        self.assertEqual(config["zero_optimization"]["stage"], 1)

    def test_fails_closed_for_resume(self):
        args = SimpleNamespace(deepspeed=True, fp16=True, cpu_optimizer=False,
                               load_oprimizer_states=False, load_lr_scheduler_states=False,
                               resume_dataloader=False, load="/unused", finetune=False,
                               prompt_tune=False)
        ds = SimpleNamespace(__version__="0.20.0")
        config = {"zero_optimization": {"stage": 1}, "fp16": {"enabled": True}}
        with self.assertRaisesRegex(ValueError, "weights-only"):
            modern_runtime.validate_runtime(ds, args, config)
        args.finetune = True
        modern_runtime.validate_runtime(ds, args, config)
        args.load_oprimizer_states = True
        with self.assertRaisesRegex(ValueError, "cannot resume"):
            modern_runtime.validate_runtime(ds, args, config)

    def test_warm_start_and_prompt_precede_optimizer_construction(self):
        # No GPU required to validate the actual orchestration in utils.
        import utils
        args = SimpleNamespace(deepspeed=True, deepspeed_modern=True, fp16=True,
                               cpu_optimizer=False, load_oprimizer_states=False,
                               load_lr_scheduler_states=False, resume_dataloader=False,
                               load="weights", finetune=True, prompt_tune=True,
                               prompt_config="prompt.json")
        config = {"zero_optimization": {"stage": 1}, "fp16": {"enabled": True}}
        with (
            patch.object(utils, "deepspeed", SimpleNamespace(__version__="0.20.0")),
            patch.object(utils, "get_model") as get_model,
            patch.object(utils, "get_optimizer") as get_optimizer,
            patch.object(utils, "get_learning_rate_scheduler") as get_scheduler,
            patch.object(utils.mpu, "get_model_parallel_rank", return_value=3),
            patch.object(utils.mpu, "get_data_parallel_rank", return_value=1),
            patch.object(modern_runtime, "load_model_weights") as load,
            patch.object(modern_runtime, "initialize_engine",
                         return_value=("engine", "optimizer", None, "scheduler")) as initialize,
        ):
            events = Mock()
            for label, mock in (
                ("load", load),
                ("prompt", get_model.return_value.init_prompt_embeds),
                ("get_optimizer", get_optimizer),
                ("get_scheduler", get_scheduler),
                ("initialize", initialize),
            ):
                events.attach_mock(mock, label)
            output = utils.setup_model_and_optimizer(args, 16, config, {"init_from_vocab": True})
            self.assertEqual([call[0] for call in events.mock_calls],
                             ["load", "prompt", "get_optimizer", "get_scheduler", "initialize"])
            load.assert_called_once_with(get_model.return_value, "weights", 3, True)
            self.assertEqual(output, ("engine", "optimizer", "scheduler"))
            self.assertEqual(args.iteration, 0)

    def test_synthetic_rank_shard_round_trip(self):
        model = torch.nn.Linear(2, 1)
        loaded = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = modern_runtime.save_model_weights(model, directory, 17, 3)
            tracker = Path(directory) / "latest_checkpointed_iteration.txt"
            tracker.write_text("17")
            self.assertEqual(modern_runtime.load_model_weights(loaded, directory, 3), path)
            for key, param in model.state_dict().items():
                torch.testing.assert_close(param, loaded.state_dict()[key])
            torch.save({"module": {"bias": torch.zeros(1)}}, path)
            with self.assertRaisesRegex(ValueError, "keys differ"):
                modern_runtime.load_model_weights(loaded, directory, 3)
            tracker.write_text("../escape")
            with self.assertRaisesRegex(ValueError, "Invalid CPM checkpoint tag"):
                modern_runtime.load_model_weights(loaded, directory, 3)

    def test_fresh_process_loads_shard_and_takes_another_step(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = torch.nn.Linear(2, 1)
            with torch.no_grad():
                model.weight.fill_(2)
                model.bias.fill_(3)
            optim = torch.optim.SGD(model.parameters(), lr=0.25)
            model(torch.ones(1, 2)).sum().backward()
            optim.step()
            modern_runtime.save_model_weights(model, root, 1, 0)
            (root / "latest_checkpointed_iteration.txt").write_text("1")
            source = """
import sys
import torch
import modern_runtime
model = torch.nn.Linear(2, 1)
modern_runtime.load_model_weights(model, sys.argv[1], 0)
torch.testing.assert_close(model.weight, torch.full((1, 2), 1.75))
torch.testing.assert_close(model.bias, torch.full((1,), 2.75))
optim = torch.optim.SGD(model.parameters(), lr=0.25)
model(torch.ones(1, 2)).sum().backward()
optim.step()
torch.testing.assert_close(model.weight, torch.full((1, 2), 1.5))
torch.testing.assert_close(model.bias, torch.full((1,), 2.5))
"""
            subprocess.run([sys.executable, "-c", source, directory], check=True,
                           cwd=Path(__file__).resolve().parents[1], timeout=30)

    def test_load_uses_restricted_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "latest_checkpointed_iteration.txt").write_text("release")
            model = torch.nn.Linear(2, 1)
            captured_load = Mock(return_value={"module": model.state_dict()})
            with patch.object(modern_runtime.torch, "load", captured_load):
                modern_runtime.load_model_weights(model, root, 0)
            captured_load.assert_called_once_with(
                root / "release/mp_rank_00_model_states.pt",
                map_location="cpu", weights_only=True
            )


if __name__ == "__main__":
    unittest.main()
