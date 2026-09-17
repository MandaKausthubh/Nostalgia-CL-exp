"""Round-trip test for training.resume bundles.

Highest-value check: fisher/theta_star are keyed by ``id(p)`` at runtime, which
is NOT stable across processes — this test forces a fresh model (different
object ids) and asserts the re-keying by parameter name still matches.

Run: python -m tests.test_resume_bundle
"""

import os
import tempfile
import types

import torch

from baselines.agem import ReplayBuffer
from models_utils.image_model import ImageModelModule
from training.resume import (
    build_bundle,
    experiment_key,
    find_resume,
    restore_bundle,
    resume_dir,
    save_bundle,
    write_progress,
)


TASKS = ["synth_a", "synth_b"]
DATASET_CONFIG = {
    "synth_a": {"batch_size": 4, "max_train_samples": 16},
    "synth_b": {"batch_size": 4, "max_train_samples": 16},
}


def _tasks():
    return [
        {"name": "synth_a", "num_classes": 3},
        {"name": "synth_b", "num_classes": 3},
    ]


def _args(checkpoint_dir):
    return types.SimpleNamespace(
        method="ewc",
        backbone="resnet10",
        model_name="gpt2",
        seed=0,
        tasks=TASKS,
        image_size=32,
        epochs_phase2=2,
        lr=1e-3,
        head_lr=5e-4,
        warmup_steps=10,
        total_steps=50,
        k=4,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        wandb_name="roundtrip_test",
        checkpoint_dir=checkpoint_dir,
        hf_hub_namespace=None,
        resume="auto",
    )


def _build_model():
    torch.manual_seed(0)
    return ImageModelModule(
        lr=1e-3,
        head_lr=5e-4,
        tasks_config={"synth_a": 3, "synth_b": 3},
        method="ewc",
        backbone_name="resnet10",
        image_size=32,
        pretrained=False,
        use_lora=False,
    )


def _make_scheduler_state(model):
    backbone = model.get_backbone_params_dict()
    return types.SimpleNamespace(
        tasks=_tasks(),
        Q_memory=torch.randn(4, 8),
        Lambda_memory=torch.rand(4),
        fisher_memory={id(p): torch.rand_like(p) for p in backbone.values()},
        theta_star_memory={id(p): p.detach().clone() for p in backbone.values()},
        replay_buffer=_filled_buffer(),
        teacher_memory=None,
    )


def _filled_buffer():
    buf = ReplayBuffer(capacity=4)
    for i in range(3):
        buf.add(
            torch.randn(3, 8, 8),
            torch.ones(8, dtype=torch.long),
            torch.tensor(i % 3),
            TASKS[i % 2],
        )
    return buf


class _StubScheduler:
    def __init__(self, state):
        self.__dict__.update(vars(state))

    def __getattr__(self, item):
        raise AttributeError(item)


def _assert_close(a, b, label):
    assert torch.allclose(a, b), f"{label} mismatch"


def main():
    tasks = _tasks()
    with tempfile.TemporaryDirectory() as tmp:
        args = _args(tmp)

        model = _build_model()
        # Give heads/backbone non-default weights so the round-trip is meaningful.
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        model.completed_tasks = {"synth_a"}
        model.global_step_counter = 123
        model.alignment_step_counter = 7

        sched = _StubScheduler(_make_scheduler_state(model))

        # A teacher snapshot (SDFT-style) to exercise the teacher path.
        teacher = _build_model()
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        sched.teacher_memory = teacher

        key_hash, key = experiment_key(args, tasks, DATASET_CONFIG)
        bundle = build_bundle(
            model=model,
            scheduler_cb=sched,
            args=args,
            key_hash=key_hash,
            key=key,
            tasks=tasks,
            wandb_run_id="abc123",
            val_metrics={"synth_a": 0.91},
        )

        directory = resume_dir(args)
        latest = os.path.join(directory, "latest.pt")
        save_bundle(latest, bundle)
        write_progress(directory, bundle)

        assert os.path.exists(latest), "bundle not written"

        # --- discovery finds it by matching config hash ---
        found = find_resume(args, tasks, DATASET_CONFIG)
        assert found is not None, "find_resume returned None"
        assert found["resume_after_idx"] == 1
        assert found["wandb_run_id"] == "abc123"

        # --- restore onto a FRESH model (different id(p) values) ---
        model2 = _build_model()
        sched2 = _StubScheduler(
            types.SimpleNamespace(
                tasks=_tasks(),
                Q_memory=None,
                Lambda_memory=None,
                fisher_memory=None,
                theta_star_memory=None,
                replay_buffer=None,
                teacher_memory=None,
            )
        )
        args2 = _args(tmp)
        run_id = restore_bundle(found, model2, sched2, args2)

        assert run_id == "abc123"
        assert args2.resume_after_idx == 1
        assert args2.epochs_phase1 == 0
        assert model2.completed_tasks == {"synth_a"}
        assert model2.global_step_counter == 123
        assert model2.alignment_step_counter == 7

        # model weights
        for name, tensor in model.state_dict().items():
            if name.startswith("teacher_model."):
                continue
            _assert_close(tensor, model2.state_dict()[name], f"state_dict[{name}]")

        # eigenspace
        _assert_close(sched.Q_memory, sched2.Q_memory, "Q_memory")
        _assert_close(sched.Lambda_memory, sched2.Lambda_memory, "Lambda_memory")

        # fisher / theta_star re-keyed by NAME (id(p) differs across models)
        backbone1 = model.get_backbone_params_dict()
        backbone2 = model2.get_backbone_params_dict()
        assert set(backbone1) == set(backbone2), "backbone param names differ"
        for name in backbone1:
            pid1, pid2 = id(backbone1[name]), id(backbone2[name])
            assert pid1 != pid2, "expected distinct param ids across models"
            _assert_close(sched.fisher_memory[pid1], sched2.fisher_memory[pid2], f"fisher[{name}]")
            _assert_close(sched.theta_star_memory[pid1], sched2.theta_star_memory[pid2], f"theta_star[{name}]")
        # exactly the same number of entries survived the re-key
        assert len(sched2.fisher_memory) == len(backbone2)

        # replay buffer
        assert sched2.replay_buffer.capacity == sched.replay_buffer.capacity
        assert sched2.replay_buffer._pos == sched.replay_buffer._pos
        assert len(sched2.replay_buffer) == len(sched.replay_buffer)
        for (ii1, am1, lb1, tk1), (ii2, am2, lb2, tk2) in zip(
            sched.replay_buffer._store, sched2.replay_buffer._store
        ):
            _assert_close(ii1, ii2, "replay input_ids")
            _assert_close(am1, am2, "replay attention_mask")
            _assert_close(lb1, lb2, "replay label")
            assert tk1 == tk2

        # teacher
        for name, tensor in teacher.state_dict().items():
            _assert_close(tensor, sched2.teacher_memory.state_dict()[name], f"teacher[{name}]")

        # --- key mismatch must NOT resume ---
        other = dict(DATASET_CONFIG)
        other["synth_a"] = {"batch_size": 999, "max_train_samples": 16}
        args_mismatch = _args(tmp)
        args_mismatch.epochs_phase2 = 99
        assert find_resume(args_mismatch, tasks, DATASET_CONFIG) is None, \
            "config change must invalidate the bundle"

        # --- resume=never must NOT resume ---
        args_never = _args(tmp)
        args_never.resume = "never"
        assert find_resume(args_never, tasks, DATASET_CONFIG) is None

    print("OK: resume bundle round-trip passed")


if __name__ == "__main__":
    main()
