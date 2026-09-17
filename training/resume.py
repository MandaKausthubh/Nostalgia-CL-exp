"""Crash-resume bundles for the sequential CL pipeline.

The pipeline persists model weights (``lightning_state_dict.pt``) after every
task, but ALL cross-task continual-learning state lives in RAM only — the
Nostalgia/GPM eigenspace (``Q``/``Lambda``), EWC's Fisher + ``theta_star``, the
A-GEM replay buffer, the SDFT teacher, the completed-task set and the global step
counters. A crash therefore loses the whole run.

This module bundles that state at each **task boundary** (a task = one domain's
Phase-2 block, written after its Phase-3 estimation) into one resumable blob,
stored locally and optionally mirrored to the HuggingFace Hub. On startup the
pipeline asks a simple yes/no (or auto-resumes when non-interactive).

Granularity rationale: the optimizer is torn down and rebuilt at every
task/phase transition (``phase_scheduler.py``), so Adam moments reset at task
boundaries anyway — there is no optimizer continuity to preserve. Worst case a
crash redoes one task's Phase-2 (≤ ``epochs_phase2`` epochs).

Size caveat: the A-GEM replay buffer stores raw ``224x224x3`` float32 tensors
(~1.2 GB at ``--agem_mem_size 2000``), so bundles can be large. We upload only
the *latest* bundle as ``latest.pt`` (overwritten each task) plus a small
``progress.json``; at most two per-task copies are kept locally. HF keeps commit
history, so the repo still grows — an optional follow-up is to store buffer
tensors as uint8.
"""

import hashlib
import json
import os
import random
import shutil
import sys
import time

import numpy as np
import torch

BUNDLE_VERSION = 1

PROGRESS_FILE = "progress.json"
LATEST_FILE = "latest.pt"
DECISION_FILE = "decision.json"


# ---------------------------------------------------------------------------
# Identity / paths
# ---------------------------------------------------------------------------

def exp_name(args) -> str:
    """Stable experiment name — the wandb run name (repo + resume dir key)."""
    name = getattr(args, "wandb_name", None)
    if name:
        return name
    tasks = "-".join(getattr(args, "tasks", []) or [])
    return f"{tasks or 'run'}_{getattr(args, 'backbone', 'model')}_{getattr(args, 'method', 'method')}"


def experiment_key(args, tasks, dataset_config) -> tuple:
    """sha1 over the config that makes two runs resumable-compatible.

    Mirrors the Phase-1 cache-key pattern. Deliberately EXCLUDES paths, dirs,
    checkpoint_dir and num_workers — those must not invalidate a bundle.
    """
    first_cfg = dataset_config[tasks[0]["name"]]
    key = {
        "method": getattr(args, "method", None),
        "backbone": getattr(args, "backbone", None),
        "model_name": getattr(args, "model_name", None),
        "seed": getattr(args, "seed", None),
        "tasks": sorted(t["name"] for t in tasks),
        "image_size": getattr(args, "image_size", None),
        "epochs_phase2": getattr(args, "epochs_phase2", None),
        "batch_size": first_cfg["batch_size"],
        "lr": getattr(args, "lr", None),
        "head_lr": getattr(args, "head_lr", None),
        "warmup_steps": getattr(args, "warmup_steps", None),
        "total_steps": getattr(args, "total_steps", None),
        "k": getattr(args, "k", None),
        "use_lora": getattr(args, "use_lora", False),
        "lora_r": getattr(args, "lora_r", None),
        "lora_alpha": getattr(args, "lora_alpha", None),
        "lora_dropout": getattr(args, "lora_dropout", None),
        "max_train_samples": first_cfg["max_train_samples"],
    }
    key_str = json.dumps(key, sort_keys=True)
    key_hash = hashlib.sha1(key_str.encode("utf-8")).hexdigest()
    return key_hash, key


def resume_dir(args) -> str:
    return os.path.join(
        os.path.abspath(getattr(args, "checkpoint_dir", "./checkpoints")),
        "resume",
        exp_name(args),
    )


def hf_repo_id(args) -> str:
    namespace = getattr(args, "hf_hub_namespace", None)
    if not namespace:
        raise ValueError("--push_to_hub requires --hf_hub_namespace or HF_USERNAME/HF_ORG")
    return f"{namespace}/{exp_name(args)}"


# ---------------------------------------------------------------------------
# Atomic local I/O
# ---------------------------------------------------------------------------

def save_bundle(path, bundle):
    """Atomic write: tmp file + os.replace (mirrors the Phase-1 cache writer)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(bundle, tmp_path)
    os.replace(tmp_path, path)


def _write_json_atomic(path, payload):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp_path, path)


def _read_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _load_bundle(path):
    """Load a bundle (numpy/python RNG states are not in the weights_only allowlist)."""
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# HuggingFace Hub mirror
# ---------------------------------------------------------------------------

def upload_bundle(args, paths):
    """create_repo(exist_ok=True) + upload_file for each local path."""
    from huggingface_hub import HfApi

    repo_id = hf_repo_id(args)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=getattr(args, "hf_hub_private", False),
        exist_ok=True,
    )
    for path in paths:
        if not os.path.exists(path):
            continue
        api.upload_file(
            repo_id=repo_id,
            repo_type="model",
            path_or_fileobj=path,
            path_in_repo=os.path.basename(path),
            commit_message=f"Resume bundle: {os.path.basename(path)}",
        )
    return repo_id


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_resume(args, tasks, dataset_config):
    """Locate a bundle matching this exact experiment config, or None.

    Order: local resume_dir -> HF Hub (if a namespace is configured).
    Returns None on no match, key mismatch, or ``--resume never``.
    """
    if getattr(args, "resume", "prompt") == "never":
        return None

    key_hash, _ = experiment_key(args, tasks, dataset_config)
    directory = resume_dir(args)
    progress_path = os.path.join(directory, PROGRESS_FILE)
    latest_path = os.path.join(directory, LATEST_FILE)

    # 1. Local
    if os.path.exists(progress_path) and os.path.exists(latest_path):
        try:
            progress = _read_json(progress_path)
            if progress.get("key_hash") == key_hash:
                return _load_bundle(latest_path)
            print(f"[Resume] Local bundle key mismatch ({directory}); ignoring.", flush=True)
        except Exception as exc:  # corrupt/partial file — fall through to Hub
            print(f"[Resume] Failed to read local bundle ({exc}); ignoring.", flush=True)

    # 2. HF Hub
    namespace = getattr(args, "hf_hub_namespace", None)
    if namespace:
        try:
            from huggingface_hub import hf_hub_download

            repo_id = hf_repo_id(args)
            os.makedirs(directory, exist_ok=True)
            progress_path = hf_hub_download(
                repo_id=repo_id, filename=PROGRESS_FILE,
                repo_type="model", local_dir=directory,
            )
            progress = _read_json(progress_path)
            if progress.get("key_hash") != key_hash:
                print(f"[Resume] Hub bundle key mismatch ({repo_id}); ignoring.", flush=True)
                return None
            latest_path = hf_hub_download(
                repo_id=repo_id, filename=LATEST_FILE,
                repo_type="model", local_dir=directory,
            )
            return _load_bundle(latest_path)
        except Exception as exc:
            print(f"[Resume] No usable Hub bundle ({exc}).", flush=True)

    return None


def summarize(bundle, args) -> str:
    completed = bundle.get("completed_tasks", [])
    total = len(bundle.get("tasks", []))
    metrics = bundle.get("val_metrics") or {}
    last_acc = metrics.get(completed[-1]) if completed else None
    acc_str = f"{last_acc:.4f}" if isinstance(last_acc, (int, float)) else "n/a"
    return (
        "\n" + "=" * 60 + "\n"
        f"[Resume] Found checkpoint for '{exp_name(args)}'\n"
        f"  completed {len(completed)}/{total} tasks: {completed}\n"
        f"  last val acc ({completed[-1] if completed else '-'}): {acc_str}\n"
        f"  saved at: {bundle.get('timestamp', 'unknown')}\n"
        f"  repo:     {hf_repo_id(args) if getattr(args, 'hf_hub_namespace', None) else 'local only'}\n"
        + "=" * 60
    )


def prompt_resume(summary, mode) -> bool:
    """mode = args.resume ('prompt'|'auto'|'never'). Non-TTY -> auto-resume."""
    if mode == "never":
        return False
    if mode == "auto" or not sys.stdin.isatty():
        if mode != "auto":
            print("[Resume] Non-interactive session -> auto-resuming.", flush=True)
        return True
    print(summary, flush=True)
    answer = input("Resume this run? [Y/n]: ").strip().lower()
    return answer in ("", "y", "yes")


# ---------------------------------------------------------------------------
# DDP-safe prompt sync (no process group exists this early)
# ---------------------------------------------------------------------------

def write_decision(directory, resume: bool):
    _write_json_atomic(os.path.join(directory, DECISION_FILE), {"resume": bool(resume)})


def read_decision(directory) -> bool:
    path = os.path.join(directory, DECISION_FILE)
    if not os.path.exists(path):
        return False
    try:
        return bool(_read_json(path).get("resume", False))
    except Exception:
        return False


def wait_for_decision(directory, timeout=600.0) -> bool:
    """Non-rank-0 ranks block until rank 0 publishes the resume decision."""
    path = os.path.join(directory, DECISION_FILE)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return read_decision(directory)
        time.sleep(1.0)
    raise RuntimeError(
        f"[Resume] Timed out waiting for resume decision at {path}. "
        f"Rank 0 may have failed during the prompt."
    )


# ---------------------------------------------------------------------------
# Bundle build / restore
# ---------------------------------------------------------------------------

def _serialize_replay_buffer(buffer):
    if buffer is None:
        return None
    return {
        "capacity": int(buffer.capacity),
        "pos": int(buffer._pos),
        "store": list(buffer._store),
    }


def _restore_replay_buffer(blob):
    if blob is None:
        return None
    from baselines.agem import ReplayBuffer

    buffer = ReplayBuffer(blob["capacity"])
    buffer._store = list(blob["store"])
    buffer._pos = blob["pos"]
    return buffer


def _serialize_teacher(teacher):
    if teacher is None:
        return None
    return {
        "state_dict": {k: v.detach().cpu() for k, v in teacher.state_dict().items()},
    }


def _restore_teacher(model, blob, device):
    if blob is None:
        return None
    # Rebuild from the *current* model's hparams (identical config) rather than
    # pickling hparams, which may carry a non-serializable writer reference.
    hparams = dict(model.hparams)
    hparams.pop("writer", None)
    teacher = type(model)(**hparams)
    teacher.load_state_dict(blob["state_dict"])
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _rekey_by_name(memory, backbone_params):
    """{id(p): tensor} -> {name: tensor}. ``id(p)`` is NOT stable across processes."""
    if memory is None:
        return None
    name_by_id = {id(p): name for name, p in backbone_params.items()}
    out = {}
    for pid, tensor in memory.items():
        name = name_by_id.get(pid)
        if name is not None:
            out[name] = tensor.detach().to("cpu")
    return out


def _unkey_by_name(memory, backbone_params, device):
    """{name: tensor} -> {id(p): tensor} against this process's parameters."""
    if memory is None:
        return None
    out = {}
    for name, tensor in memory.items():
        param = backbone_params.get(name)
        if param is None:
            continue
        out[id(param)] = tensor.to(device=device, dtype=torch.float32)
    return out


def build_bundle(model, scheduler_cb, args, key_hash, key, tasks, wandb_run_id, val_metrics):
    """Snapshot everything needed to resume this run after the current task."""
    backbone_params = model.get_backbone_params_dict()

    # Exclude the SDFT teacher submodule from the student state_dict: the fresh
    # model has teacher_model=None, so strict load would reject those keys. The
    # teacher travels in cl_state["teacher"] instead.
    model_state_dict = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("teacher_model.")
    }

    completed_ordered = [
        task["name"] for task in tasks if task["name"] in model.completed_tasks
    ]

    cl_state = {
        "Q": None if scheduler_cb.Q_memory is None else scheduler_cb.Q_memory.detach().cpu(),
        "Lambda": None if scheduler_cb.Lambda_memory is None else scheduler_cb.Lambda_memory.detach().cpu(),
        "fisher": _rekey_by_name(scheduler_cb.fisher_memory, backbone_params),
        "theta_star": _rekey_by_name(scheduler_cb.theta_star_memory, backbone_params),
        "replay_buffer": _serialize_replay_buffer(scheduler_cb.replay_buffer),
        "teacher": _serialize_teacher(scheduler_cb.teacher_memory),
    }

    rng = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }

    return {
        "format_version": BUNDLE_VERSION,
        "key_hash": key_hash,
        "key": key,
        "exp_name": exp_name(args),
        "method": getattr(args, "method", None),
        "backbone": getattr(args, "backbone", None),
        "seed": getattr(args, "seed", None),
        "tasks": [task["name"] for task in tasks],
        "completed_tasks": completed_ordered,
        "resume_after_idx": len(completed_ordered),
        "global_step_counter": int(getattr(model, "global_step_counter", 0)),
        "alignment_step_counter": int(getattr(model, "alignment_step_counter", 0)),
        "model_state_dict": model_state_dict,
        "cl_state": cl_state,
        "rng": rng,
        "wandb_run_id": wandb_run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "val_metrics": {k: float(v) for k, v in (val_metrics or {}).items()},
    }


def restore_bundle(bundle, model, scheduler_cb, args):
    """In-place restore of model weights, CL state, counters and RNG."""
    if bundle.get("format_version") != BUNDLE_VERSION:
        raise ValueError(
            f"Bundle format_version={bundle.get('format_version')} "
            f"!= supported {BUNDLE_VERSION}"
        )

    device = model.device
    model.load_state_dict(bundle["model_state_dict"])

    model.global_step_counter = int(bundle.get("global_step_counter", 0))
    model.alignment_step_counter = int(bundle.get("alignment_step_counter", 0))
    model.completed_tasks = set(bundle.get("completed_tasks", []))

    cl_state = bundle.get("cl_state", {})
    scheduler_cb.Q_memory = (
        None if cl_state.get("Q") is None else cl_state["Q"].to(device)
    )
    scheduler_cb.Lambda_memory = (
        None if cl_state.get("Lambda") is None else cl_state["Lambda"].to(device)
    )

    backbone_params = model.get_backbone_params_dict()
    scheduler_cb.fisher_memory = _unkey_by_name(cl_state.get("fisher"), backbone_params, device)
    scheduler_cb.theta_star_memory = _unkey_by_name(cl_state.get("theta_star"), backbone_params, device)
    scheduler_cb.replay_buffer = _restore_replay_buffer(cl_state.get("replay_buffer"))
    scheduler_cb.teacher_memory = _restore_teacher(model, cl_state.get("teacher"), device)

    rng = bundle.get("rng")
    if rng is not None:
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])

    args.resume_after_idx = int(bundle.get("resume_after_idx", 0))
    # All Phase-1 head_align epochs happened before the bundle was written; the
    # aligned heads are already inside model_state_dict. Drop them (same trick as
    # the Phase-1 cache load).
    args.epochs_phase1 = 0
    return bundle.get("wandb_run_id")


# ---------------------------------------------------------------------------
# Per-task bundle housekeeping
# ---------------------------------------------------------------------------

def write_progress(directory, bundle):
    _write_json_atomic(
        os.path.join(directory, PROGRESS_FILE),
        {
            "format_version": bundle["format_version"],
            "key_hash": bundle["key_hash"],
            "exp_name": bundle["exp_name"],
            "tasks": bundle["tasks"],
            "completed_tasks": bundle["completed_tasks"],
            "resume_after_idx": bundle["resume_after_idx"],
            "global_step_counter": bundle["global_step_counter"],
            "wandb_run_id": bundle.get("wandb_run_id"),
            "timestamp": bundle["timestamp"],
            "val_metrics": bundle.get("val_metrics", {}),
        },
    )


def snapshot_task_copy(directory, latest_path, task_idx, keep=2):
    """Mirror latest.pt to task_{idx}.pt and prune to the newest `keep` copies."""
    task_copy = os.path.join(directory, f"task_{task_idx}.pt")
    shutil.copy2(latest_path, task_copy)

    copies = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.startswith("task_") and name.endswith(".pt")
    ]
    copies.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for stale in copies[keep:]:
        try:
            os.remove(stale)
        except OSError:
            pass
    return task_copy
