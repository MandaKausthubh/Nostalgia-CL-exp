"""Image classification backbone + LightningModule for the main CL pipeline.

Mirrors the interface the rest of the CL machinery expects:
    - `.backbone` (nn.Module with `forward(inputs)` reading `inputs["input_ids"]`)
    - `.task_head_list` (ModuleDict keyed by task name)
    - `.active_task`, `.training_phase`, `.logging_disabled`
    - `.global_step_counter`, `.alignment_step_counter`, `.completed_tasks`
    - `.method`, `.base_optimizer_name`, `.sgd_momentum`, `.weight_decay`
    - `.log_every`, `.writer`, `.ewc_lambda`, `.agem_mem_size`, `.gpm_threshold`
    - `get_backbone_params_dict()`, `preprocess_inputs(inputs)`,
      `forward(input_ids=, attention_mask=, task_name=)`
    - `_shared_step`, `training_step`, `optimizer_step`,
      `on_validation_epoch_start`, `validation_step`, `on_validation_epoch_end`
    - `_build_base_optimizer(param_groups)`, `configure_optimizers()`

This lets `PhaseSchedulerCallback`, `baselines/*`, `utils/hessians.py`, and
`utils/nostalgia.py` drive image models through the same code paths as the LM pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import AutoConfig, AutoModel, get_linear_schedule_with_warmup
import torchvision
from torchvision import models

from utils.nostalgia import NostalgiaOptimizer
from baselines.ewc import EWCOptimizer
from baselines.ewc_nostalgia import EWCNostalgiaOptimizer
from baselines.agem import AGEMOptimizer
from baselines.sdft import (
    snapshot_teacher as _snapshot_teacher_helper,
    compute_sdft_distillation_loss,
)


# ---------------------------------------------------------------------------
# ResNet-10 backbone — residual blocks, Hessian-safe (no MaxPool, no ReLU-inplace at skip)
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Standard ResNet residual block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN, plus skip."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        if cin != cout or stride != 1:
            self.shortcut = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet10(nn.Module):
    """ResNet-10 for 32x32 image classification."""

    def __init__(self, in_channels: int = 3, feat_dim: int = 512):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = feat_dim

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.stage4 = nn.Sequential(
            ResidualBlock(256, feat_dim, stride=1),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        x = inputs["input_ids"]
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x).flatten(1)
        return x


class ResNet18(nn.Module):
    """ResNet-18 backbone for images, Hessian-safe (AvgPool2d replaces MaxPool2d)."""

    def __init__(self, in_channels: int = 3, weights="DEFAULT"):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = 512

        if weights == "DEFAULT":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        net = models.resnet18(weights=weights)
        net.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        net.fc = nn.Identity()

        if in_channels != 3:
            net.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.net = net

    def forward(self, inputs):
        x = inputs["input_ids"]
        return self.net(x)


class HfViTBackbone(nn.Module):
    """ViT-B/16 backbone from transformers ("google/vit-base-patch16-224").

    HF ViT (not torchvision) so LoRA can target the separate
    `attention.attention.query/key/value` Linear layers — torchvision's fused
    MHA stores qkv as a single raw Parameter that peft cannot wrap.
    `attn_implementation="eager"` keeps attention fully double-differentiable
    for the Hessian-vector products.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        model_name = "google/vit-base-patch16-224"
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        else:
            config = AutoConfig.from_pretrained(model_name)
            config._attn_implementation = "eager"
            self.model = AutoModel.from_config(config)
        self.feat_dim = 768

    def forward(self, inputs):
        x = inputs["input_ids"]
        out = self.model(pixel_values=x)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


class SigLIPBackbone(nn.Module):
    """SigLIP-B/16 vision backbone from transformers."""

    def __init__(self, model_name: str = "google/siglip-base-patch16-224"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.feat_dim = 768

    def forward(self, inputs):
        x = inputs["input_ids"]
        out = self.model.vision_model(pixel_values=x)
        return out.pooler_output


def _build_image_backbone(name: str, in_channels: int = 3, feat_dim: int = 512,
                          pretrained: bool = True):
    """Factory: return (backbone, feat_dim) for the requested image backbone."""
    name = name.lower()
    weights = "DEFAULT" if pretrained else None
    if name == "resnet10":
        backbone = ResNet10(in_channels=in_channels, feat_dim=feat_dim)
        return backbone, backbone.feat_dim
    if name == "resnet18":
        backbone = ResNet18(in_channels=in_channels, weights=weights)
        return backbone, backbone.feat_dim
    if name == "vit":
        backbone = HfViTBackbone(pretrained=pretrained)
        return backbone, backbone.feat_dim
    if name == "siglip":
        backbone = SigLIPBackbone()
        return backbone, backbone.feat_dim
    raise ValueError(f"Unknown image backbone: {name}. Choose from resnet10, resnet18, vit, siglip.")


def _resolve_lora_targets(backbone, candidate_sets):
    """Pick the first LoRA target-name set whose names all appear as module leaves.

    transformers refactored ViT attention (PR #41693, 2026): the projection
    layers `query`/`value` became `q_proj`/`v_proj`. peft then raises
    NoMatchingPeftModuleError on the stale names. Probe the live module tree so
    both layouts work instead of hardcoding one.
    """
    leaves = {full.rsplit(".", 1)[-1] for full, _ in backbone.named_modules()}
    for names in candidate_sets:
        if all(n in leaves for n in names):
            return list(names)
    seen = sorted(n for n in leaves if any(k in n for k in ("proj", "query", "value", "qkv")))
    raise ValueError(
        f"No LoRA target names matched for {type(backbone).__name__}; tried "
        f"{candidate_sets}. Attention-ish leaves seen: {seen[:20]}"
    )


def _apply_lora_to_backbone(backbone, backbone_name, lora_r, lora_alpha, lora_dropout):
    """Inject LoRA adapters into an image backbone in place (peft).

    Uses `inject_adapter_in_model` (NOT `get_peft_model`) so the module tree,
    forward signatures, and state-dict keys stay untouched — the CL machinery
    (Hessian flattening, functional_call HVPs, checkpointing) sees the same
    model with a few extra `lora_` parameters. peft freezes the base weights,
    so `requires_grad`-based machinery (Hessian selection, optimizer groups,
    theta_star/Fisher snapshots) auto-scopes to the adapters.
    """
    from peft import inject_adapter_in_model, LoraConfig, TaskType

    name = backbone_name.lower()
    if name in ("resnet10", "resnet18"):
        # Full dotted names of every Conv2d submodule (peft matches on
        # end-of-name, full names are unambiguous).
        target_modules = [
            f"{parent}.{leaf}" if parent else leaf
            for parent, module in backbone.named_modules()
            for leaf, child in module.named_children()
            if isinstance(child, nn.Conv2d)
        ]
    elif name == "vit":
        # pre-#41693 ViT: query/value; post-refactor: q_proj/v_proj.
        target_modules = _resolve_lora_targets(
            backbone, (("query", "value"), ("q_proj", "v_proj"))
        )
    elif name == "siglip":
        target_modules = _resolve_lora_targets(
            backbone, (("q_proj", "v_proj"), ("query", "value"))
        )
    else:
        raise ValueError(f"LoRA not supported for backbone: {backbone_name}")

    # SigLIP: inject into the vision tower ONLY. Injecting into the whole
    # SiglipModel would also hit the text tower's q/v projections —
    # trainable-but-unused params that pollute the Hessian/optimizer space.
    target_model = backbone.model.vision_model if name == "siglip" else backbone

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
    )
    inject_adapter_in_model(peft_config, target_model)

    # Freeze everything that is not a LoRA adapter — across the whole backbone
    # (including the SigLIP text tower, which receives no adapters) so the
    # requires_grad-scoped CL machinery contains adapters only.
    for param_name, p in backbone.named_parameters():
        if "lora_" not in param_name:
            p.requires_grad = False

    trainable = [(n, p.numel()) for n, p in backbone.named_parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            f"LoRA injection matched no modules for backbone={backbone_name} "
            f"(targets={target_modules})"
        )
    bad = [n for n, _ in trainable if "lora_" not in n]
    if bad:
        raise RuntimeError(
            f"LoRA injection left non-adapter params trainable for "
            f"backbone={backbone_name}: {bad[:5]}. This would silently turn "
            f"the run into full finetuning."
        )
    n_params = sum(c for _, c in trainable)
    print(f"[LoRA] backbone={backbone_name} r={lora_r} alpha={lora_alpha} "
          f"dropout={lora_dropout}: {len(trainable)} adapter tensors, "
          f"{n_params:,} trainable params (targets={len(target_modules)})", flush=True)
    return backbone


# ---------------------------------------------------------------------------
# LightningModule
# ---------------------------------------------------------------------------

class ImageModelModule(pl.LightningModule):
    """LightningModule for image classification with per-task heads."""

    def __init__(
        self,
        lr: float = 1e-3,
        head_lr: float = None,
        warmup_steps: int = 100,
        total_steps: int = 1000,
        tasks_config=None,
        method: str = "nostalgia",
        base_optimizer_name: str = "adamw",
        sgd_momentum: float = 0.9,
        weight_decay: float = 0.01,
        log_every: int = 50,
        writer=None,
        ewc_lambda: float = 400.0,
        ewc_nostalgia_lambda: float = 400.0,
        agem_mem_size: int = 500,
        gpm_threshold: float = 0.925,
        in_channels: int = 3,
        feat_dim: int = 512,
        run_debug_checks: bool = False,
        nostalgia_alpha: float = 1.0,
        backbone_name: str = "resnet10",
        image_size: int = 32,
        pretrained: bool = True,
        sdft_lambda_distillation: float = 1.0,
        sdft_temperature: float = 2.0,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        channels_last: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tasks_config = tasks_config
        self.backbone_name = backbone_name
        self.image_size = image_size

        self.backbone, actual_feat_dim = _build_image_backbone(
            backbone_name, in_channels=in_channels, feat_dim=feat_dim,
            pretrained=pretrained,
        )

        if use_lora:
            self.backbone = _apply_lora_to_backbone(
                self.backbone, backbone_name, lora_r, lora_alpha, lora_dropout,
            )

        # channels_last only helps conv backbones on CUDA; ViT/SigLIP patch-embed
        # + eager attention gain nothing, and MPS/CPU do not support the format.
        self.channels_last = bool(
            channels_last
            and backbone_name.lower() in ("resnet10", "resnet18")
            and torch.cuda.is_available()
        )
        if self.channels_last:
            self.backbone = self.backbone.to(memory_format=torch.channels_last)
            print(f"[channels_last] backbone={backbone_name} converted to "
                  f"channels_last memory format", flush=True)

        if tasks_config is not None:
            self.task_head_list = torch.nn.ModuleDict({
                task_name: nn.Linear(actual_feat_dim, num_classes)
                for task_name, num_classes in tasks_config.items()
            })
            self.criterion = torch.nn.CrossEntropyLoss()
            self.active_task = list(tasks_config.keys())[0]
        else:
            self.task_head_list = None
            self.criterion = None
            self.active_task = None

        # CRITICAL: snapshot AFTER LoRA injection. With LoRA the adapters are
        # the only trainable backbone params, so the Phase-2 name-based
        # unfreeze (phase_scheduler.py) restores adapters only. Snapshotting
        # before the injection would unfreeze full base weights = silent
        # full finetuning (guarded by the assert in _apply_lora_to_backbone).
        self.trainable_backbone_param_names = {
            name for name, p in self.backbone.named_parameters() if p.requires_grad
        }

        self.global_step_counter = 0
        self.alignment_step_counter = 0
        self.completed_tasks: set = set()
        self.training_phase = "nostalgia"
        self.logging_disabled = False

        self._val_losses_per_task = {}
        self._val_accs_per_task = {}
        self._val_preds = {}

        self.log_every = log_every
        self.writer = writer
        self.method = method
        self.base_optimizer_name = base_optimizer_name
        self.sgd_momentum = sgd_momentum
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda
        self.ewc_nostalgia_lambda = ewc_nostalgia_lambda
        self.agem_mem_size = agem_mem_size
        self.gpm_threshold = gpm_threshold
        self.run_debug_checks = run_debug_checks
        self.nostalgia_alpha = nostalgia_alpha
        self.sdft_lambda_distillation = sdft_lambda_distillation
        self.sdft_temperature = sdft_temperature

        self.Q_memory = None
        self.Lambda_memory = None
        self.fisher_memory = None
        self.theta_star_memory = None
        self.replay_buffer = None
        self.teacher_model = None

    def get_backbone_params_dict(self):
        return {name: p for name, p in self.backbone.named_parameters() if p.requires_grad}

    def preprocess_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return {"input_ids": self._to_channels_last(inputs)}
        if isinstance(inputs, dict):
            return {
                "input_ids": self._to_channels_last(inputs["input_ids"]),
                "attention_mask": inputs.get("attention_mask", None),
            }
        return inputs

    def _to_channels_last(self, x):
        if self.channels_last and torch.is_tensor(x) and x.dim() == 4:
            return x.contiguous(memory_format=torch.channels_last)
        return x

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None, **kwargs):
        t_name = task_name if task_name is not None else self.active_task
        input_ids = self._to_channels_last(input_ids)
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        representations = self.backbone(inputs)
        return self.task_head_list[t_name](representations)

    def snapshot_teacher(self):
        """Create a frozen eval copy of the current model as the SDFT teacher."""
        self.teacher_model = _snapshot_teacher_helper(self, self.device)

    def _shared_step(self, batch, stage, task_name=None):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        targets = batch.get("label", batch.get("labels", batch.get("target", batch.get("targets", None))))
        if targets is None:
            raise ValueError("Batch does not contain any label key (tried 'label', 'labels', 'target', 'targets')")

        logits = self(input_ids=input_ids, attention_mask=attention_mask, task_name=task_name)
        ce_loss = self.criterion(logits, targets)
        loss = ce_loss

        if (
            self.method == "sdft"
            and self.teacher_model is not None
            and self.training_phase != "head_align"
        ):
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    task_name=task_name,
                )
            kl = compute_sdft_distillation_loss(
                logits, teacher_logits, temperature=self.sdft_temperature
            )
            loss = loss + self.sdft_lambda_distillation * kl
            self.log(
                f"{stage}/sdft_kl_loss",
                kl,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                add_dataloader_idx=False,
            )
            self.log(
                f"{stage}/sdft_total_loss",
                loss,
                prog_bar=False,
                sync_dist=True,
                on_step=True,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        preds = torch.argmax(logits, dim=-1)
        acc = (preds == targets).float().mean()

        # Log CE loss as the canonical {stage}/loss so all methods are comparable.
        # For SDFT the distillation-augmented loss is still used for backprop and logged separately.
        if stage.endswith("/train") or stage.endswith("/alignment"):
            self.log(f"{stage}/loss", ce_loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
        else:
            self.log(f"{stage}/loss", ce_loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)

        self._last_logits = logits
        self._last_targets = targets
        return loss, acc

    def training_step(self, batch, batch_idx):
        task_name = self.active_task
        stage = f"{task_name}/alignment" if self.training_phase == "head_align" else f"{task_name}/train"
        loss, acc = self._shared_step(batch, stage, task_name=task_name)

        if self.run_debug_checks:
            if batch_idx == 0 and self.trainer.is_global_zero:
                targets = getattr(self, "_last_targets", None)
                if targets is not None:
                    print(f"\n  [LABEL DISTRIBUTION CHECK epoch={self.current_epoch}] task={task_name!r}")
                    print(f"    min target: {targets.min().item()}")
                    print(f"    max target: {targets.max().item()}")
                    print(f"    bincount  : {torch.bincount(targets).tolist()}")

            logits = getattr(self, "_last_logits", None)
            if self.trainer.is_global_zero and self.global_step_counter % 200 == 0 and logits is not None:
                print(f"\n  [LOGITS CHECK step={self.global_step_counter}]")
                print(f"    mean: {logits.mean().item():.4f}")
                print(f"    std : {logits.std().item():.4f}")
                print(f"    max : {logits.max().item():.4f}")
                print(f"    min : {logits.min().item():.4f}")

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        should_check_param = False
        old_val = None
        target_param = None

        if self.run_debug_checks:
            head = self.task_head_list[self.active_task]
            if self.trainer.is_global_zero and self.global_step_counter % 100 == 0:
                print(f"\n  [GRADIENT CHECK step={self.global_step_counter}] task={self.active_task}")
                for n, p in head.named_parameters():
                    grad_norm = p.grad.norm().item() if p.grad is not None else "None"
                    print(f"    {n:<20} grad_norm={grad_norm}")
                print()

            if hasattr(head, "weight"):
                target_param = head.weight
            else:
                for p in head.parameters():
                    if p.requires_grad:
                        target_param = p
                        break

            should_check_param = (
                self.trainer.is_global_zero
                and self.global_step_counter % 100 == 0
                and target_param is not None
            )
            if should_check_param:
                old_val = target_param.clone()

            if self.trainer.is_global_zero and self.global_step_counter % 200 == 0:
                print(f"\n  [LR CHECK step={self.global_step_counter}] lr={optimizer.param_groups[0]['lr']:.2e}\n")

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
        if self.training_phase == "nostalgia":
            self.global_step_counter += 1
        else:
            self.alignment_step_counter += 1

        if should_check_param and old_val is not None:
            diff = (target_param - old_val).norm().item()
            print(f"  [PARAMETER UPDATE CHECK step={self.global_step_counter}]")
            print(f"    head param update norm: {diff:.6e}\n")

    def on_validation_epoch_start(self):
        self._val_losses_per_task = {}
        self._val_accs_per_task = {}
        self._val_preds = {}

    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        # Lightning default does `.to(device)` WITHOUT non_blocking, which serialises
        # H2D copies on the compute stream. With pin_memory=True on the loader this
        # kills throughput. Override to issue true async copies.
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.logging_disabled:
            return
        val_task_names = getattr(self, "val_task_names", None)
        if val_task_names and dataloader_idx < len(val_task_names):
            task_name = val_task_names[dataloader_idx]
        else:
            task_name = self.active_task
            if val_task_names is not None:
                print(
                    f"WARNING: dataloader_idx={dataloader_idx} is out of bounds "
                    f"for val_task_names={val_task_names}. Falling back to active_task={task_name}"
                )

        stage = f"{task_name}/validation"
        loss, acc = self._shared_step(batch, stage, task_name=task_name)
        if loss is not None:
            self._val_losses_per_task.setdefault(task_name, []).append(loss.detach())
        if acc is not None:
            acc_t = acc.detach() if isinstance(acc, torch.Tensor) else torch.tensor(acc, device=self.device)
            self._val_accs_per_task.setdefault(task_name, []).append(acc_t)

        logits = getattr(self, "_last_logits", None)
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)
            if task_name not in self._val_preds:
                self._val_preds[task_name] = []
            self._val_preds[task_name].append(preds.detach().cpu())

    def on_validation_epoch_end(self):
        if self.logging_disabled:
            return

        started = set(self.completed_tasks or []) | {self.active_task}
        task_losses = []
        task_accs = []
        for task_name, losses in getattr(self, "_val_losses_per_task", {}).items():
            if task_name not in started or not losses:
                continue
            task_losses.append(torch.stack(losses).mean())
        for task_name, accs in getattr(self, "_val_accs_per_task", {}).items():
            if task_name not in started or not losses:
                continue
            task_accs.append(torch.stack(accs).mean())

        if task_losses:
            self.log("total/validation/loss", torch.stack(task_losses).mean(),
                     prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)
        if task_accs:
            self.log("total/validation/acc", torch.stack(task_accs).mean(),
                     prog_bar=True, sync_dist=True, on_step=False, on_epoch=True)

        if hasattr(self, "_val_losses_per_task") and self._val_losses_per_task:
            self._val_losses_per_task.clear()
        if hasattr(self, "_val_accs_per_task") and self._val_accs_per_task:
            self._val_accs_per_task.clear()

        if hasattr(self, "_val_preds") and self._val_preds:
            if self.run_debug_checks and self.trainer.is_global_zero:
                print("\n  [VAL PREDICTION FREQUENCIES]")
                for task_name, preds_list in self._val_preds.items():
                    all_preds = torch.cat(preds_list)
                    counts = torch.bincount(all_preds)
                    print(f"    Task {task_name:<10}: {counts.tolist()}")
                print()
            self._val_preds.clear()

    def _build_base_optimizer(self, param_groups):
        name = self.base_optimizer_name.lower()
        if name == "adam":
            return torch.optim.Adam(param_groups)
        if name == "adamw":
            return torch.optim.AdamW(param_groups)
        if name == "sgd":
            return torch.optim.SGD(
                param_groups,
                momentum=self.sgd_momentum,
                nesterov=False,
            )
        raise ValueError(f"Unknown base optimizer: {self.base_optimizer_name}")

    def configure_optimizers(self):
        head_lr = self.hparams.head_lr if getattr(self.hparams, "head_lr", None) is not None else self.hparams.lr
        weight_decay = getattr(self, "weight_decay", 0.01)

        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        backbone_param_ids = {id(p) for p in backbone_params}
        head_params = [p for p in self.parameters() if p.requires_grad and id(p) not in backbone_param_ids]

        param_groups = []
        if backbone_params:
            param_groups.append({
                "params": backbone_params,
                "lr": self.hparams.lr,
                "weight_decay": weight_decay,
            })
        if head_params:
            param_groups.append({
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
            })

        if not param_groups:
            param_groups = [{"params": [p for p in self.parameters() if p.requires_grad]}]

        if self.training_phase == "head_align" or self.method == "naive_adam" or self.method == "sdft":
            optimizer = self._build_base_optimizer(param_groups)
        elif self.method == "nostalgia" or self.method == "gpm":
            base_optimizer = self._build_base_optimizer(param_groups)
            proj_params = [p for p in self.backbone.parameters() if p.requires_grad]
            optimizer = NostalgiaOptimizer(
                params=proj_params,
                base_optimizer=base_optimizer,
                device=self.device,
                dtype=next(self.parameters()).dtype,
                writter=self.writer,
                starting_step=self.global_step_counter,
                log_every=self.log_every,
                alpha=self.nostalgia_alpha,
            )
            if getattr(self, "Q_memory", None) is not None:
                optimizer.set_Q(self.Q_memory, self.Lambda_memory)
        elif self.method == "ewc":
            base_optimizer = self._build_base_optimizer(param_groups)
            proj_params = [p for p in self.backbone.parameters() if p.requires_grad]
            optimizer = EWCOptimizer(
                params=proj_params,
                base_optimizer=base_optimizer,
                device=self.device,
                dtype=next(self.parameters()).dtype,
                fisher=self.fisher_memory,
                theta_star=self.theta_star_memory,
                lam=self.ewc_lambda,
                writer=self.writer,
                log_every=self.log_every,
                starting_step=self.global_step_counter,
            )
        elif self.method == "ewc_nostalgia":
            base_optimizer = self._build_base_optimizer(param_groups)
            proj_params = [p for p in self.backbone.parameters() if p.requires_grad]
            optimizer = EWCNostalgiaOptimizer(
                params=proj_params,
                base_optimizer=base_optimizer,
                device=self.device,
                dtype=next(self.parameters()).dtype,
                Q=self.Q_memory,
                Lambda=self.Lambda_memory,
                theta_star=self.theta_star_memory,
                lam=self.ewc_nostalgia_lambda,
                writer=self.writer,
                log_every=self.log_every,
                starting_step=self.global_step_counter,
            )
        elif self.method == "agem":
            base_optimizer = self._build_base_optimizer(param_groups)
            proj_params = [p for p in self.backbone.parameters() if p.requires_grad]
            optimizer = AGEMOptimizer(
                params=proj_params,
                base_optimizer=base_optimizer,
                device=self.device,
                dtype=next(self.parameters()).dtype,
                replay_buffer=self.replay_buffer,
                replay_bs=min(self.agem_mem_size, 8),
                writer=self.writer,
                log_every=self.log_every,
                starting_step=self.global_step_counter,
            )
            optimizer.set_model_ref(self)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )

        scheduler._local_step = 0
        original_step = scheduler.step

        def custom_step(epoch=None):
            scheduler._local_step += 1
            return original_step(epoch=scheduler._local_step)

        scheduler.step = custom_step

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
