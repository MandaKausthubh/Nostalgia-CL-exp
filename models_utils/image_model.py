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
from transformers import AutoModel, get_linear_schedule_with_warmup
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

    def __init__(self, in_channels: int = 3, weights=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_dim = 512

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


class ViTBackbone(nn.Module):
    """ViT-B/16 backbone from torchvision, returning the CLS token."""

    def __init__(self, weights=None):
        super().__init__()
        self.vit = models.vit_b_16(weights=weights)
        self.vit.heads = nn.Identity()
        self.feat_dim = 768

    def forward(self, inputs):
        x = inputs["input_ids"]
        return self.vit(x)


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


def _build_image_backbone(name: str, in_channels: int = 3, feat_dim: int = 512):
    """Factory: return (backbone, feat_dim) for the requested image backbone."""
    name = name.lower()
    if name == "resnet10":
        backbone = ResNet10(in_channels=in_channels, feat_dim=feat_dim)
        return backbone, backbone.feat_dim
    if name == "resnet18":
        backbone = ResNet18(in_channels=in_channels)
        return backbone, backbone.feat_dim
    if name == "vit":
        backbone = ViTBackbone()
        return backbone, backbone.feat_dim
    if name == "siglip":
        backbone = SigLIPBackbone()
        return backbone, backbone.feat_dim
    raise ValueError(f"Unknown image backbone: {name}. Choose from resnet10, resnet18, vit, siglip.")


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
        sdft_lambda_distillation: float = 1.0,
        sdft_temperature: float = 2.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tasks_config = tasks_config
        self.backbone_name = backbone_name
        self.image_size = image_size

        self.backbone, actual_feat_dim = _build_image_backbone(
            backbone_name, in_channels=in_channels, feat_dim=feat_dim
        )

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
            return {"input_ids": inputs}
        if isinstance(inputs, dict):
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask", None),
            }
        return inputs

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None, **kwargs):
        t_name = task_name if task_name is not None else self.active_task
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
