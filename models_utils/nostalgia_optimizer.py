import torch
import lightning.pytorch as pl
from transformers import get_linear_schedule_with_warmup
from models_utils.language_model import LanguageModelModule
from utils.nostalgia import NostalgiaOptimizer
from baselines import get_baseline, is_baseline
from baselines.ewc import EWCOptimizer
from baselines.agem import AGEMOptimizer


class NostalgiaLanguageModelModule(LanguageModelModule):
    def __init__(
        self,
        model_name="gpt2",
        lr=5e-5,
        head_lr=None,
        warmup_steps=100,
        total_steps=1000,
        log_every=50,
        writer=None,
        tasks_config=None,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        quantization=None,
        method="nostalgia",
        pooling="last",
        run_debug_checks=False,
        precision="32-true",
        base_optimizer_name="adamw",
        sgd_momentum=0.9,
        weight_decay=0.01,
        ewc_lambda=400.0,
        agem_mem_size=500,
        gpm_threshold=0.925,
    ):
        super().__init__(
            model_name=model_name,
            lr=lr,
            head_lr=head_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            tasks_config=tasks_config,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            quantization=quantization,
            pooling=pooling,
            run_debug_checks=run_debug_checks,
            precision=precision,
        )

        self.log_every = log_every
        self.writer = writer
        self.method = method
        self.base_optimizer_name = base_optimizer_name
        self.sgd_momentum = sgd_momentum
        self.weight_decay = weight_decay
        self.ewc_lambda = ewc_lambda
        self.agem_mem_size = agem_mem_size
        self.gpm_threshold = gpm_threshold

        # Per-task baseline state (populated by PhaseSchedulerCallback)
        # EWC: fisher + theta_star dicts keyed by id(p)
        # A-GEM: ReplayBuffer
        self.fisher_memory = None
        self.theta_star_memory = None
        self.replay_buffer = None

    def _build_base_optimizer(self, param_groups):
        """Instantiate the requested base optimizer."""
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

        # Split parameters into backbone and head groups
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        backbone_param_ids = {id(p) for p in backbone_params}

        # Head params: anything requiring grad that isn't in the backbone
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

        if self.training_phase == "head_align" or self.method == "naive_adam":
            # Phase 1 and naive_adam baseline use the chosen base optimizer directly.
            optimizer = self._build_base_optimizer(param_groups)
        elif self.method == "nostalgia" or self.method == "gpm":
            # Phase 2 with Nostalgia/GPM projection (same wrapper; Q source differs).
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
            )

            # Load accumulated projection space if available
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
            # Give the optimizer a reference to the model so it can run replay forward passes
            optimizer.set_model_ref(self)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.hparams.total_steps,
        )

        # Monkey-patch step to ignore any external epoch argument (which could fast-forward
        # the scheduler based on trainer.global_step) and use local task steps instead.
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

