import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup

class BackboneWrapper(torch.nn.Module):
    def __init__(self, base_model, pooling="last", tokenizer=None):
        super().__init__()
        self.base_model = base_model
        self.pooling = pooling
        self.tokenizer = tokenizer

    def forward(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # outputs[0] is logits for CausalLM models (vocab-sized), not hidden states.
        # Use the last layer's hidden state instead.
        last_hidden_state = outputs.hidden_states[-1]  # shape: [batch, seq_len, hidden_dim]
        
        if self.pooling == "last":
            if attention_mask is not None:
                # find last non-padded token index for each sequence in the batch
                # argmax finds the first 0, so argmax - 1 is the last 1.
                # If all are 1, argmax is 0, so 0 - 1 = -1, which is the last token.
                sequence_lengths = torch.eq(attention_mask, 0).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % attention_mask.shape[-1]
                pooled = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]
            else:
                pooled = last_hidden_state[:, -1]
        elif self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = last_hidden_state.mean(dim=1)
        elif self.pooling == "eos":
            pooled = last_hidden_state[:, -1]
        elif self.pooling == "first":
            pooled = last_hidden_state[:, 0]
        elif self.pooling == "before_category":
            if self.tokenizer is not None:
                indices = []
                batch_size, seq_len = input_ids.shape
                # Pre-tokenize the markers
                marker_ids_list = []
                for marker in ["### Category:", "### Sentiment:", "###"]:
                    ids = self.tokenizer.encode(marker, add_special_tokens=False)
                    if ids:
                        marker_ids_list.append(ids)
                
                for i in range(batch_size):
                    row = input_ids[i].tolist()
                    found_idx = -1
                    for marker_ids in marker_ids_list:
                        m_len = len(marker_ids)
                        for j in range(seq_len - m_len + 1):
                            if row[j:j+m_len] == marker_ids:
                                found_idx = j
                                break
                        if found_idx != -1:
                            break
                    if found_idx > 0:
                        indices.append(found_idx - 1)
                    else:
                        if attention_mask is not None:
                            last_idx = torch.eq(attention_mask[i], 0).int().argmax(-1).item() - 1
                            last_idx = last_idx % seq_len
                            indices.append(last_idx)
                        else:
                            indices.append(seq_len - 1)
                indices = torch.tensor(indices, device=last_hidden_state.device)
                pooled = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), indices]
            else:
                if attention_mask is not None:
                    sequence_lengths = torch.eq(attention_mask, 0).int().argmax(-1) - 1
                    sequence_lengths = sequence_lengths % attention_mask.shape[-1]
                    pooled = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), sequence_lengths]
                else:
                    pooled = last_hidden_state[:, -1]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
        return pooled

def _build_head(hidden_dim: int, num_classes: int, num_layers: int = 1) -> nn.Module:
    """Build a classification head with the given number of layers.
    
    Args:
        hidden_dim: Input dimension from backbone.
        num_classes: Number of output classes.
        num_layers: Number of layers (default 1 = single linear).
    
    Returns:
        nn.Module: A single Linear or an nn.Sequential with ReLU activations.
    """
    if num_layers <= 1:
        return nn.Linear(hidden_dim, num_classes)
    
    layers = []
    for i in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*layers)


class LanguageModelModule(pl.LightningModule):
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        lr=5e-5,
        head_lr=None,
        warmup_steps=100,
        total_steps=1000,
        tasks_config=None,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        quantization=None,
        pooling="last",
        run_debug_checks=False,
        precision="32-true",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.pooling = pooling
        
        # Configure quantization
        bnb_config = None
        if quantization == "4bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if precision in ("bf16", "bf16-mixed") else torch.float32,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization == "8bit":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        # Determine torch_dtype based on precision
        torch_dtype = torch.float32
        if precision in ("bf16", "bf16-mixed"):
            torch_dtype = torch.bfloat16
        elif precision in ("16", "16-mixed"):
            torch_dtype = torch.float16

        load_kwargs = {
            "torch_dtype": torch_dtype,
            "attn_implementation": "eager"
        }
        if bnb_config is not None:
            load_kwargs["quantization_config"] = bnb_config
            # quantization requires device placement on load
            load_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        
        # Configure PEFT / LoRA
        if use_lora:
            from peft import get_peft_model, LoraConfig, TaskType
            if bnb_config is not None:
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model)
                
            target_modules = ["c_attn"] if "gpt2" in model_name else ["q_proj", "v_proj"]
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
            self.model = get_peft_model(self.model, peft_config)
        
        if tasks_config is not None:
            self.tasks_config = tasks_config
            hidden_dim = self.model.config.hidden_size
            self.task_head_list = torch.nn.ModuleDict({
                task_name: _build_head(hidden_dim, num_classes, num_layers)
                for task_name, (num_classes, num_layers) in tasks_config.items()
            })
            self.criterion = torch.nn.CrossEntropyLoss()
            self.active_task = list(tasks_config.keys())[0]
            # Use raw base model inside PeftModel if PEFT is wrapped
            base_for_wrapper = self.model.base_model if hasattr(self.model, "base_model") else self.model
            
            # Instantiate tokenizer for parsing markers if needed
            from transformers import AutoTokenizer
            self.tokenizer_instance = AutoTokenizer.from_pretrained(model_name)
            
            self.backbone = BackboneWrapper(base_for_wrapper, pooling=self.pooling, tokenizer=self.tokenizer_instance)
        else:
            self.tasks_config = None
            self.backbone = self.model
            
        # Record names of backbone parameters that require gradients at initialization
        self.trainable_backbone_param_names = {
            name for name, p in self.backbone.named_parameters() if p.requires_grad
        }

        # Global step counter persisted across trainer fits (Phase 2 only)
        self.global_step_counter = 0
        # Per-task alignment step counter (resets at each Phase 1 start)
        self.alignment_step_counter = 0
        # Track which tasks have been fully trained (completed Phase 2)
        self.completed_tasks: set = set()
        # Training phase flag: "head_align" or "nostalgia"
        self.training_phase = "nostalgia"
        # Phase 3 temporarily disables all logging while Hessians are computed.
        self.logging_disabled = False

        self._val_losses = []
        self._val_accs = []

    def get_backbone_params_dict(self):
        # return dict(self.backbone.named_parameters())
        return {name: p for name, p in self.backbone.named_parameters() if p.requires_grad}

    def preprocess_inputs(self, inputs):
        if isinstance(inputs, torch.Tensor):
            return {"input_ids": inputs}
        elif isinstance(inputs, dict):
            return {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs.get("attention_mask", None)
            }
        return inputs

    def forward(self, input_ids, attention_mask=None, labels=None, task_name=None, **kwargs):
        if self.tasks_config is not None:
            t_name = task_name if task_name is not None else self.active_task
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            representations = self.backbone(inputs)
            logits = self.task_head_list[t_name](representations)
            return logits
        else:
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def _shared_step(self, batch, stage, task_name=None):
        if self.tasks_config is not None:
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            targets = batch.get("label", batch.get("labels", batch.get("target", batch.get("targets", None))))
            if targets is None:
                raise ValueError("Batch does not contain any label key (tried 'label', 'labels', 'target', 'targets')")
                
            logits = self(input_ids=input_ids, attention_mask=attention_mask, task_name=task_name)
            loss = self.criterion(logits, targets)
            
            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            acc = (preds == targets).float().mean()
            
            if stage.endswith("/train") or stage.endswith("/alignment"):
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
                self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            else:
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f"{stage}/acc", acc, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            
            self._last_logits = logits
            self._last_targets = targets
            return loss, acc
        else:
            outputs = self(**batch)
            loss = outputs.loss
            ppl  = torch.exp(loss)          # perplexity — the language model "accuracy"
            if stage.endswith("/train"):
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
                self.log(f"{stage}/perplexity", ppl, prog_bar=True, sync_dist=True, on_step=True, on_epoch=True)
            else:
                self.log(f"{stage}/loss", loss, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
                self.log(f"{stage}/perplexity", ppl, prog_bar=True, sync_dist=True, on_step=False, on_epoch=True, add_dataloader_idx=False)
            
            self._last_logits = None
            self._last_targets = None
            return loss, None

    def training_step(self, batch, batch_idx):
        task_name = self.active_task

        if self.training_phase == "head_align":
            # Phase 1: log as {task}/alignment, use alignment counter
            stage = f"{task_name}/alignment"
            loss, acc = self._shared_step(batch, stage, task_name=task_name)
        else:
            # Phase 2: log as {task}/train, use global counter
            stage = f"{task_name}/train"
            loss, acc = self._shared_step(batch, stage, task_name=task_name)

        logits = getattr(self, "_last_logits", None)
        targets = getattr(self, "_last_targets", None)

        # Item 2, 3, 5: Debug checks (only if run_debug_checks is True)
        if getattr(self.hparams, "run_debug_checks", False):
            if batch_idx == 0 and self.trainer.is_global_zero:
                if targets is not None:
                    print(f"\n  [LABEL DISTRIBUTION CHECK epoch={self.current_epoch}] task={task_name!r}")
                    print(f"    min target: {targets.min().item()}")
                    print(f"    max target: {targets.max().item()}")
                    print(f"    bincount  : {torch.bincount(targets).tolist()}")

            if task_name == "agnews" and batch_idx == 0 and self.trainer.is_global_zero:
                if logits is not None and targets is not None:
                    preds = logits.argmax(dim=-1)
                    print(f"\n  [PREDICTION CHECK agnews epoch={self.current_epoch}]")
                    print("    targets:", targets[:20].tolist())
                    print("    preds  :", preds[:20].tolist())

            if self.trainer.is_global_zero and self.global_step_counter % 200 == 0:
                if logits is not None:
                    print(f"\n  [LOGITS CHECK step={self.global_step_counter}]")
                    print(f"    mean: {logits.mean().item():.4f}")
                    print(f"    std : {logits.std().item():.4f}")
                    print(f"    max : {logits.max().item():.4f}")
                    print(f"    min : {logits.min().item():.4f}")

        return loss

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure, **kwargs):
        # Item 7, 8, 9: Verify gradients, weights update, and learning rate (only if run_debug_checks is True)
        should_check_param = False
        old_val = None
        target_param = None

        if getattr(self.hparams, "run_debug_checks", False):
            if self.tasks_config is not None:
                head = self.task_head_list[self.active_task]
                if self.trainer.is_global_zero and self.global_step_counter % 100 == 0:
                    print(f"\n  [GRADIENT CHECK step={self.global_step_counter}] task={self.active_task}")
                    for n, p in head.named_parameters():
                        grad_norm = p.grad.norm().item() if p.grad is not None else "None"
                        print(f"    {n:<20} grad_norm={grad_norm}")
                    print()

                # Item 8: Before optimizer step
                if hasattr(head, "weight"):
                    target_param = head.weight
                else:
                    for p in head.parameters():
                        if p.requires_grad:
                            target_param = p
                            break
                
                should_check_param = self.trainer.is_global_zero and self.global_step_counter % 100 == 0 and target_param is not None
                if should_check_param:
                    old_val = target_param.clone()

            # Item 9: Verify the learning rate
            if self.trainer.is_global_zero and self.global_step_counter % 200 == 0:
                print(f"\n  [LR CHECK step={self.global_step_counter}] lr={optimizer.param_groups[0]['lr']:.2e}\n")

        # Step optimizer
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
        if self.training_phase == "nostalgia":
            self.global_step_counter += 1
        else:
            self.alignment_step_counter += 1

        # Item 8: After optimizer step
        if should_check_param and old_val is not None:
            diff = (target_param - old_val).norm().item()
            print(f"  [PARAMETER UPDATE CHECK step={self.global_step_counter}]")
            print(f"    head param update norm: {diff:.6e}\n")

    def on_validation_epoch_start(self):
        self._val_losses = []
        self._val_accs = []
        self._val_preds = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if self.logging_disabled:
            return
        # Validation only runs during Phase 2 (Phase 1 sets limit_val_batches=0)
        val_task_names = getattr(self, "val_task_names", None)
        if val_task_names and dataloader_idx < len(val_task_names):
            task_name = val_task_names[dataloader_idx]
        else:
            task_name = self.active_task
            if val_task_names is not None:
                print(f"WARNING: dataloader_idx={dataloader_idx} is out of bounds for val_task_names={val_task_names}. Falling back to active_task={task_name}")

        stage = f"{task_name}/validation"
        loss, acc = self._shared_step(batch, stage, task_name=task_name)
        if loss is not None:
            self._val_losses.append(loss.detach())
        if acc is not None:
            self._val_accs.append(acc.detach() if isinstance(acc, torch.Tensor) else torch.tensor(acc, device=self.device))

        logits = getattr(self, "_last_logits", None)
        targets = getattr(self, "_last_targets", None)

        # Item 10: Accumulate validation predictions
        if logits is not None:
            preds = torch.argmax(logits, dim=-1)
            if task_name not in self._val_preds:
                self._val_preds[task_name] = []
            self._val_preds[task_name].append(preds.detach().cpu())

    def on_validation_epoch_end(self):
        if self.logging_disabled:
            return

        # Per-task validation metrics are logged inside validation_step via _shared_step.
        # Keep internal buffers clear for the next validation epoch.
        if hasattr(self, "_val_losses") and self._val_losses:
            self._val_losses.clear()
        if hasattr(self, "_val_accs") and self._val_accs:
            self._val_accs.clear()

        # Item 10: Print per-class prediction frequencies (only if run_debug_checks is True)
        if hasattr(self, "_val_preds") and self._val_preds:
            if getattr(self.hparams, "run_debug_checks", False) and self.trainer.is_global_zero:
                print("\n  [VAL PREDICTION FREQUENCIES]")
                for task_name, preds_list in self._val_preds.items():
                    all_preds = torch.cat(preds_list)
                    counts = torch.bincount(all_preds)
                    print(f"    Task {task_name:<10}: {counts.tolist()}")
                print()
            self._val_preds.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            opt,
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

        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

