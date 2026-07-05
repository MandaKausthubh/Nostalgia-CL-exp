import torch
import pytest

from models_utils.language_model import LanguageModelModule
from models_utils.nostalgia_optimizer import NostalgiaLanguageModelModule

from datasets_utils.cnn_datast import CNNDailyMailDataModule
from datasets_utils.squad_dataset import SQuADv2DataModule
from datasets_utils.base_class import BaseTextDataModule
from utils.hessians import compute_Q_for_task
from utils.accumulate import accumulate_hessian_eigenspace_stable


def _inspect_training(
    name: str,
    dm: BaseTextDataModule,
    model_name: str = "gpt2",
):
    """
    Smoke test whether a model can actually train
    with a given DataModule.
    """

    print(f"\n{'─' * 60}")
    print(f"TRAINING SMOKE TEST: {name}")
    print(f"model = {model_name}")

    # Setup datamodule
    dm.setup()

    # Grab one batch
    loader = (dm.train_dataloader())

    batch = next(iter(loader))
    shapes = {k: tuple(v.shape) for k, v in batch.items()}
    print(f"batch shapes: {shapes}")

    # Create model
    model = LanguageModelModule(
        model_name=model_name,
        lr=5e-5,
        warmup_steps=10,
        total_steps=100,
    )

    model.train()

    # ---- FORWARD ----
    loss = None

    for batch in loader:
        outputs = model(**batch)
        assert hasattr(outputs, "loss")
        if loss is None:
            loss = outputs.loss
        else:
            loss += outputs.loss

    assert loss is not None
    assert torch.isfinite(loss)

    print(f"forward ✓ loss={loss.item():.4f}")

    # ---- BACKWARD ----
    loss.backward()

    # verify gradients exist
    grads_found = False
    for p in model.parameters():
        if p.grad is not None:
            grads_found = True
            break

    assert grads_found, "No gradients computed"

    print("backward ✓ gradients computed")

    # ---- OPTIMIZER STEP ----
    config = model.configure_optimizers()

    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    print("optimizer step ✓")
    print("scheduler step ✓")

    print(f"SUCCESS: {name} is trainable with {model_name}")


@pytest.mark.slow
def test_cnn_dailymail_training_smoke():
    _inspect_training(
        "pszemraj/cnn_dailymail-cleaned",
        CNNDailyMailDataModule(
            model_name="gpt2",
            max_length=512,  # lower for smoke test speed
            batch_size=2,
            max_train_samples=32,
            max_val_samples=8,
        ),
        model_name="Qwen/Qwen2.5-0.5B",
    )


@pytest.mark.slow
def test_squad_v2_training_smoke():
    _inspect_training(
        "rajpurkar/squad_v2",
        SQuADv2DataModule(
            model_name="gpt2",
            max_length=512,
            batch_size=2,
            max_train_samples=32,
            max_val_samples=8,
        ),
        model_name="Qwen/Qwen2.5-0.5B",
    )

def test_classification_training_smoke():
    """
    Smoke test to verify LanguageModelModule and NostalgiaLanguageModelModule
    can train with task-specific classification heads.
    """
    print("\n" + "─" * 60)
    print("TRAINING SMOKE TEST: Multi-Task Classification")
    
    tasks_config = {
        "sentiment": (2, 1),  # (num_classes, num_layers)
        "topic": (4, 1),
    }
    
    # Create models (use gpt2 for faster smoke tests)
    from models_utils.nostalgia_optimizer import NostalgiaLanguageModelModule
    
    model = LanguageModelModule(
        model_name="gpt2",
        lr=5e-5,
        warmup_steps=5,
        total_steps=10,
        tasks_config=tasks_config,
    )
    
    nostalgia_model = NostalgiaLanguageModelModule(
        model_name="gpt2",
        lr=5e-5,
        warmup_steps=5,
        total_steps=10,
        tasks_config=tasks_config,
    )

    # Let's verify classification head parameters and backbone parameters
    assert hasattr(model, "task_head_list")
    assert "sentiment" in model.task_head_list
    assert "topic" in model.task_head_list
    assert isinstance(model.task_head_list["sentiment"], torch.nn.Linear)
    
    # Check that backbone is BackboneWrapper
    from models_utils.language_model import BackboneWrapper
    assert isinstance(model.backbone, BackboneWrapper)
    
    # Check backbone params dict
    backbone_params = model.get_backbone_params_dict()
    assert len(backbone_params) > 0
    # verify that classification heads are NOT in backbone params
    for name in backbone_params.keys():
        assert "task_head_list" not in name

    # Generate dummy batch
    batch_size = 4
    seq_len = 16
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        "label": torch.randint(0, 2, (batch_size,)),
    }
    
    # Test LanguageModelModule forward and backward
    model.train()
    model.active_task = "sentiment"
    loss, _ = model._shared_step(batch, "train")
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    
    # check grads exist on backbone and active head, but NOT on inactive head
    for name, p in model.named_parameters():
        if "task_head_list.topic" in name:
            assert p.grad is None or (p.grad == 0).all()
        else:
            assert p.grad is not None, f"Parameter {name} has no gradient"
            
    print("LanguageModelModule classification forward/backward ✓")
    
    # Test NostalgiaLanguageModelModule configure_optimizers
    config = nostalgia_model.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]
    
    # Verify that projection params only contain backbone params (length should match backbone)
    assert len(optimizer.projection_params) == len(list(nostalgia_model.backbone.parameters()))
    
    # Test step
    nostalgia_model.train()
    nostalgia_model.active_task = "topic"
    batch_topic = {
        "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
        "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long),
        "target": torch.randint(0, 4, (batch_size,)),
    }
    
    optimizer.zero_grad()
    loss_topic, _ = nostalgia_model._shared_step(batch_topic, "train")
    loss_topic.backward()
    
    # Set Q for projection
    param_dim = sum(p.numel() for p in nostalgia_model.backbone.parameters())
    k = 5
    Q = torch.randn(param_dim, k, device=nostalgia_model.device)
    # orthonormalize Q
    Q, _ = torch.linalg.qr(Q)
    scaling = torch.ones(k, device=nostalgia_model.device)
    optimizer.set_Q(Q, scaling)
    
    optimizer.step()
    scheduler.step()
    
    print("NostalgiaLanguageModelModule optimizer step with classification task ✓")
    print("All classification smoke tests passed successfully!")


def test_sequential_tasks_smoke():
    """
    Smoke test to verify sequential learning across multiple tasks using a single model.
    It simulates training, Hessian computation, eigenspace accumulation/merging,
    and gradient projection for two sequential tasks.
    """
    print("\n" + "─" * 60)
    print("TRAINING SMOKE TEST: Sequential Tasks")

    # 1. Setup multi-task configurations
    tasks_config = {
        "sentiment": (2, 1),  # (num_classes, num_layers)
        "topic": (4, 1),
    }

    # 2. Instantiate NostalgiaLanguageModelModule with a lightweight model (gpt2)
    model = NostalgiaLanguageModelModule(
        model_name="gpt2",
        lr=5e-5,
        warmup_steps=5,
        total_steps=10,
        tasks_config=tasks_config,
    )
    
    device = model.device
    
    # Verify backbone parameters dict matches model.backbone
    backbone_params = model.get_backbone_params_dict()
    assert len(backbone_params) > 0

    # 3. Setup optimizer and scheduler
    config = model.configure_optimizers()
    optimizer = config["optimizer"]
    scheduler = config["lr_scheduler"]["scheduler"]

    # 4. Initialize sequential task tracking
    Q_memory = None
    Lambda_memory = None
    k = 5  # rank cap for projection space
    batch_size = 4
    seq_len = 16

    # 5. Loop over the tasks sequentially
    for task_idx, (task_name, (num_classes, _)) in enumerate(tasks_config.items(), start=1):
        print(f"\nTraining sequentially on Task {task_idx}: '{task_name}'")
        model.active_task = task_name

        # Create dummy batch for active task training
        # Training batch is a dictionary containing target labels
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len), device=device),
            "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.long, device=device),
            "target": torch.randint(0, num_classes, (batch_size,), device=device),
        }

        # Run 2 training steps
        model.train()
        for step in range(2):
            optimizer.zero_grad()
            loss, _ = model._shared_step(batch, "train")
            assert loss is not None
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            print(f"  Step {step + 1} loss: {loss.item():.4f}")

        # Verify that gradients exist on active head and backbone, but NOT on inactive heads
        for name, p in model.named_parameters():
            if "task_head_list" in name:
                if f"task_head_list.{task_name}" in name:
                    assert p.grad is not None, f"Active head parameter {name} has no gradient"
                else:
                    assert p.grad is None or (p.grad == 0).all(), f"Inactive head parameter {name} has gradients"
            else:
                assert p.grad is not None, f"Backbone parameter {name} has no gradient"

        print(f"  Training step verification on '{task_name}' ✓")

        # Create dummy dataset/dataloader for Hessian computation
        # compute_Q_for_task expects (inputs_tensor, targets_tensor)
        hessian_batch = (
            torch.randint(0, 1000, (batch_size, seq_len), device=device),
            torch.randint(0, num_classes, (batch_size,), device=device),
        )
        hessian_loader = [hessian_batch]

        # Compute Q and Lambda for the current task
        print(f"  Computing Hessian eigenspace for '{task_name}'...")
        Q_new, Lambda_new = compute_Q_for_task(
            model=model,
            k=k,
            device=device,
            train_loader=hessian_loader
        )
        
        # Verify shapes of computed eigenspace
        param_dim = sum(p.numel() for p in model.backbone.parameters())
        assert Q_new.shape[0] == param_dim
        assert Q_new.shape[1] <= k
        assert Lambda_new.shape[0] == Q_new.shape[1]
        print(f"  Computed Q shape: {list(Q_new.shape)}, Lambda shape: {list(Lambda_new.shape)} ✓")

        # Accumulate/merge task eigenspace into memory
        print(f"  Accumulating eigenspace...")
        Q_memory, Lambda_memory = accumulate_hessian_eigenspace_stable(
            Q_old=Q_memory,
            Lambda_old=Lambda_memory,
            Q_new=Q_new,
            Lambda_new=Lambda_new,
            t=task_idx+1,
            k=k,
        )
        
        # Verify accumulated memory shapes and orthogonality
        assert Q_memory.shape[0] == param_dim
        assert Q_memory.shape[1] <= k
        assert Lambda_memory.shape[0] == Q_memory.shape[1]
        
        # Check orthogonality: Q_memory^T Q_memory should be identity
        # We compute in double precision to avoid float32 round-off accumulation over 124 million parameters
        orth_err = (Q_memory.cpu().double().T @ Q_memory.cpu().double() - torch.eye(Q_memory.shape[1], device=torch.device("cpu"), dtype=torch.double)).abs().max().item()
        assert orth_err < 1e-3, f"Orthogonality error too high: {orth_err}"
        print(f"  Accumulated Q shape: {list(Q_memory.shape)}, Orthogonality error: {orth_err:.2e} ✓")

        # Set the projection matrix on the optimizer
        optimizer.set_Q(Q_memory, Lambda_memory)
        print(f"  Optimizer projection matrix updated successfully ✓")

    print("\nSequential task smoke test passed successfully!")


if __name__ == "__main__":
    test_classification_training_smoke()
    test_sequential_tasks_smoke()
    test_cnn_dailymail_training_smoke()
    test_squad_v2_training_smoke()

