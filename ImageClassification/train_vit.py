"""
train_vit.py — Share ImageClassification
=========================================
Train a ViT with LoRA or EigenFlux on image-classification class subsets in a
continual-learning fashion.

Supported methods
-----------------
* lora       – Standard LoRA (bootstrap phase; saves .pth checkpoints).
* eigenflux  – EigenFlux adapter (continual phase; loads from / saves to PEFT
               adapter directories).
* none       – Full fine-tuning (baseline).

Key flags for the eigenflux method
------------------------------------
--eigenflux_load_path   Directory of a previously saved EigenFlux PEFT adapter
                        (loaded as warm initialisation before training).
--eigenflux_save_path   Directory where the trained EigenFlux adapter will be
                        saved after training.
--subset_index          1-based index of the class subset to train on.
                        If omitted, all subsets are trained sequentially
                        (useful for the lora bootstrap phase).
--sampled_subsets_path  Path to a previously saved sampled_subsets.txt so that
                        all methods see the same class splits.

Example (lora bootstrap – trains every subset)
-----------------------------------------------
python train_vit.py --method lora --dataset CIFAR100 --subset_size 10 \
    --save_path ./checkpoints

Example (eigenflux – train subset 2 loading from subset 1's adapter)
----------------------------------------------------------------------
python train_vit.py --method eigenflux --dataset CIFAR100 --subset_size 10 \
    --subset_index 2 \
    --sampled_subsets_path ./checkpoints/lora/CIFAR100/sampled_subsets.txt \
    --eigenflux_load_path ./adapters/subset_1_trained \
    --eigenflux_save_path ./adapters/subset_2_trained \
    --eigenflux_r 8 --num_components 32
"""

import os
import random
import argparse
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from transformers import ViTModel, AutoImageProcessor
from peft import LoraConfig, get_peft_model, EigenFluxConfig
from torchvision.datasets import CIFAR100, ImageFolder
import torchvision.transforms as transforms

from utils import (
    dataloader_from_subset,
)

# ── reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================================
# Model
# ============================================================================


class ViTClassifier(nn.Module):
    """
    ViT wrapped with LoRA or EigenFlux, plus a per-subset linear head.

    The PEFT adapter (LoRA or EigenFlux) is applied to the ViT backbone.
    The classification head (``self.classifier``) is always reset to random
    weights when a new subset is started so it does not carry class indices
    from the previous task.
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        num_classes: int = 10,
        method: str = "lora",
        lora_rank: int = 8,
        eigenflux_r: int = 8,
        num_components: int = 32,
        use_rank_updates: bool = False,
        num_rank_updates: int = 0,
        eigenflux_load_path: str = None,
        adapter_name: str = "default",
        device: str = "cuda:0",
    ):
        super().__init__()
        self.method = method
        self.adapter_name = adapter_name
        self.device = device if torch.cuda.is_available() else "cpu"

        base_vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_classes)

        if method == "lora":
            config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 16,
                target_modules=["query", "value"],
            )
            self.vit = get_peft_model(base_vit, config)

        elif method == "eigenflux":
            config = EigenFluxConfig(
                r=eigenflux_r,
                num_components=num_components,
                use_rank_updates=use_rank_updates,
                num_rank_updates=num_rank_updates,
                target_modules=["query", "value"],
            )
            self.vit = get_peft_model(base_vit, config, adapter_name)
            if eigenflux_load_path is not None:
                print(f"Loading EigenFlux adapter from: {eigenflux_load_path}")
                self.vit.load_adapter(
                    eigenflux_load_path,
                    adapter_name,
                    is_trainable=True,
                )
                self.vit.set_adapter(adapter_name)

        elif method == "none":
            self.vit = base_vit
        else:
            raise ValueError(f"Unknown method: {method}")

        self.vit.to(self.device)
        self.classifier.to(self.device)

        print(f"Model on {self.device}")
        self._print_trainable_parameters()

    def _print_trainable_parameters(self):
        trainable, total = 0, 0
        for p in self.vit.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        for p in self.classifier.parameters():
            total += p.numel()
            trainable += p.numel()
        print(
            f"Trainable: {trainable:,} / {total:,} " f"({100 * trainable / total:.2f}%)"
        )

    def reset_classifier(self, num_classes: int):
        """Replace the classifier head for a new subset (different #classes)."""
        self.classifier = nn.Linear(768, num_classes).to(self.device)

    def forward(self, x):
        features = self.vit(x).pooler_output  # (batch, 768)
        return self.classifier(features)

    # ── training loop ────────────────────────────────────────────────────────

    def train_model(
        self,
        train_loader,
        test_loader,
        epochs: int = 40,
        lr: float = 5e-4,
        weight_decay: float = 1e-6,
        use_rank_updates: bool = False,
        use_wandb: bool = False,
    ):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        # EigenFlux / EigenLoRA benefits from plateau scheduling;
        # LoRA does fine with a linear decay.
        if self.method in ("eigenflux",):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, total_iters=epochs
            )

        criterion = nn.CrossEntropyLoss()
        best_test_acc = -np.inf

        # Evaluate before any training to log baseline.
        test_loss, test_acc = self.evaluate(test_loader)
        if use_wandb:
            wandb.log({"test_loss": test_loss, "test_accuracy": test_acc, "epoch": 0})

        for epoch in range(epochs):
            self.train()
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_losses, train_preds, train_labels = [], [], []

            for batch in tqdm(train_loader, desc="Training"):
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                optimizer.zero_grad()
                logits = self(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_preds.append(logits.argmax(dim=-1).cpu())
                train_labels.append(batch[1])

            avg_loss = float(np.mean(train_losses))
            if self.method == "eigenflux":
                old_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(avg_loss)
                new_lr = optimizer.param_groups[0]["lr"]
                if old_lr != new_lr:
                    print(f"LR: {old_lr:.2e} → {new_lr:.2e}")
            else:
                scheduler.step()

            all_preds = torch.cat(train_preds)
            all_labels = torch.cat(train_labels)
            train_acc = (all_preds == all_labels).float().mean().item() * 100
            print(f"Train Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}%")

            if use_wandb:
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_accuracy": train_acc,
                        "epoch": epoch + 1,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            # Recalculate EigenFlux basis at the end of training when rank
            # updates are active (mirrors NLU use_rank_updates behaviour).
            if self.method == "eigenflux" and use_rank_updates and epoch == epochs - 1:
                print("  Recalculating EigenFlux basis (rank-update pass) …")
                for module in self.vit.modules():
                    if hasattr(module, "recalculate"):
                        module.recalculate(self.adapter_name)

            test_loss, test_acc = self.evaluate(test_loader)
            if use_wandb:
                wandb.log(
                    {
                        "test_loss": test_loss,
                        "test_accuracy": test_acc,
                        "epoch": epoch + 1,
                    }
                )

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                print(f"★ New best: {best_test_acc:.2f}%")
                if use_wandb:
                    wandb.log(
                        {"best_test_accuracy": best_test_acc, "best_epoch": epoch + 1}
                    )

    @torch.no_grad()
    def evaluate(self, test_loader):
        self.eval()
        criterion = nn.CrossEntropyLoss()
        losses, preds, labels = [], [], []

        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            images = batch[0].to(self.device)
            batch_labels = batch[1].to(self.device)
            logits = self(images)
            losses.append(criterion(logits, batch_labels).item())
            preds.append(logits.argmax(dim=-1).cpu())
            labels.append(batch[1])

        avg_loss = float(np.mean(losses))
        accuracy = (torch.cat(preds) == torch.cat(labels)).float().mean().item() * 100
        print(f"Test Loss: {avg_loss:.4f} | Test Acc: {accuracy:.2f}%")
        return avg_loss, accuracy

    # ── checkpoint helpers ───────────────────────────────────────────────────

    def save_lora_checkpoint(self, path: str):
        """Save full ViTClassifier state dict (.pth) for LoRA checkpoints."""
        torch.save(self.state_dict(), path)
        print(f"LoRA checkpoint saved to {path}")

    def save_eigenflux_adapter(self, adapter_dir: str):
        """Save EigenFlux PEFT adapter + classifier head."""
        os.makedirs(adapter_dir, exist_ok=True)
        self.vit.save_pretrained(adapter_dir)
        torch.save(
            self.classifier.state_dict(),
            os.path.join(adapter_dir, "classifier.pth"),
        )
        print(f"EigenFlux adapter saved to {adapter_dir}")

    def load_classifier(self, adapter_dir: str):
        """Load the classifier head saved alongside an EigenFlux adapter."""
        path = os.path.join(adapter_dir, "classifier.pth")
        if os.path.exists(path):
            self.classifier.load_state_dict(torch.load(path, map_location=self.device))
            print(f"Classifier head loaded from {path}")


# ============================================================================
# Dataset helpers  (identical to EigenLoRA's train_vit.py)
# ============================================================================


def get_image_transforms(processor, train: bool = True):
    img_size = processor.size["height"]
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=processor.image_mean, std=processor.image_std
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean, std=processor.image_std),
        ]
    )


# ImageNet-R / ImageNet-A / DomainNet use fixed 224×224 ImageNet normalisation
# regardless of the HuggingFace processor (which may differ for other models).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_transforms(train: bool = True):
    """Standard 224-px ImageNet-style transforms for ImageFolder datasets."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    224, scale=(0.05, 1.0), ratio=(3 / 4, 4 / 3)
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]
    )


# Datasets that use ImageNet-style 224-px transforms (not the ViT processor).
_IMAGENET_STYLE_DATASETS = {"imagenet_r", "imagenet_a", "domainnet"}


def load_dataset(dataset_name: str, data_root: str, processor):
    if dataset_name in _IMAGENET_STYLE_DATASETS:
        train_tf = get_imagenet_transforms(train=True)
        test_tf = get_imagenet_transforms(train=False)
    else:
        train_tf = get_image_transforms(processor, train=True)
        test_tf = get_image_transforms(processor, train=False)

    dataset_loaders = {
        "CIFAR100": lambda: (
            CIFAR100(root=data_root, train=True, transform=train_tf, download=True),
            CIFAR100(root=data_root, train=False, transform=test_tf, download=True),
        ),
        # ── ImageNet-R (200 classes, ImageFolder layout: train/ test/) ────────
        "imagenet_r": lambda: (
            ImageFolder(
                root=os.path.join(data_root, "imagenet-r", "train"), transform=train_tf
            ),
            ImageFolder(
                root=os.path.join(data_root, "imagenet-r", "test"), transform=test_tf
            ),
        ),
        # ── ImageNet-A (200 classes, ImageFolder layout: train/ test/) ────────
        "imagenet_a": lambda: (
            ImageFolder(
                root=os.path.join(data_root, "imagenet-a", "train"), transform=train_tf
            ),
            ImageFolder(
                root=os.path.join(data_root, "imagenet-a", "test"), transform=test_tf
            ),
        ),
        # ── DomainNet (345 classes, ImageFolder layout: train/ test/) ─────────
        # Expected on disk: <data_root>/domainnet/train/<class>/ and test/<class>/
        "domainnet": lambda: (
            ImageFolder(
                root=os.path.join(data_root, "domainnet", "train"), transform=train_tf
            ),
            ImageFolder(
                root=os.path.join(data_root, "domainnet", "test"), transform=test_tf
            ),
        ),
    }

    if dataset_name not in dataset_loaders:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. " f"Supported: {list(dataset_loaders)}"
        )
    return dataset_loaders[dataset_name]()


def create_class_subsets(dataset, subset_size: int, dataset_name: str = None):
    # Hard-code class counts for datasets where the test split may have fewer
    # folders than the training split (e.g. some ImageNet-A distributions).
    _KNOWN_CLASS_COUNTS = {
        "imagenet_r": 200,
        "imagenet_a": 200,
        "domainnet": 345,
    }
    if dataset_name in _KNOWN_CLASS_COUNTS:
        num_classes = _KNOWN_CLASS_COUNTS[dataset_name]
    elif hasattr(dataset, "classes"):
        num_classes = len(dataset.classes)
    else:
        raise ValueError("Cannot determine number of classes")

    if num_classes % subset_size != 0:
        raise ValueError(
            f"subset_size ({subset_size}) must evenly divide "
            f"num_classes ({num_classes})"
        )

    all_classes = list(range(num_classes))
    random.shuffle(all_classes)
    return [
        all_classes[i : i + subset_size] for i in range(0, num_classes, subset_size)
    ]


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train ViT with LoRA or EigenFlux (Share continual learning)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── method ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["lora", "eigenflux", "none"],
        help="PEFT method ('lora' for bootstrap, 'eigenflux' for continual tasks)",
    )

    # ── model ────────────────────────────────────────────────────────────────
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument(
        "--r",
        "--rank",
        dest="rank",
        type=int,
        default=8,
        help="LoRA rank (used when --method lora)",
    )

    # ── EigenFlux ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--eigenflux_r", type=int, default=8, help="LoRA rank inside EigenFlux adapter"
    )
    parser.add_argument(
        "--num_components", type=int, default=32, help="Number of EigenFlux components"
    )
    parser.add_argument(
        "--use_rank_updates",
        action="store_true",
        help="Enable rank-update vectors inside EigenFlux",
    )
    parser.add_argument(
        "--num_rank_updates", type=int, default=2, help="Number of rank-update vectors"
    )
    parser.add_argument(
        "--eigenflux_load_path",
        type=str,
        default=None,
        help="PEFT adapter directory to load as warm initialisation (eigenflux method)",
    )
    parser.add_argument(
        "--eigenflux_save_path",
        type=str,
        default=None,
        help="Where to save the trained EigenFlux adapter (eigenflux method)",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="default",
        help="PEFT adapter name",
    )

    # ── training ─────────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # ── dataset ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset",
        type=str,
        default="CIFAR100",
        choices=["CIFAR100", "imagenet_r", "imagenet_a", "domainnet"],
    )
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10,
        help="Number of classes per continual-learning subset",
    )
    parser.add_argument(
        "--subset_index",
        type=int,
        default=None,
        help="1-based index of the subset to train (None = all subsets, used for lora bootstrap)",
    )
    parser.add_argument(
        "--sampled_subsets_path",
        type=str,
        default=None,
        help="Path to sampled_subsets.txt (ensures same class splits across methods). "
        "If not given, fresh random subsets are created and saved.",
    )

    # ── output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--save_path",
        type=str,
        default="./checkpoints",
        help="Root directory for LoRA .pth checkpoints",
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="ViT_Share")
    parser.add_argument("--wandb_entity", type=str, default=None)

    return parser.parse_args()


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    print(f"\nConfiguration\n{'-' * 40}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print(f"{'-' * 40}\n")

    # ── Basic validation ──────────────────────────────────────────────────────
    if args.method == "eigenflux":
        # eigenflux_load_path may legitimately be None for the very first subset;
        # the run_share_continual.sh passes the eigenflux_init directory.
        # argument_check enforces this at the script level.
        pass
    if args.subset_size <= 0:
        raise ValueError("--subset_size must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.lr <= 0:
        raise ValueError("--lr must be positive")

    # ── Experiment directory (used for LoRA checkpoints / label mappings) ────
    experiment_dir = os.path.join(args.save_path, args.method, args.dataset)
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "label_mappings"), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "model_checkpoints"), exist_ok=True)

    # ── Image processor & dataset ────────────────────────────────────────────
    processor = AutoImageProcessor.from_pretrained(args.model_name, use_fast=True)
    train_set, test_set = load_dataset(args.dataset, args.data_root, processor)

    # ── Class subsets ────────────────────────────────────────────────────────
    if args.sampled_subsets_path and os.path.exists(args.sampled_subsets_path):
        with open(args.sampled_subsets_path, "r") as f:
            subsets = [eval(line.split(": ", 1)[1].strip()) for line in f.readlines()]
        print(f"Loaded {len(subsets)} class subsets from {args.sampled_subsets_path}")
    else:
        subsets = create_class_subsets(train_set, args.subset_size, args.dataset)
        subset_file = os.path.join(experiment_dir, "sampled_subsets.txt")
        with open(subset_file, "w") as f:
            for i, s in enumerate(subsets):
                f.write(f"Subset {i + 1}: {s}\n")
        print(f"Saved {len(subsets)} class subsets to {subset_file}")

    # ── Determine which subsets to train ─────────────────────────────────────
    if args.subset_index is not None:
        subset_range = [args.subset_index - 1]  # convert to 0-based
    else:
        subset_range = list(range(len(subsets)))

    # ── Training loop ─────────────────────────────────────────────────────────
    for subset_idx in subset_range:
        class_subset = subsets[subset_idx]
        print(f"\n{'=' * 55}")
        print(f"Subset {subset_idx + 1}/{len(subsets)}  |  Classes: {class_subset}")
        print(f"{'=' * 55}")

        # ── Build model ──────────────────────────────────────────────────────
        model = ViTClassifier(
            model_name=args.model_name,
            num_classes=args.subset_size,
            method=args.method,
            lora_rank=args.rank,
            eigenflux_r=args.eigenflux_r,
            num_components=args.num_components,
            use_rank_updates=args.use_rank_updates,
            num_rank_updates=args.num_rank_updates,
            eigenflux_load_path=args.eigenflux_load_path,
            adapter_name=args.adapter_name,
        )

        # ── W&B run ──────────────────────────────────────────────────────────
        if args.use_wandb:
            wandb.init(
                project=f"{args.wandb_project}_{args.dataset}",
                entity=args.wandb_entity,
                config=vars(args),
                name=f"{args.method}_subset_{subset_idx + 1}",
                reinit=True,
            )

        # ── Data loaders ─────────────────────────────────────────────────────
        train_loader, test_loader = dataloader_from_subset(
            train_set,
            test_set,
            class_subset,
            experiment_folder=experiment_dir,
            subset_index=subset_idx + 1,
            batch_size=args.batch_size,
        )

        # ── Train ────────────────────────────────────────────────────────────
        model.train_model(
            train_loader,
            test_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            use_rank_updates=args.use_rank_updates,
            use_wandb=args.use_wandb,
        )

        # ── Save ─────────────────────────────────────────────────────────────
        if args.method == "lora":
            ckpt_path = os.path.join(
                experiment_dir,
                "model_checkpoints",
                f"subset_{subset_idx + 1}_model.pth",
            )
            model.save_lora_checkpoint(ckpt_path)

        elif args.method == "eigenflux":
            # eigenflux_save_path is mandatory for eigenflux method
            if args.eigenflux_save_path is None:
                raise ValueError(
                    "--eigenflux_save_path must be set when --method eigenflux"
                )
            model.save_eigenflux_adapter(args.eigenflux_save_path)

        elif args.method == "none":
            ckpt_path = os.path.join(
                experiment_dir,
                "model_checkpoints",
                f"subset_{subset_idx + 1}_model.pth",
            )
            model.save_lora_checkpoint(ckpt_path)

        if args.use_wandb:
            wandb.finish()

    print(f"\n{'=' * 55}")
    print("Training complete.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
