import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

import wandb


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        config: Dict,
        checkpoint_dir: str,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.save_dir = Path(checkpoint_dir)

        self.scaler = GradScaler()

        self.best_valid_acc = 0.0

        self.train_batch_outputs = []
        self.valid_batch_outputs = []

    def train_epoch(self, train_loader):
        self.model.train()

        self.train_batch_outputs = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (data, target) in enumerate(pbar):
            start_time = time.time()

            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                pred = output.argmax(dim=1, keepdim=True)
                correct_batch = pred.eq(target.view_as(pred)).sum().item()

                self.train_batch_outputs.append(
                    {
                        "loss": loss.item(),
                        "correct": correct_batch,
                        "total": target.size(0),
                        "time": time.time() - start_time,
                    }
                )

            if batch_idx % 10 == 0:
                avg_loss = np.mean([b["loss"] for b in self.train_batch_outputs[-100:]])
                avg_acc = (
                    100.0
                    * sum(b["correct"] for b in self.train_batch_outputs[-100:])
                    / sum(b["total"] for b in self.train_batch_outputs[-100:])
                )
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

        if self.scheduler is not None:
            self.scheduler.step()
        metrics = {
            "train_loss": np.mean([b["loss"] for b in self.train_batch_outputs]),
            "train_acc": 100.0
            * sum(b["correct"] for b in self.train_batch_outputs)
            / sum(b["total"] for b in self.train_batch_outputs),
            "train_time": sum(b["time"] for b in self.train_batch_outputs),
        }
        return metrics

    @torch.no_grad()
    def validate(self, valid_loader):
        self.model.eval()
        self.valid_batch_outputs = []

        pbar = tqdm(valid_loader, desc="Validation", leave=False)
        for data, target in pbar:
            start_time = time.time()

            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            with autocast():
                output = self.model(data)
                loss = self.criterion(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct_batch = pred.eq(target.view_as(pred)).sum().item()

            self.valid_batch_outputs.append(
                {
                    "loss": loss.item(),
                    "correct": correct_batch,
                    "total": target.size(0),
                    "time": time.time() - start_time,
                }
            )

            if len(self.valid_batch_outputs) % 10 == 0:
                avg_loss = np.mean([b["loss"] for b in self.valid_batch_outputs[-100:]])
                avg_acc = (
                    100.0
                    * sum(b["correct"] for b in self.valid_batch_outputs[-100:])
                    / sum(b["total"] for b in self.valid_batch_outputs[-100:])
                )
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.2f}%"})

        metrics = {
            "valid_loss": np.mean([b["loss"] for b in self.valid_batch_outputs]),
            "valid_acc": 100.0
            * sum(b["correct"] for b in self.valid_batch_outputs)
            / sum(b["total"] for b in self.valid_batch_outputs),
            "valid_time": sum(b["time"] for b in self.valid_batch_outputs),
        }
        return metrics

    def train(self, train_loader, valid_loader, num_epochs):
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            train_metrics = self.train_epoch(train_loader)

            valid_metrics = self.validate(valid_loader)

            epoch_time = time.time() - epoch_start_time
            metrics = {**train_metrics, **valid_metrics, "epoch_time": epoch_time}

            wandb.log(metrics)

            print(
                f"Train Loss: {metrics['train_loss']:.4f} | Train Acc: {metrics['train_acc']:.2f}%"
            )
            print(
                f"Valid Loss: {metrics['valid_loss']:.4f} | Valid Acc: {metrics['valid_acc']:.2f}%"
            )
            print(
                f"Epoch Time: {epoch_time:.2f}s (Train: {metrics['train_time']:.2f}s, Valid: {metrics['valid_time']:.2f}s)"
            )

            if valid_metrics["valid_acc"] > self.best_valid_acc:
                self.best_valid_acc = valid_metrics["valid_acc"]
                self.save_model("best_model.pth")

            if (epoch + 1) % 20 == 0:
                self.save_model(f"model_epoch_{epoch+1}.pth")

    def save_model(self, filename):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler else None
                ),
                "best_valid_acc": self.best_valid_acc,
                "config": self.config,
                "scaler": self.scaler.state_dict(),
            },
            self.save_dir / filename,
        )
