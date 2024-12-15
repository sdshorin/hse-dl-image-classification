import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from box import Box
from torch.optim.lr_scheduler import OneCycleLR

import wandb
from data.dataset import get_dataloaders
from models.model_factory import create_checkpoints_dir, create_model
from training.trainer import Trainer
from utils.device import get_device
from utils.helper import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="name of the experiment to use (e.g. cnn_v1)",
    )
    parser.add_argument(
        "--wandb_config",
        type=str,
        default="src/config/wandb.yaml",
        help="path to wandb config",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    experiment_config = f"src/experiments/{args.experiment}.yaml"
    with open(experiment_config) as f:
        config = Box(yaml.safe_load(f))

    experiment_name = args.experiment

    checkpoint_dir = create_checkpoints_dir(experiment_name)

    with open(args.wandb_config) as f:
        wandb_config = Box(yaml.safe_load(f))
    wandb.login(key=wandb_config.key)
    wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=experiment_name,
        config=dict(config),
    )

    device = get_device("auto")
    print(f"Using device: {device}")
    train_loader, valid_loader = get_dataloaders(config.data)
    model = create_model(config.model).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.training.optimizer.lr,
        weight_decay=config.training.optimizer.weight_decay,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.training.scheduler.max_lr,
        epochs=config.training.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=config.training.scheduler.pct_start,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config.training,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config.training.epochs,
    )

    wandb.finish()


if __name__ == "__main__":
    main()
