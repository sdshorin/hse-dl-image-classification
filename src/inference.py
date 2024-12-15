import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml
from box import Box
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import ImageDataset
from models.model_factory import create_model
from utils.device import get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="name of the experiment to use (e.g. cnn_v1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_config = f"src/experiments/{args.experiment}.yaml"
    checkpoint_path = f"checkpoints/{args.experiment}/best_model.pth"

    output_file = f"submission_{args.experiment}.csv"

    if not Path(experiment_config).exists():
        raise FileNotFoundError(f"Config file not found: {experiment_config}")
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(experiment_config) as f:
        config = Box(yaml.safe_load(f))

    device = get_device("auto")
    print(f"Using device: {device}")

    test_dataset = ImageDataset(img_dir="dataset/test", transform=None, is_test=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = create_model(config.model).to(device)

    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(
        f"Best validation accuracy: {checkpoint.get('best_valid_acc', 'unknown'):.2f}%"
    )

    model.eval()

    predictions = []
    image_ids = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            predictions.extend(preds)

            batch_size = len(images)
            batch_ids = (
                test_dataset.df["Id"]
                .iloc[len(predictions) - batch_size : len(predictions)]
                .tolist()
            )
            image_ids.extend(batch_ids)

    submission_df = pd.DataFrame({"Id": image_ids, "Category": predictions})

    submission_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")
    print(f"Number of predictions: {len(predictions)}")


if __name__ == "__main__":
    main()
