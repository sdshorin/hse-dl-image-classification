import io
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.augmentation import get_augmentation


class ImageDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
        df: Optional[pd.DataFrame] = None,
        transform=None,
        is_test: bool = False,
        cache_images: bool = True,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.cache_images = cache_images
        self.image_cache = {}

        if df is not None:
            self.df = df
        else:
            self.df = pd.DataFrame(
                {"Id": [f for f in os.listdir(img_dir) if f.endswith(".jpg")]}
            )

        if self.cache_images:
            print("Caching images...")
            for idx in range(len(self.df)):
                img_name = self.df.iloc[idx]["Id"]
                img_path = os.path.join(self.img_dir, img_name)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.astype(np.float32) / 255.0
                self.image_cache[img_name] = image

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        img_name = self.df.iloc[idx]["Id"]

        if self.cache_images:
            image = self.image_cache[img_name]
        else:
            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.is_test:
            return image
        else:
            label = self.df.iloc[idx]["Category"]
            return image, torch.tensor(label, dtype=torch.long)


def get_dataloaders(data_config):
    from data.augmentation import get_transforms

    df = pd.read_csv(data_config.train_csv)

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(df, df["Category"]))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    augmentation_name = data_config.get("augmentation", "default")

    train_dataset = ImageDataset(
        data_config.train_dir,
        train_df,
        transform=get_augmentation(augmentation_name, train=True),
        cache_images=True,
    )

    val_dataset = ImageDataset(
        data_config.train_dir,
        val_df,
        transform=get_augmentation(augmentation_name, train=False),
        cache_images=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    return train_loader, val_loader
