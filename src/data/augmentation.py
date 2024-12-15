from typing import Callable, Dict

import albumentations as A
from albumentations.pytorch import ToTensorV2

AUGMENTATION_REGISTRY: Dict[str, Callable] = {}


def register_augmentation(name: str):
    def decorator(func: Callable):
        AUGMENTATION_REGISTRY[name] = func
        return func

    return decorator


def get_augmentation(name: str, train: bool = True):
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY.keys())}"
        )
    return AUGMENTATION_REGISTRY[name](train)


@register_augmentation("aggressive")
def get_transforms(train: bool = True):
    if train:
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.25, scale_limit=0.25, rotate_limit=60, p=0.7
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


@register_augmentation("default")
def get_transforms(train: bool = True):
    if train:
        return A.Compose(
            [
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.10, scale_limit=0.10, rotate_limit=30, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(
                            brightness_limit=0.2, contrast_limit=0.2, p=1.0
                        ),
                        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(var_limit=(10.0, 50.0)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                    ],
                    p=0.3,
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


@register_augmentation("minimal")
def get_transforms_minimal(train: bool = True):
    if train:
        return A.Compose(
            [
                A.RandomRotate90(p=0.3),
                A.Flip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])


@register_augmentation("minimal_improved")
def get_transforms_minimal(train: bool = True):
    if train:
        return A.Compose(
            [
                A.RandomRotate90(p=0.3),
                A.Flip(p=0.3),
                A.ShiftScaleRotate(
                    shift_limit=0.17, scale_limit=0.18, rotate_limit=40, p=0.5
                ),
                A.OneOf(
                    [
                        A.ColorJitter(
                            brightness=0.12,
                            contrast=0.12,
                            saturation=0.12,
                            hue=0.1,
                            p=1.0,
                        ),
                        A.HueSaturationValue(
                            hue_shift_limit=15,
                            sat_shift_limit=20,
                            val_shift_limit=10,
                            p=1.0,
                        ),
                        A.RGBShift(
                            r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0
                        ),
                    ],
                    p=0.8,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.1, p=0.3
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose([ToTensorV2()])
