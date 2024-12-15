import importlib
from pathlib import Path
from typing import Any, Dict

import torch.nn as nn


def create_model(config: Dict[str, Any]) -> nn.Module:
    module_path, class_name = config["path"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
        model = model_class(**config.get("params", {}))
        return model
    except Exception as e:
        raise ImportError(f"Failed to create model {config['path']}: {str(e)}")


def create_checkpoints_dir(experiment_name):
    checkpoints_dir = Path("checkpoints") / experiment_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    return checkpoints_dir
