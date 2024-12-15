# Image Classification Project for HSE DL Course

This is my solution for HSE Deep Learning course homework competition on Kaggle:
https://www.kaggle.com/competitions/hse-cds-dl-hw-2/overview

## Quick Start

### Local Training

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model
python src/train.py --experiment cnn_v2_minimal
```

### Remote Training 

1. Configure server settings in `scripts/server_config.sh`:
```bash
export REMOTE_USER="user"
export REMOTE_HOST="your_server"
export REMOTE_PORT="22"
export REMOTE_DIR="/home/$REMOTE_USER/image-classification"
export PROJECT_SSH_DIR="./.ssh"
export SSH_KEY_NAME="project_key"
```

2. Run setup scripts:
```bash
chmod +x scripts/*.sh
./scripts/setup_ssh.sh
./scripts/upload_project.sh
./scripts/setup_env.sh
./scripts/upload_dataset.sh
./scripts/train_model.sh cnn_v2_minimal
./scripts/check_training.sh training_X_Y
./scripts/download_models.sh cnn_v2

```


### Inference

```bash
python src/inference.py --experiment cnn_v1_minimal
```

## Dataset Structure

Put dataset files in `dataset` folder:
```
dataset/
├── trainval/
│   ├── trainval_000000.jpg
│   ├── trainval_000001.jpg
│   └── ...
├── test/
│   └── ...
└── trainval.csv
```

trainval.csv format:
```
Id,Category  
trainval_000000.jpg,91
trainval_000001.jpg,172
...
```

Images are 40x40 RGB, 200 classes total.

## Adding New Experiments

1. Create new model in `src/models/`
2. Add config in `src/experiments/your_experiment.yaml`:
```yaml
model:
  path: models.your_model_file.ModelClass
  params:
    in_channels: 3 
    num_classes: 200

training:
  epochs: 200
  optimizer:
    lr: 0.001
```

## Model Performance

| Model | Accuracy |
|-------|----------|
| baseline | 38.04% |
| cnn_v1 | 32.64% |
| cnn_v1_base | 34.64% |
| cnn_v2_minimal | 46.81% |
| **cnn_v1_minimal_improved** | 48.33% |
| cnn_v2 | 35.25% |
| cnn_v3_minimal |  47.77% |


## Results

Best model: **cnn_v1_minimal_improved** achieved 48.33% accuracy on the test set. The model weights are included in the repository.

To reproduce the results, run:
```bash
python src/inference.py --experiment cnn_v1_minimal_improved
```


