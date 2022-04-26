# Description

```
├── data # some mnist data samples
├── README.md
├── models.py # mlp + cnn for mnist
├── test.py # test mlflow server
└── train.py # training script
```

# Usage

## Launch server:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root gs://datascience-general-prod/test/artifacts \
    --host 0.0.0.0
```

- backend-store-uri: db to store result (metric of learning)
- default-artifact-root: gcs path to save model and artifacts

## Train model

```bash
python train.py --model mlp --lr 0.01 --epochs 10 --batch_size 64
python train.py --model mlp --lr 0.001 --epochs 10 --batch_size 64
python train.py --model cnn --lr 0.001 --epochs 10 --batch_size 64
python train.py --model cnn --lr 0.0001 --epochs 10 --batch_size 128
```

## See results

```bash
open http://localhost:5000
```