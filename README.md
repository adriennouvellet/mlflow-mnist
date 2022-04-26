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

or with git

```bash
mlflow run git@github.com:adriennouvellet/mlflow-mnist.git -P lr=0.001 -P batch_size=128 -P epochs=2 --no-conda --experiment-name MNIST
mlflow run git@github.com:adriennouvellet/mlflow-mnist.git -P lr=0.01 -P batch_size=128 -P epochs=2 --no-conda --experiment-name MNIST -P model=cnn 
```

## See results

```bash
open http://localhost:5000
```