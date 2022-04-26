import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import mlflow.pytorch
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from models import MNISTModel, MNISTModelConv


def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))


# Initialize our model
if __name__ == "__main__":
    assert "MLFLOW_TRACKING_URI" in os.environ
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "--model", metavar="N", type=str, help="either mlp or cnn", default="mlp"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    model_name = parser.parse_args().model
    batch_size = parser.parse_args().batch_size
    epochs = parser.parse_args().epochs
    lr = parser.parse_args().lr

    hyper_params = {
        "batch_size": batch_size,
        "learning_rate": lr,
        "hidden_dim": [16, 32] if model_name == "cnn" else 128,
    }
    if model_name == "mlp":
        mnist_model = MNISTModel(**hyper_params)
    elif model_name == "cnn":
        mnist_model = MNISTModelConv(**hyper_params)

    # Initialize DataLoader from MNIST Dataset
    train_ds = MNIST(
        os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
    )
    test_ds = MNIST(
        os.getcwd(), train=False, download=True, transform=transforms.ToTensor()
    )

    train_loader = DataLoader(
        train_ds, batch_size=hyper_params["batch_size"], num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=hyper_params["batch_size"], num_workers=0
    )
    imgs_test, labels_test = next(iter(test_loader))

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=epochs)
    
    # set experiment name
    mlflow.set_experiment("MNIST")
    # Train the model
    mlflow.pytorch.autolog()

    with mlflow.start_run() as run:
        mlflow.log_params(mnist_model.hparams)
        trainer.fit(mnist_model, train_loader, test_loader)
        print("Logging results")

        # logging some random image
        plt.figure()
        plt.plot(np.random.randn(150), "r")
        plt.savefig("test.jpg")
        mlflow.log_artifact("test.jpg")

        for a, b in test_loader:
            break
        signature = infer_signature(
            model_input=imgs_test.detach().numpy(),
            model_output=mnist_model(imgs_test).detach().numpy(),
        )
        input_example = imgs_test.detach().numpy()
        print(signature)
        mlflow.pytorch.log_model(
            mnist_model,
            "model",
            registered_model_name=model_name,
            conda_env=mlflow.pytorch.get_default_conda_env(),
            signature=signature,
            input_example=input_example,
        )
        # scripted_pytorch_model = torch.jit.script(mnist_model)
        # mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

    # fetch the auto logged parameters and metrics
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
