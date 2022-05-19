import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression

import mlflow

# enable autologging
mlflow.set_experiment("sklearn-experiment")
mlflow.sklearn.autolog()

# preparing data only use one feature
X, y = datasets.load_diabetes(return_X_y=True)
X = X[:, np.newaxis, 2]


# train a model
model = LinearRegression()
with mlflow.start_run() as run:
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.scatter(X, y, color="black")
    plt.plot(X, y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.savefig("results.jpg")
    mlflow.log_artifact("results.jpg")
