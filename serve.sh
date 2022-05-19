export MLFLOW_TRACKING_URI=http://0.0.0.0:5000
mlflow models serve --model-uri models:/cnn/1 --no-conda --port 5001