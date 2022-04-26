# Launch

```bash
mlflow server \
    --backend-store-uri sqlite:///mlruns.db \
    --default-artifact-root gs://datascience-general-prod/test/artifacts \
    --host 0.0.0.0
```