# Mlflow Tracking: Garbage Classification

In this example, we train a Pytorch model to predict or classifcy garbage/trash.

### Running the code
To run the example via MLflow, navigate to the `garbage-mlflow-tracking` directory and run the command

```
mlflow run .
```

This will run `train.py` with the default set of parameters such as  `--max_epochs=5`. You can see the default value in the `MLproject` file.

If you have the required modules for the file and would like to skip the creation of a conda environment, add the argument `--env-manager=local`.

```
mlflow run . --env-manager=local
```

### Viewing results in the MLflow UI

Once the code is finished executing, you can view the run's metrics, parameters, and details by running the command

```
mlflow ui
```

and navigating to [http://localhost:5000](http://localhost:5000).

For more details on MLflow tracking, see [the docs](https://www.mlflow.org/docs/latest/tracking.html#mlflow-tracking).

## Logging to a custom tracking server
To configure MLflow to log to a custom (non-default) tracking location, set the MLFLOW_TRACKING_URI environment variable, e.g. via export MLFLOW_TRACKING_URI=http://localhost:5000/. For more details, see [the docs](https://mlflow.org/docs/latest/tracking.html#where-runs-are-recorded).
