import os
from pathlib import Path

import mlflow

STATE_DIR = Path(os.getenv("STATE_DIR", ".")).resolve()
MLFLOW_URI = f"sqlite:///{(STATE_DIR / 'mlflow.db').as_posix()}"


def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name,
            artifact_location=(STATE_DIR / "mlruns" / experiment_name).as_uri(),
        )
    mlflow.set_experiment(experiment_name)