import joblib
import json
import logging
import optuna
import subprocess
import sys

from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.NOTSET,
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers= [
            logging.FileHandler(filename='studies.log', mode='a'),
            logging.StreamHandler(sys.stdout)
        ]
)

optuna.logging.enable_propagation()  # Propagate logs to the root logger.
optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

import time
unixtimestamp = int(time.time())
label = subprocess.check_output(["git", "describe", "--always"]).strip().decode()

study = optuna.create_study(study_name=f"{unixtimestamp}-{label}")
# study = optuna.create_study()

logger.info("Start optimization.")

study.optimize(objective, n_trials=10)

logger.info(study.best_params)

# TODO: Get hash value of code and timestamp?

best_params = {}

best_params['name'] = str(unixtimestamp) + "-" + label

for key, value in study.best_params.items():
    logger.info("   {}: {}".format(key, value))
    best_params[f"{key}"] = value

with open(f"{Path().absolute()}/runs/result.json") as jf:
    data = json.load(jf)

    previous_results = data['optuna_result']
    # appending data to optuna_result
    previous_results.append(best_params)

with open(f"{Path().absolute()}/runs/result.json", "w") as rf:
    json.dump(data, rf, sort_keys=True, indent=4)

with open(f"{Path().absolute()}/runs/study-{unixtimestamp}-{label}.pkl", "wb") as sf:
    joblib.dump(study, sf)
