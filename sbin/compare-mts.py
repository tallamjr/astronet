import argparse
import json
import sys

import pandas as pd
import tensorflow as tf

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.utils import load_dataset


def update_results(architecture: str, mode: str = "precision"):

    datasets = [
        "ArabicDigits",
        "AUSLAN",
        "CharacterTrajectories",
        "CMUsubject16",
        "ECG",
        "JapaneseVowels",
        "KickvsPunch",
        "Libras",
        "NetFlow",
        "UWave",
        "Wafer",
        "WalkvsRun",
    ]

    if mode in ["precision", "recall"]:
        results_key = f"model_predict_{mode}_score"
    else:
        results_key = "model_evaluate_on_test_acc"

    table = {}
    for dataset in datasets:
        # print(f"{dataset}")

        X_train, y_train, X_test, y_test, loss = load_dataset(dataset)

        with open(
            f"{asnwd}/astronet/{architecture}/models/{dataset}/results.json"
        ) as f:
            events = json.load(f)

        # Get params for best model with lowest precision score
        event = max(events["training_result"], key=lambda ev: ev[results_key])

        model_name = event["name"]

        table[f"{dataset}"] = event[results_key]

    df = pd.DataFrame.from_dict(table, orient="index")
    df.columns = [f"{architecture}"]
    print(df)

    return df


def update_accuracy_results():
    pass


def update_recall_results():
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate table of results for precision, recall and accuracy with regards to the MTS benchmark datasets"
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="precision",
        help="Choose which metric: {precision, recall, accuracy}",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        default="atx",
        help="Choose which architecture to compare with",
    )

    parser.add_argument(
        "-s",
        "--save",
        help="Whether to save results to disk or not",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
        print(argsdict)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    mode = args.mode
    architecture = args.architecture

    df = update_results(architecture, mode)

    if args.save is not None:
        filename = f"{asnwd}/results/mts-{architecture}-results-{mode}.csv"
        df.to_csv(filename)
