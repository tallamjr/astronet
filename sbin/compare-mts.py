import argparse
import json
import logging
import sys

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

from typing import Dict, List, Union

import pandas as pd
import tensorflow as tf

from astronet.constants import ASTRONET_WORKING_DIRECTORY as asnwd
from astronet.utils import load_dataset


def update_results(
    architectures: List[str], mode: str = "precision", save: bool = False
):

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
    for architecture in architectures:
        for dataset in datasets:

            X_train, y_train, X_test, y_test, loss = load_dataset(dataset)

            with open(
                f"{asnwd}/astronet/{architecture}/models/{dataset}/results.json"
            ) as f:
                events = json.load(f)

            # Get params for best model with highest scores
            event = max(events["training_result"], key=lambda ev: ev[results_key])

            model_name = event["name"]

            table[f"{dataset}"] = event[results_key]

        df = pd.DataFrame.from_dict(table, orient="index")
        df.columns = [f"{architecture}"]
        print(df)
        if save is not None:
            filename = f"{asnwd}/results/mts-{architecture}-results-{mode}.csv"
            df.to_csv(filename)

    return


def make_comparison_table(architectures: List[str], mode: str = "precision"):

    pd.set_option("display.precision", 1)
    pd.set_option("display.float_format", "{:.2f}".format)

    tables = []
    for architecture in architectures:
        df = pd.read_csv(
            f"{asnwd}/results/mts-{architecture}-results-{mode}.csv",
            index_col="Unnamed: 0",
        )
        tables.append(df[f"{architecture}"].multiply(100).to_frame())

    atx = tables[0]
    t2 = tables[1]

    df_combined_arch = t2.join(atx)

    # TODO: Change to respective metric file
    df_benchmark = pd.read_csv(
        f"{asnwd}/results/mts-fawaz-results.csv", index_col="Unnamed: 0"
    )

    df_combined_both = df_combined_arch.join(df_benchmark)

    filename = f"{asnwd}/results/mts-combined-results-{mode}.md"
    results = df_combined_both.to_markdown()
    print(results, file=open(filename, "w"))

    print(results)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate table of results for precision, recall and accuracy with regards to the MTS benchmark datasets"
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="precision",
        help="Choose which metric: {'precision', 'recall', 'accuracy'}",
    )

    parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        help="Choose which architecture to compare with: {'atx', 't2'}",
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

    architectures = ["atx", "t2"]
    if args.architecture is not None:
        architecture = args.architecture
        architectures = [arch for arch in architectures if arch == architecture]
        update_results(architectures, mode)
        logging.info(
            "Only updating tables. For full comparson, both architectures required"
        )
    else:
        update_results(architectures, mode)
        results = make_comparison_table(architectures, mode)
