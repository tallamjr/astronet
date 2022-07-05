# Copyright 2020 - 2022
# Author: Tarek Allam Jr.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    architectures: List[str], metric: str = "precision", save: bool = False
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

    if metric in ["precision", "recall"]:
        results_key = f"model_predict_{metric}_score"
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
        if save:
            filename = f"{asnwd}/results/mts-{architecture}-results-{metric}.csv"
            df.to_csv(filename)

    return


def make_comparison_table(architectures: List[str], metric: str = "precision"):

    pd.set_option("display.precision", 1)
    pd.set_option("display.float_format", "{:.2f}".format)

    tables = []
    for architecture in architectures:
        df = pd.read_csv(
            f"{asnwd}/results/mts-{architecture}-results-{metric}.csv",
            index_col="Unnamed: 0",
        )
        tables.append(df[f"{architecture}"].multiply(100).to_frame())

    atx = tables[0]
    t2 = tables[1]

    df_combined_arch = t2.join(atx)

    df_benchmark = pd.read_csv(
        f"{asnwd}/results/mts-results-{metric}.csv", index_col="Unnamed: 0"
    )

    df_combined_both = df_combined_arch.join(df_benchmark)

    filename = f"{asnwd}/results/mts-combined-results-{metric}.tex"
    results = df_combined_both.to_markdown(tablefmt="latex", floatfmt=".2f")
    print(results, file=open(filename, "w"))

    filename = f"{asnwd}/results/mts-combined-results-{metric}.md"
    results = df_combined_both.to_markdown(floatfmt=".2f")
    print(results, file=open(filename, "w"))

    print(results)

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate table of results for precision, recall and accuracy with regards to the MTS benchmark datasets"
    )

    parser.add_argument(
        "-m",
        "--metric",
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
        action="store_true",
        help="Whether to save results to disk or not",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
        print(argsdict)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    metric = args.metric
    save = args.save

    architectures = ["atx", "t2"]
    if args.architecture is not None:
        architecture = args.architecture
        architectures = [arch for arch in architectures if arch == architecture]

        update_results(architectures, metric, save)

        logging.info(
            "Only updating tables. For full comparson, both architectures required"
        )
    else:
        update_results(architectures, metric, save)
        results = make_comparison_table(architectures, metric)
