import argparse
import numpy as np
import pandas as pd
import sys

from pathlib import Path

from astronet.constants import astronet_working_directory as asnwd


def chunker(seq, size):
    """https://stackoverflow.com/a/434328/4521950"""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def read_large_file(filename):
    df = pd.read_csv(filename)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process PLAsTiCC Test Set in Chunks')

    parser.add_argument(
        "-n",
        "--num_objects",
        default=10000,
        type=int,
        help="Choose which dataset to process",
    )

    parser.add_argument(
        "-f",
        "--file",
        default=f"{asnwd}/data/plasticc/test_set/plasticc_test_lightcurves_01.csv",
        help="Choose which dataset to process",
    )

    try:
        args = parser.parse_args()
        argsdict = vars(args)
        print(argsdict)
    except KeyError:
        parser.print_help()
        sys.exit(0)

    filename = args.file
    num_objects_per_chunk = args.num_objects

    df = read_large_file(filename)

    object_list = list(np.unique(df['object_id']))

    for idx, objects in enumerate(chunker(object_list, num_objects_per_chunk)):
        dfs = df[df['object_id'].isin(objects)]
        print(idx, dfs.shape)
        dfs.to_csv(f"{asnwd}/data/plasticc/test_set/{Path(filename).stem}_chunk_{idx}.csv")
