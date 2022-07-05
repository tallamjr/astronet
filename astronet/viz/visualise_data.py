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

from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pylab import rcParams

rcParams["figure.figsize"] = 12, 8


def plot_wisdm_activity(activity: str, df: pd.DataFrame, cols: List) -> None:
    # TODO: Update docstrings
    data = df[df["activity"] == activity][cols][:400]
    axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axis:
        ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.5))


def plot_event(
    object_name: str, df: pd.DataFrame, filters: List, pb_colors: Dict
) -> None:
    # TODO: Update docstrings
    f, ax = plt.subplots()
    for passband in filters:
        data = df[df["object_id"] == object_name]
        data = data[data["filter"] == passband]
        ax.errorbar(
            x=data["mjd"],
            y=data["flux"],
            yerr=data["flux_error"],
            linestyle="none",
            marker="o",
            color=pb_colors[passband],
        )

    return f, ax


def plot_event_data_with_model(
    obj_data,
    obj_model=None,
    number_col: int = 2,
    show_title: bool = False,
    show_legend: bool = True,
    pb_colors: Dict = {},
) -> None:
    # TODO: Update docstrings
    """Plots real data and model fluxes at the corresponding mjd"""

    f, ax = plt.subplots()

    passbands = list(np.unique(obj_data["filter"]))
    for pb in passbands:
        obj_data_pb = obj_data[obj_data["filter"] == pb]  # obj LC in that passband
        if obj_model is not None:
            obj_model_pb = obj_model[obj_model["filter"] == pb]
            model_flux = obj_model_pb["flux"]
            ax.plot(
                obj_model_pb["mjd"],
                model_flux,
                color=pb_colors[pb],
                alpha=0.7,
                label="",
            )
            try:
                model_flux_error = obj_model_pb["flux_error"]
                ax.fill_between(
                    x=obj_model_pb["mjd"],
                    y1=model_flux - model_flux_error,
                    y2=model_flux + model_flux_error,
                    color=pb_colors[pb],
                    alpha=0.15,
                    label=None,
                )
            except Exception as e:
                print(e)
        ax.errorbar(
            obj_data_pb["mjd"],
            obj_data_pb["flux"],
            obj_data_pb["flux_error"],
            fmt="o",
            color=pb_colors[pb],
            label=pb[-1],
        )
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Flux units")
    if show_title:
        ax.title(
            "Object ID: {}\nPhoto-z = {:.3f}".format(
                obj_data.meta["name"], obj_data.meta["z"]
            )
        )
    if show_legend:
        ax.legend(
            ncol=number_col,
            handletextpad=0.3,
            borderaxespad=0.3,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )

    return f, ax


def plot_event_gp_mean(
    df: pd.DataFrame, object_id: Union[int, str], pb_colors: Dict
) -> None:
    # TODO: Update docstrings

    df = df[df["object_id"] == object_id]

    drop_columns = ["object_id"]
    if "target" in df.columns:
        print(object_id, df["target"].values[0])
        drop_columns.append("target")
    gp_mean_data = pd.DataFrame(data=df, columns=df.columns).drop(columns=drop_columns)
    ax = gp_mean_data.set_index("mjd").plot(linewidth=2.0, color=pb_colors, marker="o")
    f = ax.get_figure()

    return f, ax
