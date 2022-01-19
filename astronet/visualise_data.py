import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pylab import rcParams

from astronet.constants import LSST_PB_COLORS, ZTF_PB_COLORS

rcParams["figure.figsize"] = 12, 8


def plot_wisdm_activity(activity, df, cols):
    # TODO: Update docstrings
    data = df[df["activity"] == activity][cols][:400]
    axis = data.plot(subplots=True, figsize=(16, 12), title=activity)
    for ax in axis:
        ax.legend(loc="lower left", bbox_to_anchor=(1.0, 0.5))


def plot_plasticc_object(object_name, df, filters, pb_colors):
    # TODO: Update docstrings
    for passband in filters:
        data = df[df["object_id"] == object_name]
        data = data[data["filter"] == passband]
        plt.errorbar(
            x=data["mjd"],
            y=data["flux"],
            yerr=data["flux_error"],
            linestyle="none",
            marker="o",
            color=pb_colors[passband],
        )


def plot_plasticc_single_object_gp_mean(df, object_id):
    # TODO: Update docstrings
    df = df[df["object_id"] == object_id]
    print(object_id, df["target"].values[0])
    gp_mean_data = pd.DataFrame(data=df, columns=df.columns).drop(
        columns=["object_id", "target"]
    )
    gp_mean_data.set_index("mjd").plot()


def plot_plasticc_real_data_and_model(
    obj_data,
    obj_model=None,
    number_col=2,
    show_title=False,
    show_legend=True,
    pb_colors=None,
):
    # TODO: Update docstrings
    """Plots real data and model fluxes at the corresponding mjd"""

    passbands = np.unique(obj_data["filter"])
    passbands = ["lsstu", "lsstg", "lsstr", "lssti", "lsstz", "lssty"]
    for pb in passbands:
        obj_data_pb = obj_data[obj_data["filter"] == pb]  # obj LC in that passband
        if obj_model is not None:
            obj_model_pb = obj_model[obj_model["filter"] == pb]
            model_flux = obj_model_pb["flux"]
            plt.plot(
                obj_model_pb["mjd"],
                model_flux,
                color=pb_colors[pb],
                alpha=0.7,
                label="",
            )
            try:
                model_flux_error = obj_model_pb["flux_error"]
                plt.fill_between(
                    x=obj_model_pb["mjd"],
                    y1=model_flux - model_flux_error,
                    y2=model_flux + model_flux_error,
                    color=pb_colors[pb],
                    alpha=0.15,
                    label=None,
                )
            except:
                pass
        plt.errorbar(
            obj_data_pb["mjd"],
            obj_data_pb["flux"],
            obj_data_pb["flux_error"],
            fmt="o",
            color=pb_colors[pb],
            label=pb[-1],
        )
    plt.xlabel("Time (days)")
    plt.ylabel("Flux units")
    if show_title:
        plt.title(
            "Object ID: {}\nPhoto-z = {:.3f}".format(
                obj_data.meta["name"], obj_data.meta["z"]
            )
        )
    if show_legend:
        plt.legend(
            ncol=number_col,
            handletextpad=0.3,
            borderaxespad=0.3,
            labelspacing=0.2,
            borderpad=0.3,
            columnspacing=0.4,
        )
