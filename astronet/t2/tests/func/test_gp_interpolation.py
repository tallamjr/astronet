import numpy as np
import pandas as pd
import pytest

from astronet.t2.constants import pb_wavelengths, astronet_working_directory as asnwd
from astronet.t2.preprocess import predict_2d_gp, fit_2d_gp
from astronet.t2.utils import __transient_trim, __filter_dataframe_only_supernova, __remap_filters


def test_plasticc_gp_interpolation():

    data = pd.read_csv(
        f"{asnwd}/data/plasticc/training_set.csv",
        sep=",",
    )
    data = __remap_filters(df=data)
    data.rename(
        {"flux_err": "flux_error"}, axis="columns", inplace=True
    )  # snmachine and PLAsTiCC uses a different denomination

    filters = data['filter']
    filters = list(np.unique(filters))
    assert len(filters) == 6

    assert data.shape == (1421705, 6)
    df = __filter_dataframe_only_supernova(
        f"{asnwd}/data/plasticc/train_subset.txt",
        data,
    )
    assert df.shape == (764572, 6)

    object_list = list(np.unique(df['object_id']))
    object_list = object_list[2:3]
    assert object_list == [1124]

    obs_transient_single = __transient_trim(object_list, df)

    gp_predict = fit_2d_gp(obs_transient_single)
    number_gp = 100
    gp_times = np.linspace(min(obs_transient_single['mjd']), max(obs_transient_single['mjd']), number_gp)
    assert gp_times.shape == (100,)

    gp_wavelengths = np.vectorize(pb_wavelengths.get)(filters)
    assert gp_wavelengths.shape == (6,)

    obj_gps = predict_2d_gp(gp_predict, gp_times, gp_wavelengths)
    inverse_pb_wavelengths = {v: k for k, v in pb_wavelengths.items()}
    obj_gps['filter'] = obj_gps['filter'].map(inverse_pb_wavelengths)

    assert obj_gps['flux'].iat[0] == pytest.approx(8.266201199288844, 0.0000001)
    assert obj_gps['flux'].iat[3] == pytest.approx(4.777761407937418, 0.0000001)
