from pathlib import Path

# Central passbands wavelengths
pb_wavelengths = {
    "lsstu": 3685.0,
    "lsstg": 4802.0,
    "lsstr": 6231.0,
    "lssti": 7542.0,
    "lsstz": 8690.0,
    "lssty": 9736.0,
}

# Colours for the plotting light curves
pb_colors = {
    "lsstu": "#984ea3",  # Purple: https://www.color-hex.com/color/984ea3
    "lsstg": "#4daf4a",  # Green: https://www.color-hex.com/color/4daf4a
    "lsstr": "#e41a1c",  # Red: https://www.color-hex.com/color/e41a1c
    "lssti": "#377eb8",  # Blue: https://www.color-hex.com/color/377eb8
    "lsstz": "#ff7f00",  # Orange: https://www.color-hex.com/color/ff7f00
    "lssty": "#e3c530",  # Yellow: https://www.color-hex.com/color/e3c530
}

astronet_working_directory = f"{Path(__file__).absolute().parent.parent.parent}"

plasticc_weights_dict = {
    6: 1 / 18,
    15: 1 / 9,
    16: 1 / 18,
    42: 1 / 18,
    52: 1 / 18,
    53: 1 / 18,
    62: 1 / 18,
    64: 1 / 9,
    65: 1 / 18,
    67: 1 / 18,
    88: 1 / 18,
    90: 1 / 18,
    92: 1 / 18,
    95: 1 / 18,
    99: 1 / 19,
    1: 1 / 18,
    2: 1 / 18,
    3: 1 / 18,
}
