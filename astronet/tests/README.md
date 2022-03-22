# Tests

```bash

.
├── README.md
├── func
│   ├── test_model_save_load_predict.py
│   └── test_train.py
├── int
│   ├── test_gp_interpolation.py
│   └── test_numpy_api.py
└── unit
    ├── atx
    │   ├── test_atx_model.py
    │   └── test_tf_dense_layer.py
    ├── t2
    │   ├── test_attention.py
    │   ├── test_t2_model.py
    │   ├── test_tf_dense_layer.py
    │   ├── test_tf_multihead_attention.py
    │   └── test_transformer.py
    ├── test_evaluate.py
    ├── test_import.py
    ├── test_metrics.py
    ├── test_preprocess.py
    ├── test_tf_dense_layer.py
    └── test_utils.py

5 directories, 18 files
```

Above is the directory structure of the tests, separated into unit, integration and functional tests

#### Unit Tests `unit`

Testing smallest units or modules individually.

#### Integration Tests `int`

Testing integration of two or more units/modules combined for performing tasks.

#### Functional Tests `unit`

Testing the behaviour of the application as per the requirement.

### Running Tests

This package uses `pytest` with additional options defined in `pytest.ini` file.

To run, simply use `pytest .`

Note: some tests require large data files

If a new plot is created, it should be visually inspected and a new baseline generated like so:

```bash
$ cd astronet/tests/unit/viz
$ pytest --mpl-generate-path=baseline --ignore-glob="*.ipynb" test_plots.py
```
Then, the hash of the image is to be stored in the SHA library file with:

```bash
$ pytest --mpl-generate-hash-library=astronet/tests/unit/viz/baseline/hashlib.json --ignore-glob="*.ipynb" test_plots.py
```
Finally, the suite is ready to be tested by running:
```bash
$ pytest --ignore-glob="*.ipynb" test_plots.py
```
