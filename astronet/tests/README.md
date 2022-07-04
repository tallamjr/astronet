# Tests

```bash
$ tree . I "baseline|pytest.log"
.
├── README.md
├── __init__.py
├── conftest.py
├── func
│   ├── __init__.py
│   ├── astronet
│   │   └── tests
│   ├── test_train.py
│   └── test_wisdm_train.py
├── int
│   ├── __init__.py
│   ├── test_gp_interpolation.py
│   └── test_numpy_api.py
├── reg
│   ├── astronet
│   │   └── tests
│   ├── lnprofile.py
│   ├── lnprofile.py.lprof
│   ├── pytest-plots.sh
│   ├── test_inference.py
│   ├── test_plots.py
│   └── test_profiling.py
└── unit
    ├── __init__.py
    ├── atx
    │   ├── __init__.py
    │   ├── test_atx_model.py
    │   └── test_tf_dense_layer.py
    ├── t2
    │   ├── __init__.py
    │   ├── astronet
    │   │   └── tests
    │   ├── test_attention.py
    │   ├── test_t2_model.py
    │   ├── test_tf_dense_layer_example.py
    │   ├── test_tf_multihead_attention.py
    │   └── test_transformer.py
    ├── test_import.py
    ├── test_metrics.py
    ├── test_preprocess.py
    ├── test_utils.py
    ├── tinho
    │   └── test_tinho_model.py
    └── viz
        └── test_visualise_data.ipynb

14 directories, 31 files
```

Above is the directory structure of the tests, separated into unit, integration and functional tests

**Unit Tests `unit`**: Testing smallest units or modules individually.

**Integration Tests `int`**: Testing integration of two or more units/modules combined for performing tasks.

**Functional Tests `func`**: Testing the behaviour of the application as per the requirement.

**Regression Tests `reg`**: Ensure the results obtained are the same as before

### Running Tests

This package uses `pytest` with additional options defined in `pytest.ini` file.

To run, simply use `pytest .`

Note: some tests require large data files

If a new plot is created, it should be visually inspected and a new baseline generated like so:

🚧 **NEEDS TO BE UPDATE BELOW** 🚧

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

