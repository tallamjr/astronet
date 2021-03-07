import numpy as np

from astronet.constants import astronet_working_directory as asnwd

from astronet.utils import load_dataset

X_train, y_train, X_test, y_test, loss, Z_train, Z_test = load_dataset(
    dataset="plasticc",
    redshift=True,
    testset=True
)

X_train_arrays = np.array_split(X_train, 2)
X_train_1 = X_train_arrays[0]
X_train_2 = X_train_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_train_1.npy",
    X_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_train_2.npy",
    X_train_2,
)

y_train_arrays = np.array_split(y_train, 2)
y_train_1 = y_train_arrays[0]
y_train_2 = y_train_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_train_1.npy",
    y_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_train_2.npy",
    y_train_2,
)

X_test_arrays = np.array_split(X_test, 2)
X_test_1 = X_test_arrays[0]
X_test_2 = X_test_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_test_1.npy",
    X_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/X_test_2.npy",
    X_test_2,
)


y_test_arrays = np.array_split(y_test, 2)
y_test_1 = y_test_arrays[0]
y_test_2 = y_test_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_test_1.npy",
    y_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/y_test_2.npy",
    y_test_2,
)

Z_train_arrays = np.array_split(Z_train, 2)
Z_train_1 = Z_train_arrays[0]
Z_train_2 = Z_train_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_train_1.npy",
    Z_train_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_train_2.npy",
    Z_train_2,
)

Z_test_arrays = np.array_split(Z_test, 2)
Z_test_1 = Z_test_arrays[0]
Z_test_2 = Z_test_arrays[1]
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_test_1.npy",
    Z_test_1,
)
np.save(
    f"{asnwd}/data/plasticc/test_set/splits/Z_test_2.npy",
    Z_test_2,
)
