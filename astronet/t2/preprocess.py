def robust_scale(df_train, df_val, df_test, scale_columns):
    from sklearn.preprocessing import RobustScaler

    scaler = RobustScaler()

    scaler = scaler.fit(df_train[scale_columns])

    df_train.loc[:, scale_columns] = scaler.transform(
        df_train[scale_columns].to_numpy()
    )
    df_val.loc[:, scale_columns] = scaler.transform(
        df_val[scale_columns].to_numpy()
    )
    df_test.loc[:, scale_columns] = scaler.transform(
        df_test[scale_columns].to_numpy()
    )


def one_hot_encode(y_train, y_val, y_test):
    from sklearn.preprocessing import OneHotEncoder

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

    enc = enc.fit(y_train)

    y_train = enc.transform(y_train)
    y_val = enc.transform(y_val)
    y_test = enc.transform(y_test)

    return enc, y_train, y_val, y_test
