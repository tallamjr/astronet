import polars as pl

pl.Config.set_tbl_rows(20)

# =========== FLUX DATA ===========
df = pl.scan_csv("plasticc_*lightcurves*.csv")

df = df.lazy().with_columns(
    [
        pl.col("object_id").cast(pl.UInt64, strict=False),
        pl.col("mjd").cast(pl.Float32, strict=False),
        pl.col("passband").cast(pl.UInt8, strict=False),
        pl.col("flux").cast(pl.Float32, strict=False),
        pl.col("flux_err").cast(pl.Float32, strict=False),
        pl.col("detected_bool").cast(pl.Boolean, strict=False),
    ]
)

df.sink_parquet("plasticc_test_lightcurves.parquet")

print(df.head().collect())

# =========== METADATA ===========
mdf = pl.scan_csv("*meta*")
anon_class = {991, 992, 993, 994}

df = (
    mdf.select(["object_id", "hostgal_photoz", "hostgal_photoz_err", "true_target"])
    .filter(~pl.col("true_target").is_in(anon_class))
    .rename({"true_target": "target"})
)

# df.select("true_target").to_series().value_counts()

df = df.with_columns(
    [
        pl.col("object_id").cast(pl.UInt64, strict=False),
        pl.col("hostgal_photoz").cast(pl.Float32, strict=False),
        pl.col("hostgal_photoz_err").cast(pl.Float32, strict=False),
        pl.col("target").cast(pl.UInt8, strict=False),
    ]
)

df.sink_parquet("plasticc_test_metadata.parquet")

print(df.head().collect())
