import numpy as np
import polars as pl

from astronet.constants import LSST_FILTER_MAP
from astronet.preprocess import (
    generate_gp_all_objects,
)

SEED = 9001

# flux data
df = pl.scan_parquet("plasticc_*_lightcurves.parquet")

# metadata
mdf = pl.scan_parquet("plasticc_*_metadata.parquet")

df = (
    df.rename(
        {"passband": "filter", "flux_err": "flux_error", "detected_bool": "detected"}
    )
    .with_columns(pl.col("filter").map_dict(LSST_FILTER_MAP))
    .filter(pl.col("detected") is True)
    .collect(streaming=True)
)

object_list = df.select("object_id").unique().to_series().to_list()

print(f"NUM OBJECTS TO BE PROCESSED: {len(object_list)}")

chunk_list = np.array_split(object_list, 100)

schema = {
    "mjd": pl.Float64,
    "lsstu": pl.Float64,
    "lsstg": pl.Float64,
    "lsstr": pl.Float64,
    "lssti": pl.Float64,
    "lsstz": pl.Float64,
    "lssty": pl.Float64,
    "object_id": pl.Int64,
}

gdf = pl.DataFrame(
    data=[],
    schema=schema,
)

for num, chunk in enumerate(chunk_list):
    # Chunking required due to python code within GP fitting function.
    print(f"ITERATION : {num}")

    ddf = df.filter(pl.col("object_id").is_in(chunk_list[num].tolist()))

    print(f"NUM ALERTS IN CHUNK : {len(chunk)}")
    gdf = generate_gp_all_objects(chunk, ddf)

    assert gdf.select("object_id").unique().height == len(chunk)

    time_series_feats = [
        "mjd",
        "lsstg",
        "lssti",
        "lsstr",
        "lsstu",
        "lssty",
        "lsstz",
    ]

    assert gdf.shape == (
        len(chunk) * 100,
        len(time_series_feats) + len(["object_id"]),
    )

    jdf = gdf.lazy().join(
        mdf.with_columns(pl.col("object_id").cast(pl.Int64)),
        on="object_id",
        how="inner",
    )

    pdf = jdf.with_columns(
        [
            pl.col("mjd").cast(pl.Float32, strict=False),
            pl.col("^lsst.*$").cast(pl.Float32, strict=False),
            pl.col("object_id").cast(pl.UInt64, strict=False),
            pl.col("hostgal_photoz").cast(pl.Float32, strict=False),
            pl.col("hostgal_photoz_err").cast(pl.Float32, strict=False),
            pl.col("target").cast(pl.UInt8, strict=False),
        ]
    )

    pdf.sink_parquet(f"chunks/chunk-{num:03d}.parquet")

    print(pdf.head().collect())
