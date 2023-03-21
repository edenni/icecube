
import json
import multiprocessing
import os
from pathlib import Path

import pandas as pd
import polars as pl
import webdataset as wds
from tqdm import tqdm

from graphnet.data.constants import FEATURES, TRUTH

meta_data_path = Path("../../input/icecube/icecube-neutrinos-in-deep-ice/train_meta.parquet")
geometry_path = Path("../../input/icecube/icecube-neutrinos-in-deep-ice/sensor_geometry.csv")
input_data_path = "../../input/icecube/icecube-neutrinos-in-deep-ice/train/batch_{batch_id}.parquet"
shard_dir = Path("../../input/webdatasets")
shard_dir.mkdir(exist_ok=True)

shard_filename = "../../input/webdatasets/batch-%03d.tar"


from webdataset import TarWriter


def write_one_batch(batch_id, meta_data, geometry_table,):
    pattern = "../../input/webdatasets/batch-%03d.tar"
    print(f"Start saving batch {batch_id}")

    meta_data_batch = meta_data[meta_data.batch_id == batch_id]
    event_ids = meta_data_batch["event_id"].unique()

    df_batch = pd.read_parquet(input_data_path.format(batch_id=batch_id))
    df_batch[["x", "y", "z"]] = geometry_table.loc[df_batch.sensor_id.values, ["x", "y", "z"]].values

    fname = pattern % batch_id
    stream = TarWriter(fname)

    for event_id in tqdm(event_ids, desc=f"Batch {batch_id}"):
        df_event = df_batch[df_batch.index == event_id].copy()
        write_samples_into_single_shard(stream, meta_data_batch, event_id, df_event)
    print(f"Finished saving batch {batch_id}")

    stream.close()


def write_samples_into_single_shard(stream, meta_data_batch, event_id, df_batch):
    truth = meta_data_batch[meta_data_batch.event_id == event_id][TRUTH.KAGGLE].values
    features = df_batch.loc[event_id, FEATURES.KAGGLE].values

    data = {
            "__key__": str(event_id),
            "pickle": (
                features, truth,
            )
        }

    size = stream.write(data)
    return size

batch_ids = range(51, 101)


import joblib

meta_data = pd.read_parquet(meta_data_path)
geometry_table = pd.read_csv(geometry_path) # dtypes={"sensor_id": pl.Int16}
geometry_table = geometry_table.set_index("sensor_id")
joblib.Parallel(n_jobs=4, verbose=11)(joblib.delayed(write_one_batch)(batch_id, meta_data, geometry_table) for batch_id in batch_ids)
