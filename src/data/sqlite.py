from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import sqlalchemy
from tqdm import tqdm
from typing import List, Optional

from graphnet.utilities.logging import get_logger
from graphnet.data.sqlite.sqlite_utilities import create_table

from ..constants import log_dir


logger = get_logger()


def load_input(
    meta_batch: pd.DataFrame, input_data_folder: str, geometry_table: pd.DataFrame
) -> pd.DataFrame:
    """
    Will load the corresponding detector readings associated with the meta data batch.
    """
    batch_id = pd.unique(meta_batch["batch_id"])

    assert (
        len(batch_id) == 1
    ), "contains multiple batch_ids. Did you set the batch_size correctly?"

    detector_readings = pd.read_parquet(
        path=f"{input_data_folder}/batch_{batch_id[0]}.parquet"
    )
    sensor_positions = geometry_table.loc[
        detector_readings["sensor_id"], ["x", "y", "z"]
    ]
    sensor_positions.index = detector_readings.index

    for column in sensor_positions.columns:
        if column not in detector_readings.columns:
            detector_readings[column] = sensor_positions[column]

    detector_readings["auxiliary"] = detector_readings["auxiliary"].replace(
        {True: 1, False: 0}
    )
    return detector_readings.reset_index()


def add_to_table(
    database_path: str,
    df: pd.DataFrame,
    table_name: str,
    is_primary_key: bool,
) -> None:
    """Writes meta data to sqlite table.

    Args:
        database_path (str): the path to the database file.
        df (pd.DataFrame): the dataframe that is being written to table.
        table_name (str, optional): The name of the meta table. Defaults to 'meta_table'.
        is_primary_key(bool): Must be True if each row of df corresponds to a unique event_id. Defaults to False.
    """
    try:
        create_table(
            columns=df.columns,
            database_path=database_path,
            table_name=table_name,
            integer_primary_key=is_primary_key,
            index_column="event_id",
        )
    except sqlite3.OperationalError as e:
        if "already exists" in str(e):
            pass
        else:
            raise e
    engine = sqlalchemy.create_engine("sqlite:///" + str(database_path))

    df.to_sql(table_name, con=engine, index=False, if_exists="append", chunksize=200000)
    engine.dispose()
    return


def convert_to_sqlite(
    meta_data_path: Path,
    database_path: Path,
    geometry_path: Path,
    input_data_folder: Path,
    batch_ids: Optional[List[int]] = None,
) -> None:
    """Converts a selection of the Competition's parquet files to a single sqlite database.

    Args:
        meta_data_path (Path): Path to the meta data file.
        database_path (Path): path to database. E.g. '/my_folder/data/my_new_database.db'
        geometry_path (Path): Path to geometry file.
        input_data_folder (Path): folder containing the parquet input files.
        batch_ids (List[int]): The batch_ids you want converted. Defaults to None (all batches will be converted)
    """
    if batch_ids is None:
        batch_ids = np.arange(1, 661, 1).to_list()
    else:
        assert isinstance(batch_ids, list), "Variable 'batch_ids' must be list."

    if database_path.suffix == "":
        database_path /= ".db"

    meta_data = pd.read_parquet(meta_data_path)
    geometry_table = pd.read_csv(geometry_path)

    for batch_id in tqdm(batch_ids, desc="Convert to sqlite"):
        meta_data_batch = meta_data[meta_data["batch_id"] == batch_id]
        if len(meta_data_batch) == 0:
            logger.warning(f"No data found in batch {batch_id}")
        
        # Add to meta table
        add_to_table(
            database_path=database_path,
            df=meta_data_batch,
            table_name="meta_table",
            is_primary_key=True,
        )
        
        # Add to pulse table
        pulses = load_input(
            meta_batch=meta_data_batch,
            input_data_folder=input_data_folder,
            geometry_table=geometry_table,
        )
        del meta_data_batch  # memory
        add_to_table(
            database_path=database_path,
            df=pulses,
            table_name="pulse_table",
            is_primary_key=False,
        )
        del pulses  # memory

    print(f"Conversion Complete!. Database available at\n {database_path}")
