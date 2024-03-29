from pathlib import Path

project_dir = Path("/media/eden/sandisk/projects/icecube")

# Data
input_dir = project_dir / "input"
raw_data_dir = input_dir / "icecube-neutrinos-in-deep-ice"
database_dir = input_dir / "sqlite"
metadata_path = raw_data_dir / "train_meta.parquet"
geometry_path = raw_data_dir / "sensor_geometry.csv"
train_dir = raw_data_dir / "train"

# Logs
log_dir = project_dir / "logs"
