import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseSettings, validator

from chemsearch.constants import (
    LOG_DIR,
    RESULTS_DIR,
    Cartridges,
    Databases,
    FingerprintMethods,
)
from chemsearch.io import is_s3_uri


class Settings(BaseSettings):
    molecules: str
    fingerprints: str
    bit_length: int
    cartridge: Cartridges = Cartridges.RDKit
    database: Databases = Databases.DASK
    processes: int = 8
    threads_per_process: int = 1
    partitions_per_thread: int = 1
    fingerprint_method: FingerprintMethods = FingerprintMethods.GPU
    log_file: Path = LOG_DIR / f"app_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S.log')}"
    output_dir: str = str(RESULTS_DIR)

    class Config:
        env_file = ".env"

    def __hash__(self):
        return hash((type(self),) + tuple(self.__dict__.values()))

    @validator("molecules")
    def _check_molecules_file_exists(cls, filepath):
        if not is_s3_uri(filepath) and not Path(filepath).exists():
            raise FileNotFoundError(f"Molecules file not found at: {filepath}")
        return filepath

    @validator("fingerprints")
    def _check_fingerprints_file_exists(cls, filepath):
        if not is_s3_uri(filepath) and not Path(filepath).exists():
            raise FileNotFoundError(f"Fingerprints file not found at: {filepath}")
        return filepath


def read_settings_from_json(filepath: Path) -> list[Settings]:
    """Read all configurations from settings JSON file.

    Args:
        filepath (Path): Path to JSON file.

    Returns:
        list[Settings]: All configurations.
    """
    with open(filepath) as fh:
        lines = json.loads(fh.read())

    all_settings = [Settings(**i) for i in lines]

    return all_settings
