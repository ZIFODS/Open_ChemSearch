from pathlib import Path

from chemsearch.constants import Databases
from chemsearch.molecule import MoleculeFactory

from . import Database
from .dask import DaskDatabase
from .pandas import PandasDatabase


class DatabaseFactory:
    def __init__(self, method: Databases):
        match method:
            case Databases.PANDAS:
                db_class = PandasDatabase

            case Databases.DASK:
                db_class = DaskDatabase

            case _:
                raise ValueError(
                    "Unrecognised database implementation. "
                    "Only Pandas and Dask are supported."
                )

        self._db_class = db_class

    def from_file(self, filepath: Path, factory: MoleculeFactory, **kwargs) -> Database:
        """Create molecules database from file.

        Filetype is inferred from the file extension. Supported filetypes:
            - Parquet
            - SDF
            - SMI

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Raises:
            ValueError: If filetype unrecognised.

        Returns:
            Database: Molecules database.
        """
        match filepath.suffix.lower():
            case ".parquet":
                db = self._db_class.from_parquet(filepath, factory, **kwargs)

            case ".sdf":
                db = self._db_class.from_sdf_file(filepath, factory, **kwargs)

            case ".smi":
                db = self._db_class.from_smi_file(filepath, factory, **kwargs)

            case _:
                raise ValueError(
                    "Unrecognised file extension. "
                    "Only parquet, SDF and SMI files can be read."
                )

        return db
