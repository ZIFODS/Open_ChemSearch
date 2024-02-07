from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, wait

from chemsearch import io
from chemsearch.molecule import Molecule, MoleculeFactory

from . import Database


class DaskDatabase(Database):
    """This class requires a Dask cluster to be running."""

    def __init__(self, ddf: dd.DataFrame, persist: bool = True):
        self._ddf = ddf
        self._smiles = self._ddf["SMILES"]

        if persist:
            self._persist()

        # Divisions must be calculated for mask to work with partition info
        self._ddf.divisions = calculate_divisions(self._ddf)

    def _persist(self) -> None:
        client = Client.current()
        workers = tuple(client.scheduler_info()["workers"].keys())

        # Cache dataframe across workers
        self._ddf = client.persist(self._ddf, workers=workers)

        # Ensure even distribution of partitions across workers
        wait(self._ddf)
        client.rebalance()

        # Cache SMILES strings outside workers
        self._smiles = self._smiles.compute()

    @classmethod
    def from_parquet(
        cls,
        filepath: Path,
        factory: MoleculeFactory,
        partitions_per_thread: int = 1,
        persist: bool = True,
        **kwargs,
    ) -> DaskDatabase:
        """Create molecules database from parquet file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.
            partitions_per_thread (int, optional): Partitions per thread. Defaults to 1.
            persist (bool, optional): Persist data to local cluster. Defaults to True.

        Returns:
            DaskDatabase: Molecules database.
        """
        ddf = dd.read_parquet(filepath)
        ddf = ddf.repartition(npartitions=cls._get_npartitions(partitions_per_thread))

        ddf["Molecule"] = ddf["Molecule"].apply(
            factory.from_bytes, meta=pd.Series(dtype=np.dtype("O"))
        )

        return cls(ddf, persist=persist)

    @classmethod
    def from_sdf_file(
        cls,
        filepath: Path,
        factory: MoleculeFactory,
        partitions_per_thread: int = 1,
        persist: bool = True,
        **kwargs,
    ) -> DaskDatabase:
        """Create molecules database from SDF file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.
            partitions_per_thread (int, optional): Partitions per thread. Defaults to 1.
            persist (bool, optional): Persist data to local cluster. Defaults to True.

        Returns:
            DaskDatabase: Molecules database.
        """
        ddf = dd.from_pandas(
            pd.DataFrame({"MOL Block": io.read_sdf_file(filepath)}),
            npartitions=cls._get_npartitions(partitions_per_thread),
        )

        ddf["Molecule"] = ddf["MOL Block"].apply(
            factory.from_mol_block, meta=pd.Series(dtype=np.dtype("O"))
        )
        ddf = ddf.drop(columns=["MOL Block"])

        ddf["SMILES"] = ddf["Molecule"].apply(
            lambda x: x.to_smiles(canonical=True), meta=pd.Series(dtype=np.dtype("O"))
        )

        return cls(ddf, persist=persist)

    @classmethod
    def from_smi_file(
        cls,
        filepath: Path,
        factory: MoleculeFactory,
        partitions_per_thread: int = 1,
        persist: bool = True,
        **kwargs,
    ) -> DaskDatabase:
        """Create molecules database from SMI file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.
            partitions_per_thread (int, optional): Partitions per thread. Defaults to 1.
            persist (bool, optional): Persist data to local cluster. Defaults to True.

        Returns:
            DaskDatabase: Molecules database.
        """
        df = pd.DataFrame({"SMILES": io.read_smi_file(filepath)})

        ddf = dd.from_pandas(
            df, npartitions=cls._get_npartitions(partitions_per_thread)
        )

        ddf = ddf.map_partitions(
            cls._add_molecules_from_smiles,
            factory=factory,
            meta={"SMILES": np.dtype("O"), "Molecule": np.dtype("O")},
        )

        return cls(ddf, persist=persist)

    def get_substructure_hits(
        self, query: Molecule, mask: np.ndarray | None = None
    ) -> list[str]:
        """Determine which molecules contain the substructure by subgraph isomorphism.

        Args:
            query (Molecule): Substructure query.
            mask (np.ndarray | None, optional): Boolean array of molecules to screen.
                Defaults to None.

        Returns:
            list[str]: SMILES strings of hits.
        """
        if (mask is not None) and (mask.sum() == 0):
            return []

        hits = self._ddf.map_partitions(
            self._get_substructure_hits, query=query, mask=mask, meta=("hits", int)
        ).compute()

        return self._smiles.loc[hits].to_list()

    @property
    def molecules(self) -> list[Molecule]:
        """Get all molecules in database.

        Returns:
            list[Molecule]: All molecules.
        """
        return list(self._ddf["Molecule"])

    @property
    def smiles(self) -> list[str]:
        """Get all SMILES strings in database.

        Returns:
            list[Molecule]: All SMILES strings.
        """
        return list(self._smiles)

    @staticmethod
    def _add_molecules_from_smiles(
        df: pd.DataFrame, factory: MoleculeFactory
    ) -> pd.DataFrame:
        return df.assign(Molecule=df["SMILES"].apply(lambda x: factory.from_smiles(x)))

    @staticmethod
    def _get_npartitions(partitions_per_thread) -> int:
        return partitions_per_thread * get_thread_count()

    @staticmethod
    def _get_substructure_hits(
        df: pd.DataFrame, query: Molecule, mask: np.ndarray | None, partition_info: dict
    ) -> pd.Index:
        if mask is not None:
            mask = mask[
                partition_info["division"] : partition_info["division"] + len(df)
            ]
            df = df.loc[mask]

        hits = df.loc[df["Molecule"].apply(lambda x: x.has_substructure(query))].index

        return hits


def calculate_divisions(ddf: dd.DataFrame) -> list[int]:
    """Calculate start and stop indices for each partition in dataframe.

    Args:
        ddf (dd.DataFrame): Dask dataframe.

    Returns:
        list[int]: Division indices.
    """
    divisions = set()
    for partition in ddf.partitions:
        index = partition.index.compute()
        divisions.add(index.start)
        divisions.add(index.stop)

    return tuple(sorted(divisions))


def get_thread_count() -> int:
    """Get count of threads across all workers in cluster.

    Returns:
        int: Thread count.
    """
    client = Client.current()
    workers = client.cluster.workers.values()

    threads = 0
    for worker in workers:
        threads += worker.nthreads

    return threads
