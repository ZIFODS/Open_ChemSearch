from __future__ import annotations

from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from attrs import define
from tqdm import tqdm

from chemsearch.molecule import Molecule, MoleculeFactory

from . import Fingerprints


@define
class CPUFingerprints(Fingerprints):
    matrix: np.ndarray

    def check_compatible(self, molecules: list[Molecule]) -> None:
        """Check for inconsistencies between molecules and fingerprints.

        Args:
            molecules (list[Molecule]): Molecules.

        Raises:
            AssertionError: If molecules and fingerprints are inconsistent.
        """
        try:
            assert self.matrix.shape[0] == len(molecules)

        except AssertionError:
            raise ValueError(
                "Input files must share the same index."
                f" Different numbers of molecules ({len(molecules)}) and fingerprints"
                f" ({self.matrix.shape[0]}) read."
            )

    @classmethod
    def from_file(cls, filepath: Path) -> CPUFingerprints:
        """Read fingerprints from numpy binary file.

        Args:
            filepath (Path): Path to file.

        Returns:
            CPUFingerprints: Fingerprints.
        """
        matrix = np.load(filepath)

        return cls(matrix)

    @classmethod
    def from_mol_blocks(
        cls,
        mol_blocks: list[str],
        length: int,
        factory: MoleculeFactory,
        chunk_size: int = 1_000,
    ) -> CPUFingerprints:
        """Generate fingerprints for molecules.

        Args:
            mol_blocks (list[str]): MOL blocks.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.
            chunk_size (int): Molecules per partition in parallelisation of work.

        Returns:
            CPUFingerprints: Fingerprints.
        """
        ddf = dd.from_pandas(
            pd.DataFrame({"MOL Block": mol_blocks}), chunksize=chunk_size
        )

        matrix = ddf.map_partitions(
            cls._calculate_fingerprints_from_mol_blocks,
            length=length,
            factory=factory,
            meta=("Fingerprint", np.dtype("O")),
        ).compute()
        matrix = np.stack(matrix)

        return cls(matrix)

    @classmethod
    def from_molecules(cls, molecules: list[Molecule], length: int) -> CPUFingerprints:
        """Calculate fingerprints for molecules.

        Args:
            molecules (list[Molecule]): Molecules.
            length (int): Length of bit vector.

        Returns:
            CPUFingerprints: Fingerprints.
        """
        if len(molecules) == 0:
            raise ValueError("At least one molecule should be received.")

        matrix = np.array(
            [
                mol.get_substructure_fingerprint(length)
                for mol in tqdm(molecules, desc="Calculating fingerprints")
            ]
        )

        return cls(matrix)

    @classmethod
    def from_smiles(
        cls,
        smiles: list[str],
        length: int,
        factory: MoleculeFactory,
        chunk_size: int = 1_000,
    ) -> CPUFingerprints:
        """Calculate fingerprints for molecules.

        Args:
            smiles (list[str]): SMILES strings.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.
            chunk_size (int): Molecules per partition in parallelisation of work.

        Returns:
            CPUFingerprints: Fingerprints.
        """
        ddf = dd.from_pandas(pd.DataFrame({"SMILES": smiles}), chunksize=chunk_size)

        matrix = ddf.map_partitions(
            cls._calculate_fingerprints_from_smiles,
            length=length,
            factory=factory,
            meta=("Fingerprint", np.dtype("O")),
        ).compute()
        matrix = np.stack(matrix)

        return cls(matrix)

    def to_file(self, filepath: Path) -> None:
        """Write fingerprints to numpy binary file.

        Args:
            filepath (Path): Path to file.
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)

        np.save(filepath, self.matrix)

    def screen(self, query: np.ndarray) -> np.ndarray:
        """Determine which fingerprints are a match for substructure query.

        A match is where all bits in the query fingerprint are present.

        Args:
            query (np.ndarray): Fingerprint of substructure query.

        Returns:
            np.ndarray: Boolean array of hits in matrix.
        """
        features = query.sum()

        result = np.bitwise_and(query, self.matrix)

        hits = result.sum(axis=1) == features

        return hits
