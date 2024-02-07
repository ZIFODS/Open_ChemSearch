from __future__ import annotations

import abc
from pathlib import Path

import numpy as np
import pandas as pd

from chemsearch.molecule import Molecule, MoleculeFactory


class Fingerprints(abc.ABC):
    @abc.abstractmethod
    def check_compatible(self, molecules: list[Molecule]) -> None:
        """Check for inconsistencies between molecules and fingerprints.

        Args:
            molecules (list[Molecule]): Molecules.

        Raises:
            AssertionError: If molecules and fingerprints are inconsistent.
        """
        pass

    @abc.abstractclassmethod
    def from_file(cls, filepath: Path) -> Fingerprints:
        """Read fingerprints from numpy binary file.

        Args:
            filepath (Path): Path to file.

        Returns:
            Fingerprints: Fingerprints.
        """
        pass

    @abc.abstractclassmethod
    def from_mol_blocks(
        cls, mol_blocks: list[str], length: int, factory: MoleculeFactory
    ) -> Fingerprints:
        """Generate fingerprints for molecules.

        Args:
            mol_blocks (list[str]): MOL blocks.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Fingerprints: Fingerprints.
        """
        pass

    @abc.abstractclassmethod
    def from_molecules(cls, molecules: list[Molecule], length: int) -> Fingerprints:
        """Calculate fingerprints for molecules.

        Args:
            molecules (list[Molecule]): Molecules.
            length (int): Length of bit vector.

        Returns:
            Fingerprints: Fingerprints.
        """
        pass

    @abc.abstractclassmethod
    def from_smiles(
        cls, smiles: list[str], length: int, factory: MoleculeFactory
    ) -> Fingerprints:
        """Calculate fingerprints for molecules.

        Args:
            smiles (list[str]): SMILES strings.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Fingerprints: Fingerprints.
        """
        pass

    @abc.abstractmethod
    def to_file(self, filepath: Path) -> None:
        """Write fingerprints to numpy binary file.

        Args:
            filepath (Path): Path to file.
        """
        pass

    @abc.abstractmethod
    def screen(self, query: np.ndarray) -> np.ndarray:
        """Determine which fingerprints are a match for substructure query.

        A match is where all bits in the query fingerprint are present.

        Args:
            query (np.ndarray): Fingerprint of substructure query.

        Returns:
            np.ndarray: Boolean array of hits in matrix.
        """
        pass

    def _calculate_fingerprints_from_mol_blocks(
        df: pd.DataFrame, length: int, factory: MoleculeFactory
    ) -> pd.DataFrame:
        molecules = df["MOL Block"].apply(factory.from_mol_block).to_list()
        fps = [mol.get_substructure_fingerprint(length) for mol in molecules]
        return pd.Series(fps, name="Fingerprint", dtype=np.dtype("O"))

    def _calculate_fingerprints_from_smiles(
        df: pd.DataFrame, length: int, factory: MoleculeFactory
    ) -> pd.DataFrame:
        molecules = df["SMILES"].apply(factory.from_smiles).to_list()
        fps = [mol.get_substructure_fingerprint(length) for mol in molecules]
        return pd.Series(fps, name="Fingerprint", dtype=np.dtype("O"))
