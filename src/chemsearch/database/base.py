from __future__ import annotations

import abc
from pathlib import Path

import numpy as np

from chemsearch.molecule import Molecule, MoleculeFactory


class Database(abc.ABC):
    @abc.abstractclassmethod
    def from_parquet(cls, filepath: Path) -> Database:
        """Create molecules database from parquet file.

        Args:
            filepath (Path): Path to file.

        Returns:
            Database: Molecules database.
        """
        pass

    @abc.abstractclassmethod
    def from_sdf_file(cls, filepath: Path, factory: MoleculeFactory) -> Database:
        """Create molecules database from SDF file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Database: Molecules database.
        """
        pass

    @abc.abstractclassmethod
    def from_smi_file(cls, filepath: Path, factory: MoleculeFactory) -> Database:
        """Create molecules database from SMI file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Database: Molecules database.
        """
        pass

    @abc.abstractmethod
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
        pass

    @abc.abstractproperty
    def molecules(self) -> list[Molecule]:
        """Get all molecules in database.

        Returns:
            list[Molecule]: All molecules.
        """
        pass

    @abc.abstractproperty
    def smiles(self) -> list[str]:
        """Get all SMILES strings in database.

        Returns:
            list[Molecule]: All SMILES strings.
        """
        pass
