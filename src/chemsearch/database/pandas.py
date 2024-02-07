from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from chemsearch import io
from chemsearch.molecule import Molecule, MoleculeFactory

from . import Database
from .schema import MoleculesSchema


class PandasDatabase(Database):
    def __init__(self, df: pd.DataFrame):
        self._df = df

    @classmethod
    def from_parquet(
        cls, filepath: Path, factory: MoleculeFactory, **kwargs
    ) -> PandasDatabase:
        """Create molecules database from parquet file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            PandasDatabase: Molecules database.
        """
        df = pd.read_parquet(filepath)

        df["Molecule"] = df["Molecule"].apply(factory.from_bytes)

        MoleculesSchema.validate(df)

        return cls(df)

    @classmethod
    def from_sdf_file(
        cls, filepath: Path, factory: MoleculeFactory, **kwargs
    ) -> PandasDatabase:
        """Create molecules database from SDF file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            PandasDatabase: Molecules database.
        """
        df = pd.DataFrame()

        mol_blocks = io.read_sdf_file(filepath)
        df["Molecule"] = io.read_mol_blocks(mol_blocks, factory)

        df["SMILES"] = io.serialise_as_smiles(df["Molecule"])

        return cls(df)

    @classmethod
    def from_smi_file(
        cls, filepath: Path, factory: MoleculeFactory, **kwargs
    ) -> PandasDatabase:
        """Create molecules database from SMI file.

        Args:
            filepath (Path): Path to file.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            PandasDatabase: Molecules database.
        """
        df = pd.DataFrame()
        df["SMILES"] = io.read_smi_file(filepath)

        df["Molecule"] = io.read_smiles_strings(df["SMILES"], factory)

        return cls(df)

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
        df = self._df.loc[mask] if mask is not None else self._df

        hits = df.loc[
            df["Molecule"].apply(lambda x: x.has_substructure(query)), "SMILES"
        ]

        return hits.to_list()

    @property
    def molecules(self) -> list[Molecule]:
        """Get all molecules in database.

        Returns:
            list[Molecule]: All molecules.
        """
        return self._df["Molecule"].to_list()

    @property
    def smiles(self) -> list[str]:
        """Get all SMILES strings in database.

        Returns:
            list[Molecule]: All SMILES strings.
        """
        return self._df["SMILES"].to_list()
