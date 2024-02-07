from __future__ import annotations

import abc

import numpy as np


class Molecule(abc.ABC):
    @abc.abstractclassmethod
    def from_bytes(cls, array: bytes) -> Molecule:
        """Generate molecule from binary string representation.

        Args:
            array (bytes): Binary string representation of molecule.

        Returns:
            Molecule: Molecule.
        """
        pass

    @abc.abstractclassmethod
    def from_mol_block(cls, text: str) -> Molecule:
        """Generate molecule from contents of MOL file or single SDF record.

        Args:
            text (str): Contents of MOL file or single SDF record.

        Returns:
            Molecule: Molecule.
        """
        pass

    @abc.abstractclassmethod
    def from_smarts(cls, smarts: str) -> Molecule:
        """Generate molecule from SMARTS string.

        Args:
            smarts (str): SMARTS string.

        Returns:
            Molecule: Molecule.
        """
        pass

    @abc.abstractclassmethod
    def from_smiles(cls, smiles: str) -> Molecule:
        """Generate molecule from SMILES string.

        Args:
            smiles (str): SMILES string.

        Returns:
            Molecule: Molecule.
        """
        pass

    @abc.abstractmethod
    def get_substructure_fingerprint(self, length: int) -> np.ndarray:
        """Generate substructure-compatible fingerprint as dense bit vector.

        Args:
            length (int): _description_

        Returns:
            np.ndarray: Fingerprint.
        """
        pass

    @abc.abstractmethod
    def has_substructure(self, target: Molecule) -> bool:
        """Determine if molecule contains the target substructure.

        Args:
            target (Molecule): Target substructure.

        Returns:
            bool: If present in molecule.
        """
        pass

    @abc.abstractproperty
    def sha1_hash(self):
        """Calculate SHA-1 hash from molecule.

        Returns:
            str: SHA-1 hash.
        """
        pass

    @abc.abstractmethod
    def to_bytes(self) -> bytes:
        """Generate binary string representation from molecule.

        Returns:
            bytes: Binary string representation.
        """
        pass

    @abc.abstractmethod
    def to_smiles(self, canonical: bool = False) -> str:
        """Generate SMILES string from molecule.

        Args:
            canonical (bool, optional): Determine canonical form. Defaults to False.

        Returns:
            str: SMILES string.
        """
        pass
