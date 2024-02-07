from __future__ import annotations

from hashlib import sha1

import numpy as np
from attrs import define
from rdkit import Chem, RDLogger
from rdkit.Chem.rdmolops import PatternFingerprint

from chemsearch.exceptions import MoleculeParsingException

from . import Molecule, utils

RDLogger.DisableLog("rdApp.*")


@define(eq=False)
class RDKitMolecule:
    _molecule: Chem.Molecule

    def __eq__(self, other: Molecule):
        """Compare molecules by their canonical SMILES string.

        Args:
            other (Molecule): Molecule to compare against.
        """
        return utils.compare_by_canonical_smiles(self, other)

    @classmethod
    def from_bytes(cls, array: bytes) -> Molecule:
        """Generate molecule from binary string representation.

        Args:
            array (bytes): Binary string representation.

        Returns:
            Molecule: Molecule.
        """
        molecule = Chem.Mol(array)
        return cls(molecule=molecule)

    @classmethod
    def from_mol_block(cls, text: str) -> Molecule:
        """Generate molecule from contents of MOL file or single SDF record.

        Args:
            text (str): Contents of MOL file or single SDF record.

        Returns:
            Molecule: Molecule.
        """
        try:
            molecule = Chem.MolFromMolBlock(text)

        except TypeError as exc:
            raise MoleculeParsingException(exc)

        if molecule is None:
            raise MoleculeParsingException(text)

        return cls(molecule=molecule)

    @classmethod
    def from_smarts(cls, smarts: str) -> Molecule:
        """Generate molecule from SMARTS string.

        Args:
            smarts (str): SMARTS string.

        Returns:
            Molecule: Molecule.
        """
        if smarts == "":
            raise MoleculeParsingException("Empty string cannot be parsed to molecule.")

        try:
            molecule = Chem.MolFromSmarts(smarts)

        except TypeError as exc:
            raise MoleculeParsingException(exc)

        if molecule is None:
            raise MoleculeParsingException(smarts)

        return cls(molecule=molecule)

    @classmethod
    def from_smiles(cls, smiles: str) -> Molecule:
        """Generate molecule from SMILES string.

        Args:
            smiles (str): SMILES string.

        Returns:
            Molecule: Molecule.
        """
        if smiles == "":
            raise MoleculeParsingException("Empty string cannot be parsed to molecule.")

        try:
            molecule = Chem.MolFromSmiles(smiles)

        except TypeError as exc:
            raise MoleculeParsingException(exc)

        if molecule is None:
            raise MoleculeParsingException(smiles)

        return cls(molecule=molecule)

    def get_substructure_fingerprint(self, length: int = 2048) -> np.ndarray:
        """Generate substructure-compatible fingerprint as dense bit vector.

        Args:
            length (int, optional): Length of bit vector. Defaults to 2048.

        Returns:
            np.ndarray: Fingerprint.
        """
        fp = PatternFingerprint(self._molecule, fpSize=length)

        fp = np.array(fp.ToList(), dtype="int8")

        fp = utils.pad_fingerprint_to_multiple_of_64(fp)
        fp = utils.pack_fingerprint_to_64_bit_integers(fp)

        return fp

    def has_substructure(self, query: RDKitMolecule) -> bool:
        """Determine if molecule contains the query substructure.

        Args:
            query (Molecule): Target substructure.

        Returns:
            bool: If present in molecule.
        """
        result = self._molecule.HasSubstructMatch(query._molecule)

        return result

    @property
    def sha1_hash(self):
        """Calculate SHA-1 hash from molecule.

        Returns:
            str: SHA-1 hash.
        """
        smarts = Chem.MolToSmarts(self._molecule)
        return sha1(smarts.encode()).hexdigest()

    def to_bytes(self) -> bytes:
        """Generate binary string representation from molecule.

        Returns:
            bytes: Binary string representation.
        """
        return self._molecule.ToBinary()

    def to_smiles(self, canonical: bool = False) -> str:
        """Generate SMILES string from molecule.

        Args:
            canonical (bool, optional): Determine canonical form. Defaults to False.

        Returns:
            str: SMILES string.
        """
        smiles = Chem.MolToSmiles(self._molecule, canonical=canonical)

        return smiles
