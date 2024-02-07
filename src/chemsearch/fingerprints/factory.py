from pathlib import Path

from chemsearch.constants import FingerprintMethods
from chemsearch.molecule import Molecule, MoleculeFactory

from . import Fingerprints
from .cpu import CPUFingerprints
from .gpu import GPUFingerprints


class FingerprintsFactory:
    def __init__(self, method: FingerprintMethods):
        match method:
            case FingerprintMethods.CPU:
                fingerprints_class = CPUFingerprints

            case FingerprintMethods.GPU:
                fingerprints_class = GPUFingerprints

            case _:
                raise ValueError(
                    "Unrecognised fingerprint method. Only CPU and GPU are supported."
                )

        self._fingerprints_class = fingerprints_class

    def from_file(self, filepath: Path) -> Fingerprints:
        """Read fingerprints from numpy binary file.

        Args:
            filepath (Path): Path to file.

        Returns:
            Fingerprints: Fingerprints.
        """
        fingerprints = self._fingerprints_class.from_file(filepath)

        return fingerprints

    def from_mol_blocks(
        self, mol_blocks: list[str], length: int, factory: MoleculeFactory
    ) -> Fingerprints:
        """Calculate fingerprints for molecules.

        Args:
            mol_blocks (list[str]): MOL blocks.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Fingerprints: Fingerprints.
        """
        fingerprints = self._fingerprints_class.from_mol_blocks(
            mol_blocks, length, factory
        )

        return fingerprints

    def from_molecules(self, molecules: list[Molecule], length: int) -> Fingerprints:
        """Calculate fingerprints for molecules.

        Args:
            molecules (list[Molecule]): Molecules.
            length (int): Length of bit vector.

        Returns:
            Fingerprints: Fingerprints.
        """
        fingerprints = self._fingerprints_class.from_molecules(molecules, length)

        return fingerprints

    def from_smiles(
        self, smiles: list[str], length: int, factory: MoleculeFactory
    ) -> Fingerprints:
        """Calculate fingerprints for molecules.

        Args:
            smiles (list[str]): SMILES strings.
            length (int): Length of bit vector.
            factory (MoleculeFactory): Factory to create molecules.

        Returns:
            Fingerprints: Fingerprints.
        """
        fingerprints = self._fingerprints_class.from_smiles(smiles, length, factory)

        return fingerprints
