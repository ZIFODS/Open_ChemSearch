from chemsearch.constants import Cartridges

from . import Molecule
from .rdkit import RDKitMolecule


class MoleculeFactory:
    def __init__(self, cartridge: Cartridges):
        match cartridge:
            case Cartridges.RDKit:
                molecule_class = RDKitMolecule

            case _:
                raise ValueError(
                    "Unrecognised chemistry cartridge. Only RDKit is supported."
                )

        self._molecule_class = molecule_class

    def from_bytes(self, array: bytes) -> Molecule:
        """Generate molecule from binary string representation.

        Args:
            array (bytes): Binary string representation of molecule.

        Returns:
            Molecule: Molecule.
        """
        molecule = self._molecule_class.from_bytes(array)

        return molecule

    def from_smarts(self, smarts: str) -> Molecule:
        """Generate molecule from SMARTS string.

        Args:
            smarts (str): SMARTS string.

        Returns:
            Molecule: Molecule.
        """
        molecule = self._molecule_class.from_smarts(smarts)

        return molecule

    def from_smiles(self, smiles: str) -> Molecule:
        """Generate molecule from SMILES string.

        Args:
            smiles (str): SMILES string.

        Returns:
            Molecule: Molecule.
        """
        molecule = self._molecule_class.from_smiles(smiles)

        return molecule

    def from_mol_block(self, text: str) -> Molecule:
        """Generate molecule from contents of MOL file or single SDF record.

        Args:
            text (str): Contents of MOL file or single SDF record.

        Returns:
            Molecule: Molecule.
        """
        molecule = self._molecule_class.from_mol_block(text)

        return molecule
