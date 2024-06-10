import re

import numpy as np
import pytest
from pyprojroot import here
from rdkit import Chem

from chemsearch.exceptions import MoleculeParsingException
from chemsearch.molecule.rdkit import RDKitMolecule


class TestRDKitMolecule:
    @pytest.fixture
    def mol_block(self):
        filepath = here() / "tests" / "TestData" / "sample.mol"

        with open(filepath) as fh:
            text = fh.read()

        return text

    @pytest.fixture
    def smarts(self):
        return "[#0]-[#7]1-[#6]-[#6]-[#7]-[#6]-[#6]-1"

    @pytest.fixture
    def smiles(self):
        return "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1"

    @pytest.fixture
    def molecule(self, smiles):
        return RDKitMolecule.from_smiles(smiles)

    def test_from_mol_block_completes(self, mol_block):
        RDKitMolecule.from_mol_block(mol_block)

    def test_from_mol_block_returns_molecule(self, mol_block):
        actual = RDKitMolecule.from_mol_block(mol_block)

        assert isinstance(actual, RDKitMolecule)

    def test_from_mol_block_sets_molecule_attribute(self, mol_block):
        molecule = RDKitMolecule.from_mol_block(mol_block)

        actual = molecule._molecule
        assert isinstance(actual, Chem.Mol)

    def test_from_mol_block_sets_expected_molecule(self, mol_block):
        molecule = RDKitMolecule.from_mol_block(mol_block)

        actual = molecule._molecule
        assert actual.GetNumAtoms() == 32

    @pytest.mark.parametrize("mol_block", ["unrecognised", ""])
    def test_from_mol_block_raises_exception_given_invalid_text(self, mol_block):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_mol_block(mol_block)

    def test_from_mol_block_raises_exception_given_none(self):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_mol_block(None)

    def test_from_smarts_completes(self, smarts):
        RDKitMolecule.from_smarts(smarts)

    def test_from_smarts_returns_molecule(self, smarts):
        actual = RDKitMolecule.from_smarts(smarts)

        assert isinstance(actual, RDKitMolecule)

    def test_from_smarts_sets_molecule_attribute(self, smarts):
        molecule = RDKitMolecule.from_smarts(smarts)

        actual = molecule._molecule
        assert isinstance(actual, Chem.Mol)

    def test_from_smarts_sets_expected_molecule(self, smarts):
        molecule = RDKitMolecule.from_smarts(smarts)

        actual = molecule._molecule
        assert actual.GetNumAtoms() == 7

    @pytest.mark.parametrize("smarts", ["unrecognised", ""])
    def test_from_smarts_raises_exception_given_invalid_text(self, smarts):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_smarts(smarts)

    def test_from_smarts_raises_exception_given_none(self):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_smarts(None)

    def test_from_smiles_completes(self, smiles):
        RDKitMolecule.from_smiles(smiles)

    def test_from_smiles_returns_molecule(self, smiles):
        actual = RDKitMolecule.from_smiles(smiles)

        assert isinstance(actual, RDKitMolecule)

    def test_from_smiles_sets_molecule_attribute(self, smiles):
        molecule = RDKitMolecule.from_smiles(smiles)

        actual = molecule._molecule
        assert isinstance(actual, Chem.Mol)

    def test_from_smiles_sets_expected_molecule(self, smiles):
        molecule = RDKitMolecule.from_smiles(smiles)

        actual = molecule._molecule
        assert actual.GetNumAtoms() == 32

    @pytest.mark.parametrize("smiles", ["unrecognised", ""])
    def test_from_smiles_raises_exception_given_invalid_smiles(self, smiles):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_smiles(smiles)

    def test_from_smiles_raises_exception_given_none(self):
        with pytest.raises(MoleculeParsingException):
            RDKitMolecule.from_smiles(None)

    def test_get_substructure_fingerprint_completes(self, molecule):
        molecule.get_substructure_fingerprint()

    def test_get_substructure_fingerprint_returns_vector_with_expected_type(
        self, molecule
    ):
        actual = molecule.get_substructure_fingerprint()

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype("uint64")

    @pytest.mark.parametrize("length, expected", [(1024, 16), (1025, 17), (2048, 32)])
    def test_get_substructure_fingerprint_returns_vector_with_expected_dimensions(
        self, molecule, length, expected
    ):
        actual = molecule.get_substructure_fingerprint(length)

        assert actual.shape == (expected,)

    def test_has_substructure_completes(self, molecule):
        query_molecule = RDKitMolecule.from_smiles("c1ccccc1")

        molecule.has_substructure(query_molecule)

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("c1ccccc1", True),
            ("c1cccnc1", False),
            ("C1NCCCC1", False),
            ("C1NCCNC1", True),
            ("C=O", False),
            ("CO", True),
            ("COC", True),
            ("S", False),
            ("Cl", True),
        ],
    )
    def test_has_substructure_returns_expected_match(self, molecule, query, expected):
        query_molecule = RDKitMolecule.from_smiles(query)

        actual = molecule.has_substructure(query_molecule)

        assert actual == expected

    def test_sha1_hash_returns_expected_pattern_given_smiles(self, smiles):
        molecule = RDKitMolecule.from_smiles(smiles)
        actual = molecule.sha1_hash

        assert re.match("^[a-f0-9]{40}$", actual)

    def test_sha1_hash_returns_expected_pattern_given_smarts(self, smarts):
        molecule = RDKitMolecule.from_smarts(smarts)
        actual = molecule.sha1_hash

        assert re.match("^[a-f0-9]{40}$", actual)

    def test_sha1_hash_returns_same_pattern_given_same_smiles(self, smiles):
        molecule_1 = RDKitMolecule.from_smiles(smiles)
        molecule_2 = RDKitMolecule.from_smiles(smiles)

        assert molecule_1.sha1_hash == molecule_2.sha1_hash

    def test_sha1_hash_returns_same_pattern_given_same_smarts(self, smarts):
        molecule_1 = RDKitMolecule.from_smarts(smarts)
        molecule_2 = RDKitMolecule.from_smarts(smarts)

        assert molecule_1.sha1_hash == molecule_2.sha1_hash
