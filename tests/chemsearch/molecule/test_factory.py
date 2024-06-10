import pytest
from pyprojroot import here

from chemsearch.constants import Cartridges
from chemsearch.molecule.factory import MoleculeFactory
from chemsearch.molecule.rdkit import RDKitMolecule


class TestMoleculeFactory:
    @pytest.fixture
    def factory(self, cartridge):
        return MoleculeFactory(cartridge)

    @pytest.fixture
    def mol_block(self):
        filepath = here() / "tests" / "TestData" / "sample.mol"

        with open(filepath) as fh:
            text = fh.read()

        return text

    @pytest.mark.parametrize(
        "cartridge, expected",
        [(Cartridges.RDKit, RDKitMolecule)],
    )
    def test_sets_correct_class(self, cartridge, expected):
        factory = MoleculeFactory(cartridge)

        actual = factory._molecule_class

        assert actual == expected

    def test_raises_exception_given_unrecognised_cartridge(self):
        with pytest.raises(ValueError):
            MoleculeFactory("unrecognised")

    @pytest.mark.parametrize(
        "cartridge, expected",
        [(Cartridges.RDKit, RDKitMolecule)],
    )
    def test_from_smiles_returns_expected_molecule_type(self, factory, expected):
        actual = factory.from_smiles(
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1"
        )

        assert isinstance(actual, expected)

    @pytest.mark.parametrize("cartridge, expected", [(Cartridges.RDKit, RDKitMolecule)])
    def test_from_smarts_returns_expected_molecule_type(self, factory, expected):
        actual = factory.from_smarts("[#0]-[#7]1-[#6]-[#6]-[#7]-[#6]-[#6]-1")

        assert isinstance(actual, expected)

    @pytest.mark.parametrize(
        "cartridge, expected",
        [(Cartridges.RDKit, RDKitMolecule)],
    )
    def test_from_mol_block_returns_expected_molecule_type(
        self, factory, mol_block, expected
    ):
        actual = factory.from_mol_block(mol_block)

        assert isinstance(actual, expected)
