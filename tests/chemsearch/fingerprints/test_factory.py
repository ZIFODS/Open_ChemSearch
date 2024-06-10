import pytest
from pyprojroot import here

from chemsearch import io
from chemsearch.constants import Cartridges, FingerprintMethods
from chemsearch.fingerprints.cpu import CPUFingerprints
from chemsearch.fingerprints.factory import FingerprintsFactory
from chemsearch.fingerprints.gpu import GPUFingerprints
from chemsearch.molecule import MoleculeFactory


class TestFingerprintsFactory:
    @pytest.fixture
    def factory(self, method):
        return FingerprintsFactory(method)

    @pytest.fixture
    def data_dir(self):
        return here() / "tests" / "TestData"

    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample_fingerprints.npy"

    @pytest.fixture
    def mol_blocks(self, data_dir):
        return io.read_sdf_file(data_dir / "sample.sdf")

    @pytest.fixture
    def smiles(self):
        return [
            "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
            "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "CCOc1ccccc1C(=O)N/N=C(\\\\C)C(=O)O",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
            "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ]

    @pytest.fixture
    def molecules_factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.fixture
    def molecules(self, smiles, molecules_factory):
        return [molecules_factory.from_smiles(i) for i in smiles]

    @pytest.mark.parametrize(
        "method, expected",
        [
            (FingerprintMethods.CPU, CPUFingerprints),
            (FingerprintMethods.GPU, GPUFingerprints),
        ],
    )
    def test_sets_correct_class(self, method, expected):
        factory = FingerprintsFactory(method)

        actual = factory._fingerprints_class

        assert actual == expected

    def test_raises_exception_given_unrecognised_method(self):
        with pytest.raises(ValueError):
            FingerprintsFactory("unrecognised")

    @pytest.mark.parametrize(
        "method, expected",
        [
            (FingerprintMethods.CPU, CPUFingerprints),
            (FingerprintMethods.GPU, GPUFingerprints),
        ],
    )
    def test_from_file_returns_expected_fingerprints_type(
        self, factory, filepath, expected
    ):
        actual = factory.from_file(filepath)

        assert isinstance(actual, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            (FingerprintMethods.CPU, CPUFingerprints),
            (FingerprintMethods.GPU, GPUFingerprints),
        ],
    )
    def test_from_mol_blocks_returns_expected_fingerprints_type(
        self, factory, mol_blocks, molecules_factory, expected
    ):
        actual = factory.from_mol_blocks(mol_blocks, 2048, molecules_factory)

        assert isinstance(actual, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            (FingerprintMethods.CPU, CPUFingerprints),
            (FingerprintMethods.GPU, GPUFingerprints),
        ],
    )
    def test_from_molecules_returns_expected_fingerprints_type(
        self, factory, molecules, expected
    ):
        actual = factory.from_molecules(molecules, 2048)

        assert isinstance(actual, expected)

    @pytest.mark.parametrize(
        "method, expected",
        [
            (FingerprintMethods.CPU, CPUFingerprints),
            (FingerprintMethods.GPU, GPUFingerprints),
        ],
    )
    def test_from_smiles_returns_expected_fingerprints_type(
        self, factory, smiles, molecules_factory, expected
    ):
        actual = factory.from_smiles(smiles, 2048, molecules_factory)

        assert isinstance(actual, expected)
