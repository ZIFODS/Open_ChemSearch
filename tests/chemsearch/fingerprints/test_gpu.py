import cupy as cp
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pyprojroot import here

from chemsearch import io
from chemsearch.constants import Cartridges
from chemsearch.fingerprints.gpu import GPUFingerprints
from chemsearch.molecule import MoleculeFactory


class TestGPUFingerprints:
    @pytest.fixture
    def data_dir(self):
        return here() / "tests" / "TestData"

    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample_fingerprints.npy"

    @pytest.fixture
    def fingerprints_full(self, filepath):
        return GPUFingerprints.from_file(filepath)

    @pytest.fixture
    def fingerprints(self):
        return GPUFingerprints(
            cp.array(
                [
                    [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0],
                    [0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                ],
                dtype="int8",
            )
        )

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
    def factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.fixture
    def molecules(self, smiles, factory):
        return [factory.from_smiles(i) for i in smiles]

    @pytest.fixture
    def query(self):
        return np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="int8")

    def test_check_compatible_completes_given_compatible_data(
        self, fingerprints, molecules
    ):
        fingerprints.check_compatible(molecules)

    def test_check_compatible_raises_exception_given_different_length_data(
        self, fingerprints, molecules
    ):
        with pytest.raises(ValueError) as exc:
            fingerprints.check_compatible(molecules[:3])

        assert "molecules (3) and fingerprints (10)" in str(exc.value)

    def test_completes(self, filepath):
        GPUFingerprints.from_file(filepath)

    def test_returns_matrix(self, filepath):
        actual = GPUFingerprints.from_file(filepath)

        assert isinstance(actual.matrix, cp.ndarray)

    def test_returns_matrix_with_expected_type(self, filepath):
        actual = GPUFingerprints.from_file(filepath)

        assert isinstance(actual.matrix, cp.ndarray)
        assert actual.matrix.dtype == np.dtype("uint64")

    def test_returns_matrix_with_expected_dimensions(self, filepath):
        actual = GPUFingerprints.from_file(filepath)

        assert actual.matrix.shape == (10, 32)

    def test_raises_exception_given_no_file(self):
        with pytest.raises(FileNotFoundError):
            GPUFingerprints.from_file("unknown.npy")

    def test_from_mol_blocks_completes(self, mol_blocks, factory):
        GPUFingerprints.from_mol_blocks(mol_blocks, 2048, factory)

    def test_from_mol_blocks_returns_matrix_with_expected_type(
        self, mol_blocks, factory
    ):
        actual = GPUFingerprints.from_mol_blocks(mol_blocks, 2048, factory)

        assert isinstance(actual.matrix, cp.ndarray)
        assert actual.matrix.dtype == np.dtype("uint64")

    @pytest.mark.parametrize("length, expected", [(2048, 32), (4096, 64)])
    def test_from_mol_blocks_returns_matrix_with_expected_dimensions(
        self, mol_blocks, length, factory, expected
    ):
        actual = GPUFingerprints.from_mol_blocks(mol_blocks, length, factory)

        assert actual.matrix.shape == (10, expected)

    def test_from_mol_blocks_returns_matrix_with_expected_dimensions_given_one_molecule(
        self, mol_blocks, factory
    ):
        actual = GPUFingerprints.from_mol_blocks(mol_blocks[:1], 2048, factory)

        assert actual.matrix.shape == (1, 32)

    @pytest.mark.parametrize("chunk_size", [1, 3, 5, 10])
    def test_from_mol_blocks_returns_matrix_with_expected_values(
        self, mol_blocks, factory, chunk_size, fingerprints_full
    ):
        actual = GPUFingerprints.from_mol_blocks(
            mol_blocks, 2048, factory, chunk_size=chunk_size
        )

        cp.testing.assert_array_equal(actual.matrix, fingerprints_full.matrix)

    def test_from_mol_blocks_raises_exception_given_no_mol_blocks(self, factory):
        with pytest.raises(ValueError):
            GPUFingerprints.from_mol_blocks([], 2048, factory)

    def test_from_molecules_completes(self, molecules):
        GPUFingerprints.from_molecules(molecules, 2048)

    def test_from_molecules_returns_matrix_with_expected_type(self, molecules):
        actual = GPUFingerprints.from_molecules(molecules, 2048)

        assert isinstance(actual.matrix, cp.ndarray)
        assert actual.matrix.dtype == np.dtype("uint64")

    @pytest.mark.parametrize("length, expected", [(2048, 32), (4096, 64)])
    def test_from_molecules_returns_matrix_with_expected_dimensions(
        self, molecules, length, expected
    ):
        actual = GPUFingerprints.from_molecules(molecules, length)

        assert actual.matrix.shape == (10, expected)

    def test_from_molecules_returns_matrix_with_expected_dimensions_given_one_molecule(
        self, molecules
    ):
        actual = GPUFingerprints.from_molecules(molecules[:1], 2048)

        assert actual.matrix.shape == (1, 32)

    def test_from_molecules_returns_matrix_with_expected_values(
        self, molecules, fingerprints_full
    ):
        actual = GPUFingerprints.from_molecules(molecules, 2048)

        cp.testing.assert_array_equal(actual.matrix, fingerprints_full.matrix)

    def test_from_molecules_raises_exception_given_no_molecules(self):
        with pytest.raises(ValueError):
            GPUFingerprints.from_molecules([], 2048)

    def test_from_smiles_completes(self, smiles, factory):
        GPUFingerprints.from_smiles(smiles, 2048, factory)

    def test_from_smiles_returns_matrix_with_expected_type(self, smiles, factory):
        actual = GPUFingerprints.from_smiles(smiles, 2048, factory)

        assert isinstance(actual.matrix, cp.ndarray)
        assert actual.matrix.dtype == np.dtype("uint64")

    @pytest.mark.parametrize("length, expected", [(2048, 32), (4096, 64)])
    def test_from_smiles_returns_matrix_with_expected_dimensions(
        self, smiles, length, factory, expected
    ):
        actual = GPUFingerprints.from_smiles(smiles, length, factory)

        assert actual.matrix.shape == (10, expected)

    def test_from_smiles_returns_matrix_with_expected_dimensions_given_one_molecule(
        self, smiles, factory
    ):
        actual = GPUFingerprints.from_smiles(smiles[:1], 2048, factory)

        assert actual.matrix.shape == (1, 32)

    @pytest.mark.parametrize("chunk_size", [1, 3, 5, 10])
    def test_from_smiles_returns_matrix_with_expected_values(
        self, smiles, factory, chunk_size, fingerprints_full
    ):
        actual = GPUFingerprints.from_smiles(
            smiles, 2048, factory, chunk_size=chunk_size
        )

        cp.testing.assert_array_equal(actual.matrix, fingerprints_full.matrix)

    def test_from_smiles_raises_exception_given_no_smiles(self, factory):
        with pytest.raises(ValueError):
            GPUFingerprints.from_smiles([], 2048, factory)

    def test_screen_completes(self, fingerprints, query):
        fingerprints.screen(query)

    def test_screen_returns_boolean_array(self, fingerprints, query):
        actual = fingerprints.screen(query)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype(bool)

    def test_screen_returns_expected_hits_given_empty_query(self, fingerprints):
        query = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="int8")
        expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="bool")

        actual = fingerprints.screen(query)

        assert_array_equal(actual, expected, strict=True)

    def test_screen_returns_expected_hits_given_full_query(self, fingerprints):
        query = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="int8")
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="bool")

        actual = fingerprints.screen(query)

        assert_array_equal(actual, expected, strict=True)

    @pytest.mark.parametrize(
        "query, expected",
        [
            (
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="int8"),
                np.array([1, 0, 1, 1, 0, 0, 0, 0, 1, 0], dtype="bool"),
            ),
            (
                np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="int8"),
                np.array([0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype="bool"),
            ),
            (
                np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0], dtype="int8"),
                np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype="bool"),
            ),
        ],
    )
    def test_screen_returns_expected_hits(self, fingerprints, query, expected):
        actual = fingerprints.screen(query)

        assert_array_equal(actual, expected, strict=True)

    @pytest.mark.parametrize(
        "query",
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="int8"),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype="int8"),
        ],
    )
    def test_screen_raises_exception_given_different_fingerprint_lengths(
        self, fingerprints, query
    ):
        with pytest.raises(ValueError):
            fingerprints.screen(query)
