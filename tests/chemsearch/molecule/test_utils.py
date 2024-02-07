import numpy as np
import pytest
from numpy.testing import assert_array_equal

from chemsearch.molecule import utils
from chemsearch.molecule.rdkit import RDKitMolecule


class TestCompareByCanonicalSmiles:
    @pytest.fixture(params=[RDKitMolecule])
    def identical_molecules(self, request):
        smiles = [
            r"CCOc1ccccc1C(=O)N/N=C(\C)C(=O)O",
            r"C(C)Oc1ccccc1C(=O)N/N=C(\C)C(=O)O",
        ]
        molecules = [request.param.from_smiles(i) for i in smiles]
        return molecules

    @pytest.fixture(params=[RDKitMolecule])
    def different_molecules(self, request):
        smiles = [
            r"CCOc1ccccc1C(=O)N/N=C(\C)C(=O)O",
            r"O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
        ]
        molecules = [request.param.from_smiles(i) for i in smiles]
        return molecules

    def test_completes(self, identical_molecules):
        utils.compare_by_canonical_smiles(
            identical_molecules[0], identical_molecules[1]
        )

    def test_returns_expected_result_given_identical_molecules(
        self, identical_molecules
    ):
        actual = utils.compare_by_canonical_smiles(
            identical_molecules[0], identical_molecules[1]
        )

        assert actual == True

    def test_returns_expected_result_given_different_molecules(
        self, different_molecules
    ):
        actual = utils.compare_by_canonical_smiles(
            different_molecules[0], different_molecules[1]
        )

        assert actual == False


class TestConvertFingerprintBitIndicesToVector:
    @pytest.fixture
    def indices(self):
        return [0, 8, 11]

    def test_completes(self, indices):
        utils.convert_fingerprint_bit_indices_to_vector(indices, 12)

    def test_returns_vector_with_expected_type(self, indices):
        actual = utils.convert_fingerprint_bit_indices_to_vector(indices, 12)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype("int8")

    def test_returns_expected_fingerprint_given_no_indices(self):
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="int8")

        actual = utils.convert_fingerprint_bit_indices_to_vector([], 12)

        assert_array_equal(actual, expected, strict=True)

    @pytest.mark.parametrize(
        "indices, length, expected",
        [
            (
                [0, 8, 11],
                12,
                np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1], dtype="int8"),
            ),
            ([2, 4], 12, np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype="int8")),
            ([2, 4], 10, np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype="int8")),
        ],
    )
    def test_returns_expected_fingerprint(self, indices, length, expected):
        actual = utils.convert_fingerprint_bit_indices_to_vector(indices, length)

        assert_array_equal(actual, expected, strict=True)

    def test_raises_exception_given_zero_length(self):
        with pytest.raises(ValueError):
            utils.convert_fingerprint_bit_indices_to_vector([], 0)

    def test_raises_exception_given_indices_outside_length(self, indices):
        with pytest.raises(IndexError):
            utils.convert_fingerprint_bit_indices_to_vector(indices, 11)


class TestPackFingerprintTo64BitIntegers:
    @pytest.fixture
    def fingerprint(self):
        return utils.convert_fingerprint_bit_indices_to_vector([3, 16, 43], 64)

    def test_completes(self, fingerprint):
        utils.pack_fingerprint_to_64_bit_integers(fingerprint)

    def test_returns_fingerprint_with_expected_type(self, fingerprint):
        actual = utils.pack_fingerprint_to_64_bit_integers(fingerprint)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype("uint64")

    def test_returns_fingerprint_with_expected_value(self, fingerprint):
        expected = np.array([17592194433040], dtype="uint64")

        actual = utils.pack_fingerprint_to_64_bit_integers(fingerprint)

        assert_array_equal(actual, expected)

    def test_raises_exception_given_invalid_length(self, fingerprint):
        fingerprint = np.pad(fingerprint, (0, 1))

        with pytest.raises(ValueError):
            utils.pack_fingerprint_to_64_bit_integers(fingerprint)


class TestPadFingerprintToMultipleOf64:
    @pytest.fixture
    def fingerprint(self, length):
        return utils.convert_fingerprint_bit_indices_to_vector([3, 16, 43], length)

    @pytest.mark.parametrize("length", [63])
    def test_completes(self, fingerprint):
        utils.pad_fingerprint_to_multiple_of_64(fingerprint)

    @pytest.mark.parametrize("length", [63])
    def test_returns_fingerprint_with_expected_type(self, fingerprint):
        actual = utils.pad_fingerprint_to_multiple_of_64(fingerprint)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype("int8")

    @pytest.mark.parametrize(
        "length, expected", [(63, 64), (64, 64), (65, 128), (128, 128)]
    )
    def test_returns_fingerprint_with_expected_length(self, fingerprint, expected):
        actual = utils.pad_fingerprint_to_multiple_of_64(fingerprint)

        assert len(actual) == expected

    @pytest.mark.parametrize("length", [63, 64, 65, 128])
    def test_returns_fingerprint_with_expected_value(self, fingerprint):
        actual = utils.pad_fingerprint_to_multiple_of_64(fingerprint)

        assert list(np.where(actual)[0]) == [3, 16, 43]
