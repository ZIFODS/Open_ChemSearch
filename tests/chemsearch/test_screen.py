import numpy as np
import pandas as pd
import pytest
from pyprojroot import here

from chemsearch import screen
from chemsearch.constants import Cartridges, FingerprintMethods
from chemsearch.database.pandas import PandasDatabase
from chemsearch.fingerprints import FingerprintsFactory
from chemsearch.molecule import MoleculeFactory


@pytest.fixture
def smiles():
    return [
        "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
        "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
        "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
        "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
        "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
        "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
        "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
        "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
        "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
        "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
    ]


@pytest.fixture
def factory():
    return MoleculeFactory(Cartridges.RDKit)


@pytest.fixture
def molecules(smiles, factory):
    df = pd.DataFrame(
        {"SMILES": smiles, "Molecule": [factory.from_smiles(i) for i in smiles]}
    )
    return PandasDatabase(df)


@pytest.fixture(params=[FingerprintMethods.CPU, FingerprintMethods.GPU])
def fingerprints(request):
    factory = FingerprintsFactory(request.param)
    filepath = here() / "tests" / "TestData" / "sample_fingerprints.npy"
    return factory.from_file(filepath)


class TestExecuteMultistageScreen:
    @pytest.fixture
    def query(self, factory):
        return factory.from_smiles("C1NCCNC1")

    def test_completes(self, query, molecules, fingerprints):
        screen.execute_multistage_screen(query, molecules, fingerprints, 2048)

    @pytest.mark.parametrize(
        "query, expected",
        [
            (
                "C1NCCCC1",
                [
                    "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                "C1NCCNC1",
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                ],
            ),
            (
                "c1ccccc1",
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                "C=O",
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                ],
            ),
            (
                "COC",
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
            (
                "S",
                [
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
        ],
    )
    def test_returns_expected_hits(
        self, factory, query, molecules, fingerprints, expected
    ):
        query = factory.from_smiles(query)

        actual = screen.execute_multistage_screen(query, molecules, fingerprints, 2048)

        assert actual == expected


class TestExecuteDirectGraphScreen:
    @pytest.fixture
    def query(self, factory):
        return factory.from_smiles("c1ccccc1")

    def test_completes(self, query, molecules):
        screen.execute_direct_graph_screen(query, molecules)

    @pytest.mark.parametrize(
        "query, expected",
        [
            (
                "C1NCCCC1",
                [
                    "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                "C1NCCNC1",
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                ],
            ),
            (
                "c1ccccc1",
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                "C=O",
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                ],
            ),
            (
                "COC",
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
            (
                "S",
                [
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
        ],
    )
    def test_returns_expected_hits(self, factory, query, molecules, expected):
        query = factory.from_smiles(query)

        actual = screen.execute_direct_graph_screen(query, molecules)

        assert actual == expected

    @pytest.mark.parametrize(
        "mask, expected",
        [
            (
                np.zeros(10),
                [],
            ),
            (
                np.ones(10, dtype=bool),
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=bool),
                [
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
            (
                np.array([0, 1, 1, 0, 0, 0, 1, 1, 0, 1], dtype=bool),
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
                    "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
                    "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
                ],
            ),
        ],
    )
    def test_returns_expected_hits_given_mask(self, query, molecules, mask, expected):
        actual = screen.execute_direct_graph_screen(query, molecules, mask)

        assert actual == expected

    @pytest.mark.parametrize(
        "mask",
        [
            np.ones(9, dtype=bool),
            np.ones(11, dtype=bool),
        ],
    )
    def test_raises_exception_given_mask_with_incorrect_dimension(
        self, query, molecules, mask
    ):
        with pytest.raises(IndexError):
            screen.execute_direct_graph_screen(query, molecules, mask)


class TestExecuteFingerprintScreen:
    @pytest.fixture
    def query(self, factory):
        return factory.from_smiles("C1NCCNC1")

    def test_completes(self, query, fingerprints):
        screen.execute_fingerprint_screen(query, fingerprints, 2048)

    def test_returns_expected_array(self, query, fingerprints):
        actual = screen.execute_fingerprint_screen(query, fingerprints, 2048)

        assert isinstance(actual, np.ndarray)
        assert actual.dtype == np.dtype("bool_")
        assert actual.shape == (10,)

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("C1NCCCC1", 3),
            ("C1NCCNC1", 2),
            ("c1ccccc1", 8),
            ("C=O", 5),
            ("COC", 6),
            ("S", 2),
        ],
    )
    def test_returns_expected_mask(self, factory, query, fingerprints, expected):
        query = factory.from_smiles(query)

        actual = screen.execute_fingerprint_screen(query, fingerprints, 2048)

        assert expected <= actual.sum() <= 10
