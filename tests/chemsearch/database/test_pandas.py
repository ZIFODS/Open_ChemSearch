import numpy as np
import pandas as pd
import pandera as pa
import pytest
from pandera.typing import Series
from pyprojroot import here

from chemsearch.constants import Cartridges
from chemsearch.database.pandas import PandasDatabase
from chemsearch.molecule.factory import MoleculeFactory


class MoleculesSchema(pa.SchemaModel):
    SMILES: Series[str]
    Molecule: Series[pa.Object]


class TestPandasDatabase:
    @pytest.fixture
    def data_dir(self):
        return here() / "tests" / "TestData"

    @pytest.fixture
    def filepath(self, data_dir, filename):
        return data_dir / filename

    @pytest.fixture
    def sdf_file(self, data_dir):
        return data_dir / "sample.sdf"

    @pytest.fixture
    def smi_file(self, data_dir):
        return data_dir / "sample.smi"

    @pytest.fixture
    def parquet_file(self, data_dir):
        return data_dir / "sample.parquet"

    @pytest.fixture
    def smiles(self):
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
    def factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.fixture
    def query(self, factory):
        return factory.from_smiles("C1NCCNC1")

    @pytest.fixture
    def molecules(self, smiles, factory):
        return [factory.from_smiles(i) for i in smiles]

    @pytest.fixture
    def dataframe(self, smiles, molecules):
        return pd.DataFrame({"SMILES": smiles, "Molecule": molecules})

    @pytest.fixture
    def database(self, dataframe):
        return PandasDatabase(dataframe)

    @pytest.fixture
    def boolean_array(self):
        return np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype="bool")

    def test_get_substructure_hits_completes(self, query, database):
        database.get_substructure_hits(query)

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
    def test_get_substructure_hits_returns_expected_hits(
        self, factory, query, database, expected
    ):
        query = factory.from_smiles(query)

        actual = database.get_substructure_hits(query)

        assert actual == expected

    @pytest.mark.parametrize(
        "mask, expected",
        [
            (
                np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype="bool"),
                [],
            ),
            (
                np.array([0, 1, 1, 1, 1, 0, 1, 1, 0, 1], dtype="bool"),
                [
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                ],
            ),
            (
                np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype="bool"),
                [
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
            (
                np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype="bool"),
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                ],
            ),
            (
                np.array([1, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype="bool"),
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
            (
                np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype="bool"),
                [
                    "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
                    "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
                    "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
                    "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
                    "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
                    "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
                ],
            ),
        ],
    )
    def test_get_substructure_hits_returns_expected_hits_given_mask(
        self, factory, database, mask, expected
    ):
        query = factory.from_smiles("COC")

        actual = database.get_substructure_hits(query, mask=mask)

        assert actual == expected

    def test_from_sdf_file_completes(self, sdf_file, factory):
        PandasDatabase.from_sdf_file(sdf_file, factory)

    def test_from_sdf_file_sets_expected_dataframe(self, sdf_file, factory):
        actual = PandasDatabase.from_sdf_file(sdf_file, factory)

        MoleculesSchema.validate(actual._df)

    def test_from_sdf_file_sets_expected_molecules(self, sdf_file, factory, molecules):
        actual = PandasDatabase.from_sdf_file(sdf_file, factory)

        assert actual.molecules == molecules

    def test_from_sdf_file_sets_expected_smiles(self, sdf_file, factory, smiles):
        actual = PandasDatabase.from_sdf_file(sdf_file, factory)

        assert actual.smiles == smiles

    def test_from_sdf_file_raises_exception_given_no_file(self, factory):
        with pytest.raises(FileNotFoundError):
            PandasDatabase.from_sdf_file("unrecognised", factory)

    def test_from_smi_file_completes(self, smi_file, factory):
        PandasDatabase.from_smi_file(smi_file, factory)

    def test_from_smi_file_sets_expected_dataframe(self, smi_file, factory):
        actual = PandasDatabase.from_smi_file(smi_file, factory)

        MoleculesSchema.validate(actual._df)

    def test_from_smi_file_sets_expected_molecules(self, smi_file, factory, molecules):
        actual = PandasDatabase.from_smi_file(smi_file, factory)

        assert actual.molecules == molecules

    def test_from_smi_file_sets_expected_smiles(self, smi_file, factory, smiles):
        actual = PandasDatabase.from_smi_file(smi_file, factory)

        assert actual.smiles == smiles

    def test_from_smi_file_raises_exception_given_no_file(self, factory):
        with pytest.raises(FileNotFoundError):
            PandasDatabase.from_smi_file("unrecognised", factory)

    def test_from_parquet_completes(self, parquet_file, factory):
        PandasDatabase.from_parquet(parquet_file, factory)

    def test_from_parquet_sets_expected_dataframe(self, parquet_file, factory):
        actual = PandasDatabase.from_parquet(parquet_file, factory)

        MoleculesSchema.validate(actual._df)

    def test_from_parquet_returns_database_with_expected_molecules(
        self, parquet_file, factory, molecules
    ):
        actual = PandasDatabase.from_parquet(parquet_file, factory)

        assert actual.molecules == molecules

    def test_from_parquet_returns_database_with_expected_smiles(
        self, parquet_file, factory, smiles
    ):
        actual = PandasDatabase.from_parquet(parquet_file, factory)

        assert actual.smiles == smiles

    def test_from_parquet_raises_exception_given_no_file(self, factory):
        with pytest.raises(FileNotFoundError):
            PandasDatabase.from_parquet("unrecognised", factory)
