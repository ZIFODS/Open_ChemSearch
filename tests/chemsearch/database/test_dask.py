import dask.dataframe as dd
import numpy as np
import pandas as pd
import pandera as pa
import pytest
from dask.distributed import Client, LocalCluster
from pandera.typing import Series
from pyprojroot import here

from chemsearch.constants import Cartridges
from chemsearch.database.dask import DaskDatabase, calculate_divisions, get_thread_count
from chemsearch.molecule.factory import MoleculeFactory


class MoleculesSchema(pa.SchemaModel):
    SMILES: Series[str]
    Molecule: Series[pa.Object]


@pytest.fixture
def data_dir():
    return here() / "tests" / "TestData"


class TestDaskDatabase:
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
    def dask_dataframe(self, dataframe):
        return dd.from_pandas(dataframe, npartitions=2)

    @pytest.fixture
    def database(self, dask_dataframe):
        return DaskDatabase(dask_dataframe)

    @pytest.fixture(scope="class")
    def client(self):
        with LocalCluster(n_workers=2, threads_per_worker=1) as local_cluster, Client(
            local_cluster
        ) as client:
            yield client

    @pytest.fixture
    def boolean_array(self):
        return np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype="bool")

    def test_get_substructure_hits_completes(self, client, query, database):
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
        self, client, factory, query, database, expected
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
        self, client, factory, database, mask, expected
    ):
        query = factory.from_smiles("COC")

        actual = database.get_substructure_hits(query, mask=mask)

        assert actual == expected

    def test_from_sdf_file_completes(self, client, sdf_file, factory):
        DaskDatabase.from_sdf_file(sdf_file, factory)

    def test_from_sdf_file_sets_expected_dataframe(self, client, sdf_file, factory):
        actual = DaskDatabase.from_sdf_file(sdf_file, factory)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_sdf_file_sets_expected_dataframe_given_persist(
        self, client, sdf_file, factory, persist
    ):
        actual = DaskDatabase.from_sdf_file(sdf_file, factory, persist=persist)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("partitions_per_thread, expected", [(2, 4), (4, 8)])
    def test_from_sdf_file_sets_dataframe_with_expected_partitions(
        self, client, sdf_file, factory, partitions_per_thread, expected
    ):
        actual = DaskDatabase.from_sdf_file(
            sdf_file, factory, partitions_per_thread=partitions_per_thread
        )

        assert actual._ddf.npartitions == expected

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_sdf_file_sets_expected_molecules(
        self, client, sdf_file, factory, persist, molecules
    ):
        actual = DaskDatabase.from_sdf_file(sdf_file, factory, persist=persist)

        assert actual.molecules == molecules

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_sdf_file_sets_expected_smiles(
        self, client, sdf_file, factory, persist, smiles
    ):
        actual = DaskDatabase.from_sdf_file(sdf_file, factory, persist=persist)

        assert actual.smiles == smiles

    def test_from_sdf_file_raises_exception_given_no_file(self, client, factory):
        with pytest.raises(FileNotFoundError):
            DaskDatabase.from_sdf_file("unrecognised", factory)

    def test_from_smi_file_completes(self, client, smi_file, factory):
        DaskDatabase.from_smi_file(smi_file, factory)

    def test_from_smi_file_sets_expected_dataframe(self, client, smi_file, factory):
        actual = DaskDatabase.from_smi_file(smi_file, factory)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_smi_file_sets_expected_dataframe_given_persist(
        self, client, smi_file, factory, persist
    ):
        actual = DaskDatabase.from_smi_file(smi_file, factory, persist=persist)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("partitions_per_thread, expected", [(2, 4), (4, 8)])
    def test_from_smi_file_sets_dataframe_with_expected_partitions(
        self, client, smi_file, factory, partitions_per_thread, expected
    ):
        actual = DaskDatabase.from_smi_file(
            smi_file, factory, partitions_per_thread=partitions_per_thread
        )

        assert actual._ddf.npartitions == expected

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_smi_file_sets_expected_molecules(
        self, client, smi_file, factory, persist, molecules
    ):
        actual = DaskDatabase.from_smi_file(smi_file, factory, persist=persist)

        assert actual.molecules == molecules

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_smi_file_sets_expected_smiles(
        self, client, smi_file, factory, persist, smiles
    ):
        actual = DaskDatabase.from_smi_file(smi_file, factory, persist=persist)

        assert actual.smiles == smiles

    def test_from_smi_file_raises_exception_given_no_file(self, client, factory):
        with pytest.raises(FileNotFoundError):
            DaskDatabase.from_smi_file("unrecognised", factory)

    def test_from_parquet_completes(self, client, parquet_file, factory):
        DaskDatabase.from_parquet(parquet_file, factory)

    def test_from_parquet_sets_expected_dataframe(self, client, parquet_file, factory):
        actual = DaskDatabase.from_parquet(parquet_file, factory)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_parquet_sets_expected_dataframe_given_persist(
        self, client, parquet_file, factory, persist
    ):
        actual = DaskDatabase.from_parquet(parquet_file, factory, persist=persist)

        MoleculesSchema.validate(actual._ddf)

        assert actual._ddf.npartitions == 2
        assert actual._ddf.known_divisions

    @pytest.mark.parametrize("partitions_per_thread, expected", [(2, 4), (4, 8)])
    def test_from_parquet_sets_dataframe_with_expected_partitions(
        self, client, parquet_file, factory, partitions_per_thread, expected
    ):
        actual = DaskDatabase.from_parquet(
            parquet_file, factory, partitions_per_thread=partitions_per_thread
        )

        assert actual._ddf.npartitions == expected

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_parquet_sets_expected_molecules(
        self, client, parquet_file, factory, persist, molecules
    ):
        actual = DaskDatabase.from_parquet(parquet_file, factory, persist=persist)

        assert actual.molecules == molecules

    @pytest.mark.parametrize("persist", [True, False])
    def test_from_parquet_sets_expected_smiles(
        self, client, parquet_file, factory, persist, smiles
    ):
        actual = DaskDatabase.from_parquet(parquet_file, factory, persist=persist)

        assert actual.smiles == smiles

    def test_from_parquet_raises_exception_given_no_file(self, factory):
        with pytest.raises(FileNotFoundError):
            DaskDatabase.from_parquet("unrecognised", factory)


class TestCalculateDivisions:
    @pytest.fixture
    def parquet_file(self, data_dir):
        return data_dir / "sample.parquet"

    @pytest.fixture
    def dataframe(self, parquet_file):
        return dd.read_parquet(parquet_file)

    def test_completes(self, dataframe):
        calculate_divisions(dataframe)

    @pytest.mark.parametrize(
        "npartitions, expected",
        [(1, (0, 10)), (2, (0, 5, 10)), (3, (0, 3, 6, 10)), (5, (0, 2, 4, 6, 8, 10))],
    )
    def test_returns_expected_indices(self, dataframe, npartitions, expected):
        dataframe = dataframe.repartition(npartitions=npartitions)

        actual = calculate_divisions(dataframe)

        assert actual == expected


class TestGetThreadCount:
    @pytest.fixture
    def client(self, n_workers, threads_per_worker):
        with LocalCluster(
            n_workers=n_workers, threads_per_worker=threads_per_worker
        ) as local_cluster, Client(local_cluster) as client:
            yield client

    @pytest.mark.parametrize(
        "n_workers, threads_per_worker, expected", [(2, 1, 2), (2, 2, 4)]
    )
    def test_returns_expected_count(self, client, expected):
        actual = get_thread_count()

        assert actual == expected

    def test_raises_exception_given_no_client(self):
        with pytest.raises(ValueError):
            get_thread_count()
