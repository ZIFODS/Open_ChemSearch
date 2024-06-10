import pytest
from pyprojroot import here

from chemsearch.constants import Cartridges, Databases
from chemsearch.database import DatabaseFactory
from chemsearch.database.dask import DaskDatabase
from chemsearch.database.pandas import PandasDatabase
from chemsearch.molecule import MoleculeFactory


class TestDatabaseFactory:
    @pytest.fixture
    def data_dir(self):
        return here() / "tests" / "TestData"

    @pytest.fixture
    def filepath(self, data_dir, filename):
        return data_dir / filename

    @pytest.fixture
    def factory(self):
        return DatabaseFactory(Databases.PANDAS)

    @pytest.fixture
    def molecule_factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.mark.parametrize(
        "method, expected",
        [
            (Databases.PANDAS, PandasDatabase),
            (Databases.DASK, DaskDatabase),
        ],
    )
    def test_sets_correct_class(self, method, expected):
        factory = DatabaseFactory(method)

        actual = factory._db_class

        assert actual == expected

    def test_raises_exception_given_unrecognised_method(self):
        with pytest.raises(ValueError):
            DatabaseFactory("unrecognised")

    @pytest.mark.parametrize(
        "filename",
        [
            "sample.sdf",
            "sample.smi",
            "sample.parquet",
        ],
    )
    def test_from_file_completes(self, filepath, factory, molecule_factory):
        factory.from_file(filepath, molecule_factory)

    @pytest.mark.parametrize(
        "filename",
        [
            "sample.sdf",
            "sample.smi",
            "sample.parquet",
        ],
    )
    def test_from_file_returns_expected_database(
        self, filepath, factory, molecule_factory
    ):
        actual = factory.from_file(filepath, molecule_factory)

        assert isinstance(actual, PandasDatabase)

    @pytest.mark.parametrize(
        "filename, method",
        [
            ("sample.sdf", "from_sdf_file"),
            ("sample.smi", "from_smi_file"),
            ("sample.parquet", "from_parquet"),
        ],
    )
    def test_from_file_calls_expected_method(
        self, mocker, method, filepath, factory, molecule_factory
    ):
        spy = mocker.spy(PandasDatabase, method)

        factory.from_file(filepath, molecule_factory)

        spy.assert_called()

    @pytest.mark.parametrize(
        "filename, method",
        [
            ("sample.sdf", "from_sdf_file"),
            ("sample.smi", "from_smi_file"),
            ("sample.parquet", "from_parquet"),
        ],
    )
    def test_from_file_calls_method_with_expected_args(
        self, mocker, method, filepath, factory, molecule_factory
    ):
        spy = mocker.spy(PandasDatabase, method)

        factory.from_file(filepath, molecule_factory)

        spy.assert_called_once_with(filepath, molecule_factory)

    @pytest.mark.parametrize(
        "filename, method",
        [
            ("sample.SDF", "from_sdf_file"),
            ("sample.SMI", "from_smi_file"),
            ("sample.PARQUET", "from_parquet"),
        ],
    )
    def test_from_file_calls_method_with_expected_args_given_upper_case_extension(
        self, mocker, method, filepath, factory, molecule_factory
    ):
        spy = mocker.patch.object(PandasDatabase, method)

        factory.from_file(filepath, molecule_factory)

        spy.assert_called_once_with(filepath, molecule_factory)

    @pytest.mark.parametrize("filename", ["sample.txt"])
    def test_from_file_raises_exception_given_invalid_extension(
        self, filepath, factory, molecule_factory
    ):
        with pytest.raises(ValueError):
            factory.from_file(filepath, molecule_factory)
