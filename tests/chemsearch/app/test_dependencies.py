from pathlib import Path

import pytest
from pyprojroot import here

from chemsearch.app import dependencies
from chemsearch.app.config import Settings
from chemsearch.constants import Databases
from chemsearch.database.dask import get_thread_count


@pytest.fixture
def tmp_dir(tmpdir):
    return Path(tmpdir)


@pytest.fixture
def test_dir():
    return here() / "tests" / "TestData"


class TestGetDaskClient:
    @pytest.fixture
    def settings(self, tmp_dir, test_dir, processes, threads_per_process):
        return Settings(
            molecules=str(test_dir / "sample.parquet"),
            fingerprints=str(test_dir / "sample_fingerprints.npy"),
            bit_length=2048,
            log_file=tmp_dir / "test.log",
            database="dask",
            processes=processes,
            threads_per_process=threads_per_process,
        )

    @pytest.fixture
    def logger(self, settings):
        yield dependencies.get_logger(settings)

        dependencies.get_logger.cache_clear()

    @pytest.fixture
    def client(self, logger, settings):
        client = dependencies.get_dask_client(logger, settings)
        yield client

        dependencies.get_dask_client.cache_clear()
        client.close()

    @pytest.mark.parametrize("processes, threads_per_process", [(1, 1)])
    def test_completes(self, client):
        _ = next(client)

    @pytest.mark.parametrize("processes, threads_per_process", [(1, 1), (2, 1)])
    def test_yields_client_with_expected_processes(self, processes, client):
        actual = next(client)

        assert len(actual.scheduler_info()["workers"]) == processes

    @pytest.mark.parametrize(
        "processes, threads_per_process, expected", [(1, 1, 1), (2, 1, 2), (1, 4, 4)]
    )
    def test_yields_client_with_expected_threads(self, client, expected):
        _ = next(client)

        assert get_thread_count() == expected

    @pytest.mark.parametrize("processes, threads_per_process", [(1, 1)])
    def test_yields_none_given_pandas(self, logger, settings):
        settings.database = Databases.PANDAS

        client = next(dependencies.get_dask_client(logger, settings))

        assert client is None


class TestGetDaskReportFilepath:
    @pytest.fixture
    def settings(self, tmp_dir, test_dir):
        return Settings(
            molecules=str(test_dir / "sample.parquet"),
            fingerprints=str(test_dir / "sample_fingerprints.npy"),
            bit_length=2048,
            log_file=tmp_dir / "test.log",
        )

    def test_completes(self, settings):
        dependencies.get_dask_report_filepath(settings)

    def test_returns_expected_filepath(self, settings, tmp_dir):
        actual = dependencies.get_dask_report_filepath(settings)

        assert actual == tmp_dir / "test.html"

    def test_returns_expected_filepath_given_file_already_exists(
        self, settings, tmp_dir
    ):
        (tmp_dir / "test.html").touch()

        actual = dependencies.get_dask_report_filepath(settings)

        assert actual == tmp_dir / "test_1.html"

    def test_returns_expected_filepath_given_files_already_exist(
        self, settings, tmp_dir
    ):
        (tmp_dir / "test.html").touch()
        (tmp_dir / "test_1.html").touch()

        actual = dependencies.get_dask_report_filepath(settings)

        assert actual == tmp_dir / "test_2.html"
