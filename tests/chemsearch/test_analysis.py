from copy import deepcopy

import pandas as pd
import pandera as pa
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal
from pyprojroot import here

from chemsearch import analysis


@pytest.fixture
def filepath():
    return here() / "tests" / "logs" / "*.log"


@pytest.fixture
def logs(filepath):
    return analysis.read_logs(filepath)


@pytest.fixture
def fp_only_logs(logs):
    return logs.loc[
        logs["event"].isin(
            [
                "Parsed query SMILES string to molecule.",
                "Created query fingerprint.",
                "Completed fingerprint screen.",
                "Received hits from GET request.",
            ]
        )
    ]


@pytest.fixture
def flattened_logs():
    filepath = here() / "tests" / "logs" / "flattened.csv"
    return pd.read_csv(filepath)


@pytest.fixture
def flattened_fp_only_logs(flattened_logs):
    return flattened_logs.drop(columns=["Duration (DG screen, ms)", "Hits (DG screen)"])


class TestAddTotalDuration:
    def test_completes(self, flattened_logs):
        analysis.add_total_duration(flattened_logs)

    def test_completes_given_fp_only_screen(self, flattened_fp_only_logs):
        analysis.add_total_duration(flattened_fp_only_logs)

    def test_returns_dataframe(self, flattened_logs):
        actual = analysis.add_total_duration(flattened_logs)

        assert isinstance(actual, pd.DataFrame)

    def test_does_not_modify_dataframe_in_place(self, flattened_logs):
        original = deepcopy(flattened_logs)

        analysis.add_total_duration(flattened_logs)

        assert_frame_equal(flattened_logs, original)

    def test_adds_expected_column(self, flattened_logs):
        actual = analysis.add_total_duration(flattened_logs)

        assert "Duration (All steps, ms)" in actual.columns

    def test_adds_expected_values(self, flattened_logs):
        expected = pd.Series(
            [
                1547.2564,
                1450.2283,
                1385.1505,
                1211.6526,
                389.7198,
                356.7742,
                116.7043,
                398.0499,
                328.3262,
                1855.8419,
                2312.6985,
                1932.9947,
                2245.5713,
            ],
            name="Duration (All steps, ms)",
        )

        actual = analysis.add_total_duration(flattened_logs)

        assert_series_equal(
            actual["Duration (All steps, ms)"], expected, check_exact=False
        )

    def test_adds_expected_values_given_fp_only_screen(self, flattened_fp_only_logs):
        expected = pd.Series(
            [
                38.1650,
                35.2931,
                12.2431,
                10.6351,
                88.4631,
                77.5778,
                57.0071,
                89.1519,
                47.7632,
                43.8711,
                31.1856,
                141.7143,
                1.2039,
            ],
            name="Duration (All steps, ms)",
        )

        actual = analysis.add_total_duration(flattened_fp_only_logs)

        assert_series_equal(
            actual["Duration (All steps, ms)"], expected, check_exact=False
        )


class TestAddSpecificity:
    def test_completes(self, flattened_logs):
        analysis.add_specificity(flattened_logs)

    def test_returns_dataframe(self, flattened_logs):
        actual = analysis.add_specificity(flattened_logs)

        assert isinstance(actual, pd.DataFrame)

    def test_does_not_modify_dataframe_in_place(self, flattened_logs):
        original = deepcopy(flattened_logs)

        analysis.add_specificity(flattened_logs)

        assert_frame_equal(flattened_logs, original)

    def test_adds_expected_column(self, flattened_logs):
        actual = analysis.add_specificity(flattened_logs)

        assert "Specificity (FP screen)" in actual.columns

    def test_adds_expected_values(self, flattened_logs):
        expected = pd.Series(
            [
                93.007,
                96.971,
                93.007,
                96.971,
                99.494,
                99.167,
                99.173,
                99.494,
                99.167,
                90.907,
                89.607,
                90.907,
                89.607,
            ],
            name="Specificity (FP screen)",
        )

        actual = analysis.add_specificity(flattened_logs)

        assert_series_equal(
            actual["Specificity (FP screen)"], expected, check_exact=False
        )


class TestAddPrecision:
    def test_completes(self, flattened_logs):
        analysis.add_precision(flattened_logs)

    def test_returns_dataframe(self, flattened_logs):
        actual = analysis.add_precision(flattened_logs)

        assert isinstance(actual, pd.DataFrame)

    def test_does_not_modify_dataframe_in_place(self, flattened_logs):
        original = deepcopy(flattened_logs)

        analysis.add_precision(flattened_logs)

        assert_frame_equal(flattened_logs, original)

    def test_adds_expected_column(self, flattened_logs):
        actual = analysis.add_precision(flattened_logs)

        assert "Precision (FP screen)" in actual.columns

    def test_adds_expected_values(self, flattened_logs):
        expected = pd.Series(
            [
                15.051,
                15.563,
                15.051,
                15.563,
                71.009,
                39.572,
                39.065,
                71.009,
                39.572,
                11.991,
                4.982,
                11.991,
                4.982,
            ],
            name="Precision (FP screen)",
        )

        actual = analysis.add_precision(flattened_logs)

        assert_series_equal(actual["Precision (FP screen)"].round(3), expected)


class TestFlattenLogs:
    @pytest.fixture
    def schema(self):
        return pa.DataFrameSchema(
            {
                "File (Molecules)": pa.Column(str),
                "File (Fingerprints)": pa.Column(str),
                "Cartridge": pa.Column(str),
                "Database": pa.Column(str),
                "Fingerprint Method": pa.Column(str),
                "Bit Length": pa.Column(int),
                "Processes": pa.Column(int),
                "Query": pa.Column(str),
                "Duration (Query parsing, ms)": pa.Column(float),
                "Duration (FP generation, ms)": pa.Column(float),
                "Duration (FP screen, ms)": pa.Column(float),
                "Duration (DG screen, ms)": pa.Column(float),
                "Duration (GET request, ms)": pa.Column(float),
                "Screened": pa.Column(int),
                "Hits (FP screen)": pa.Column(int),
                "Hits (DG screen)": pa.Column(int),
            }
        )

    def test_completes(self, logs):
        analysis.flatten_logs(logs)

    def test_returns_dataframe(self, logs):
        actual = analysis.flatten_logs(logs)

        assert isinstance(actual, pd.DataFrame)

    def test_returns_dataframe_given_fp_only_screen(self, fp_only_logs):
        actual = analysis.flatten_logs(fp_only_logs)

        assert isinstance(actual, pd.DataFrame)

    def test_returns_dataframe_with_expected_schema(self, logs, schema):
        actual = analysis.flatten_logs(logs)

        schema.validate(actual)

    def test_returns_dataframe_with_expected_length(self, logs):
        actual = analysis.flatten_logs(logs)

        assert len(actual) == 13

    def test_returns_dataframe_with_expected_data(self, logs, flattened_logs):
        actual = analysis.flatten_logs(logs)

        assert_frame_equal(actual.sort_index(axis=1), flattened_logs.sort_index(axis=1))

    def test_returns_dataframe_with_expected_data_given_fp_only_screen(
        self, fp_only_logs, flattened_fp_only_logs
    ):
        actual = analysis.flatten_logs(fp_only_logs)

        assert_frame_equal(
            actual.sort_index(axis=1), flattened_fp_only_logs.sort_index(axis=1)
        )


class TestReadLogs:
    @pytest.fixture
    def columns(self):
        return [
            "molecules",
            "fingerprints",
            "cartridge",
            "fingerprint_method",
            "bit_length",
            "duration",
            "event",
            "query",
            "screened",
            "hits",
        ]

    def test_completes(self, filepath):
        analysis.read_logs(filepath)

    def test_returns_dataframe(self, filepath):
        actual = analysis.read_logs(filepath)

        assert isinstance(actual, pd.DataFrame)

    def test_returns_dataframe_with_expected_length(self, filepath):
        actual = analysis.read_logs(filepath)

        assert len(actual) == 89

    def test_returns_dataframe_with_expected_columns(self, filepath, columns):
        actual = analysis.read_logs(filepath, columns)

        assert_index_equal(actual.columns, pd.Index(columns))
