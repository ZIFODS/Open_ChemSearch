import pytest
from pyprojroot import here

from chemsearch.app.config import Settings, read_settings_from_json


class TestSettings:
    def test_completes(self):
        _ = Settings(
            molecules="tests/data/sample.smi",
            fingerprints="tests/data/sample_fingerprints.npy",
            bit_length=2048,
        )

    def test_returns_settings(self):
        actual = Settings(
            molecules="tests/data/sample.smi",
            fingerprints="tests/data/sample_fingerprints.npy",
            bit_length=2048,
        )

        assert isinstance(actual, Settings)

    def test_raises_exception_given_molecules_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            _ = Settings(
                molecules="unrecognised",
                fingerprints="tests/data/sample_fingerprints.npy",
                bit_length=2048,
            )

        assert str(excinfo.value) == "Molecules file not found at: unrecognised"

    def test_raises_exception_given_fingerprints_file_does_not_exist(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            _ = Settings(
                molecules="tests/data/sample.smi",
                fingerprints="unrecognised",
                bit_length=2048,
            )

        assert str(excinfo.value) == "Fingerprints file not found at: unrecognised"


class TestReadSettingsFromJSON:
    @pytest.fixture
    def filepath(self):
        return here() / "tests" / "data" / "settings.json"

    def test_completes(self, filepath):
        read_settings_from_json(filepath)

    def test_returns_expected_settings(self, filepath):
        actual = read_settings_from_json(filepath)

        assert actual == [
            Settings(
                molecules="tests/data/sample.smi",
                fingerprints="tests/data/sample_fingerprints.npy",
                cartridge="rdkit",
                bit_length=2048,
                fingerprint_method="cpu",
            ),
            Settings(
                molecules="tests/data/sample.smi",
                fingerprints="tests/data/sample_fingerprints.npy",
                cartridge="rdkit",
                bit_length=2048,
                fingerprint_method="gpu",
            ),
        ]

    def test_raises_exception_given_no_file(self):
        with pytest.raises(FileNotFoundError):
            read_settings_from_json("unrecognised")
