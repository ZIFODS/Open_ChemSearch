import pytest

from chemsearch.fingerprints import Fingerprints


class TestFingerprints:
    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Fingerprints()
