import pytest

from chemsearch.molecule import Molecule


class TestMolecule:
    def test_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            Molecule()
