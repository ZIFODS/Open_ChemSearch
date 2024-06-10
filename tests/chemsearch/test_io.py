from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyprojroot import here

from chemsearch import io
from chemsearch.constants import Cartridges
from chemsearch.exceptions import MoleculeParsingException
from chemsearch.molecule.factory import MoleculeFactory
from chemsearch.molecule.rdkit import RDKitMolecule


@pytest.fixture
def data_dir():
    return here() / "tests" / "TestData"


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
    return [factory.from_smiles(i) for i in smiles]


class TestConvertSdfToParquet:
    @pytest.fixture
    def parquet_file(self, tmpdir):
        return Path(tmpdir) / "sample.parquet"

    @pytest.fixture
    def sdf_file(self, data_dir):
        return data_dir / "sample.sdf"

    @pytest.fixture
    def factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.fixture
    def molecules(self, smiles, factory):
        return [factory.from_smiles(i) for i in smiles]

    def test_completes(self, sdf_file, parquet_file, factory):
        io.convert_sdf_to_parquet(sdf_file, parquet_file, factory)

    def test_writes_expected_parquet_file(self, sdf_file, parquet_file, factory):
        io.convert_sdf_to_parquet(sdf_file, parquet_file, factory)

        assert parquet_file.exists()

    def test_writes_parquet_file_with_expected_smiles(
        self, sdf_file, parquet_file, factory, smiles
    ):
        io.convert_sdf_to_parquet(sdf_file, parquet_file, factory, chunk_size=3)

        df = pd.read_parquet(parquet_file)
        actual = df["SMILES"].to_list()

        assert actual == smiles

    def test_writes_parquet_file_with_expected_molecules(
        self, sdf_file, parquet_file, factory, molecules
    ):
        io.convert_sdf_to_parquet(sdf_file, parquet_file, factory, chunk_size=3)

        df = pd.read_parquet(parquet_file)
        actual = df["Molecule"].apply(factory.from_bytes).to_list()

        assert actual == molecules


class TestConvertSmiToParquet:
    @pytest.fixture
    def parquet_file(self, tmpdir):
        return Path(tmpdir) / "sample.parquet"

    @pytest.fixture
    def smi_file(self, data_dir):
        return data_dir / "sample.smi"

    @pytest.fixture
    def factory(self):
        return MoleculeFactory(Cartridges.RDKit)

    @pytest.fixture
    def molecules(self, smiles, factory):
        return [factory.from_smiles(i) for i in smiles]

    def test_completes(self, smi_file, parquet_file, factory):
        io.convert_smi_to_parquet(smi_file, parquet_file, factory)

    def test_writes_expected_parquet_file(self, smi_file, parquet_file, factory):
        io.convert_smi_to_parquet(smi_file, parquet_file, factory)

        assert parquet_file.exists()

    def test_writes_parquet_file_with_expected_smiles(
        self, smi_file, parquet_file, factory, smiles
    ):
        io.convert_smi_to_parquet(smi_file, parquet_file, factory, chunk_size=3)

        df = pd.read_parquet(parquet_file)
        actual = df["SMILES"].to_list()

        assert actual == smiles

    def test_writes_parquet_file_with_expected_molecules(
        self, smi_file, parquet_file, factory, molecules
    ):
        io.convert_smi_to_parquet(smi_file, parquet_file, factory, chunk_size=3)

        df = pd.read_parquet(parquet_file)
        actual = df["Molecule"].apply(factory.from_bytes).to_list()

        assert actual == molecules


class TestIsS3URI:
    @pytest.mark.parametrize(
        "uri",
        ["s3://chemsearch/input/chemsearch-0.smi", "/data/input/chemsearch-0.smi"],
    )
    def test_completes(self, uri):
        io.is_s3_uri(uri)

    @pytest.mark.parametrize(
        "uri, expected",
        [
            ("s3://chemsearch/input/chemsearch-0.smi", True),
            ("/data/input/chemsearch-0.smi", False),
        ],
    )
    def test_returns_expected_label(self, uri, expected):
        actual = io.is_s3_uri(uri)

        assert actual == expected


class TestReadSDFFile:
    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample.sdf"

    def test_completes(self, filepath):
        io.read_sdf_file(filepath)

    def test_returns_expected_number_of_elements(self, filepath):
        actual = io.read_sdf_file(filepath)

        assert len(actual) == 10


class TestReadMolBlocks:
    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample.sdf"

    @pytest.fixture
    def mol_blocks(self, filepath):
        return io.read_sdf_file(filepath)

    def test_completes(self, mol_blocks, factory):
        io.read_mol_blocks(mol_blocks, factory)

    def test_returns_expected_number_of_elements(self, mol_blocks, factory):
        actual = io.read_mol_blocks(mol_blocks, factory)

        assert len(actual) == 10

    def test_returns_expected_molecules(self, mol_blocks, factory, molecules):
        actual = io.read_mol_blocks(mol_blocks, factory)

        assert actual == molecules

    def test_raises_exception_given_invalid_format(self, factory):
        with pytest.raises(MoleculeParsingException):
            io.read_mol_blocks("unrecognised", factory)


class TestReadSmilesStrings:
    @pytest.fixture
    def smiles(self, smiles):
        return smiles[:3]

    def test_completes(self, smiles, factory):
        io.read_smiles_strings(smiles, factory)

    def test_returns_expected_number_of_elements(self, smiles, factory):
        actual = io.read_smiles_strings(smiles, factory)

        assert len(actual) == 3

    def test_returns_list_of_molecules(self, smiles, factory):
        actual = io.read_smiles_strings(smiles, factory)

        assert isinstance(actual[0], RDKitMolecule)
        assert isinstance(actual[1], RDKitMolecule)
        assert isinstance(actual[2], RDKitMolecule)


class TestReadSMIFile:
    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample.smi"

    def test_completes(self, filepath):
        io.read_smi_file(filepath)

    def test_returns_expected_number_of_elements(self, filepath):
        actual = io.read_smi_file(filepath)

        assert len(actual) == 10

    def test_returns_expected_molecules(self, filepath, smiles):
        expected = smiles

        actual = io.read_smi_file(filepath)

        assert actual == expected

    def test_raises_exception_given_no_file(self):
        with pytest.raises(FileNotFoundError):
            io.read_smi_file("unknown.smi")


class TestSerialiseAsSmiles:
    @pytest.fixture
    def molecules(self, molecules):
        return molecules[:3]

    def test_completes(self, molecules):
        io.serialise_as_smiles(molecules)

    def test_returns_expected_number_of_elements(self, molecules):
        actual = io.serialise_as_smiles(molecules)

        assert len(actual) == 3

    def test_returns_list_of_strings(self, molecules):
        actual = io.serialise_as_smiles(molecules)

        assert isinstance(actual[0], str)
        assert isinstance(actual[1], str)
        assert isinstance(actual[2], str)

    def test_returns_expected_smiles(self, molecules):
        expected = [
            "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
        ]

        actual = io.serialise_as_smiles(molecules)

        assert actual == expected


class TestWriteSMIFile:
    @pytest.fixture
    def filepath(self, tmp_path):
        return Path(tmp_path) / "results" / "abc54.smi"

    def test_completes(self, filepath, smiles):
        io.write_smi_file(filepath, smiles)

    def test_writes_expected_smi_file(self, filepath, smiles):
        io.write_smi_file(filepath, smiles)

        assert filepath.exists()

    def test_writes_smi_file_with_expected_smiles(self, filepath, smiles):
        io.write_smi_file(filepath, smiles)

        actual = io.read_smi_file(filepath)

        assert actual == smiles


class TestSDMolIter:
    @pytest.fixture
    def filepath(self, data_dir):
        return data_dir / "sample.sdf"

    @pytest.fixture
    def mol_iter(self, filepath):
        return io.SDMolIter(filepath)

    @pytest.fixture
    def iterator(self, mol_iter):
        return iter(mol_iter)

    def test_completes(self, filepath):
        io.SDMolIter(filepath)

    def test_returns_correct_length(self, mol_iter):
        assert len(mol_iter) == 10

    def test_returns_expected_mol_blocks(self, iterator):
        assert (
            next(iterator)
            == """
     RDKit          2D

 34 35  0  0  0  0  0  0  0  0999 V2000
   10.5000   -5.1962    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    9.7500   -3.8971    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.2500   -3.8971    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    7.5000   -5.1962    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    7.5000   -2.5981    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    8.2500   -1.2990    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    6.0000   -2.5981    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    5.2500   -3.8971    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    5.2500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    6.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    3.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500   -1.2990    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000   -2.5981    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -3.8971    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000   -5.1962    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0000   -5.1962    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7500   -6.4952    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.2500   -6.4952    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.0000   -5.1962    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0000   -7.7942    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7500   -9.0933    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000   -7.7942    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -9.0933    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -6.4952    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500   -6.4952    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000    2.5981    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    2.5981    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  2  0
  2  3  1  0
  3  4  1  6
  3  5  1  0
  5  6  1  6
  5  7  1  0
  7  8  1  6
  7  9  1  0
  9 10  1  1
  9 11  1  0
 11 12  1  0
 13 12  1  1
 13 14  1  0
 14 15  1  0
 15 16  1  6
 16 17  1  0
 18 17  1  6
 18 19  1  0
 19 20  1  0
 20 21  1  1
 21 22  1  0
 20 23  1  0
 23 24  1  6
 23 25  1  0
 25 26  1  1
 25 27  1  0
 27 28  1  6
 15 29  1  0
 29 30  1  1
 29 31  1  0
 31 32  1  6
 31 33  1  0
 33 34  1  1
 33 13  1  0
 27 18  1  0
M  END
$$$$
"""
        )

        assert (
            next(iterator)
            == """
     RDKit          2D

 32 34  0  0  0  0  0  0  0  0999 V2000
    5.2500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    4.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000   -1.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    3.0000    1.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500   -1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -1.5000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -3.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
   -3.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -5.2500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -6.0000    2.5981    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -7.5000    2.5981    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -8.2500    3.8971    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -9.7500    3.8971    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -10.5000    5.1962    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -9.7500    6.4952    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -10.5000    7.7942    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0000    7.7942    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
  -12.7500    9.0933    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0000   10.3923    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.7500   11.6913    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -14.2500   11.6913    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.0000   12.9904    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -15.0000   10.3923    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -16.5000   10.3923    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  -14.2500    9.0933    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.7500    6.4952    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  -12.0000    5.1962    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7500    1.2990    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  3  5  1  0
  3  6  1  0
  6  7  2  0
  7  8  1  0
  8  9  2  0
  9 10  1  0
 10 11  1  0
 11 12  1  0
 12 13  1  0
 13 14  1  0
 14 15  1  0
 15 16  1  0
 16 17  1  0
 17 18  1  0
 18 19  1  0
 19 20  1  0
 20 21  1  0
 21 22  2  0
 22 23  1  0
 23 24  2  0
 24 25  1  0
 24 26  1  0
 26 27  1  0
 26 28  2  0
 20 29  1  0
 29 30  1  0
  9 31  1  0
 31 32  2  0
 32  6  1  0
 30 17  1  0
 28 21  1  0
M  END
$$$$
"""
        )
