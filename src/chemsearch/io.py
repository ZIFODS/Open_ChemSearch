from pathlib import Path
from urllib.parse import urlparse

import boto3
import dask.dataframe as dd
import numpy as np
import pandas as pd
from tqdm import tqdm

from chemsearch.molecule import Molecule
from chemsearch.molecule.factory import MoleculeFactory

_S3_CLIENT = boto3.client("s3")


def convert_sdf_to_parquet(
    input_filepath: Path,
    output_filepath: Path,
    factory: MoleculeFactory,
    chunk_size: int = 1_000,
) -> None:
    """Convert SDF file to parquet file with binary representation of molecules.

    Args:
        input_filepath (Path): Path to SDF file.
        output_filepath (Path): Directory to write files.
        factory (MoleculeFactory): Factory to create molecules.
        chunk_size (int): Molecules per partition in parallelisation of work.
    """
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    ddf = dd.from_pandas(
        pd.DataFrame({"MOL Block": read_sdf_file(input_filepath)}), chunksize=chunk_size
    )

    df = ddf.map_partitions(
        _convert_molecules_from_mol_blocks,
        factory=factory,
        meta={"SMILES": str, "Molecule": np.dtype("O")},
    ).compute()

    df.to_parquet(output_filepath)


def _convert_molecules_from_mol_blocks(
    df: pd.DataFrame, factory: MoleculeFactory
) -> pd.DataFrame:
    molecules = df["MOL Block"].apply(factory.from_mol_block).to_list()

    df["Molecule"] = [mol.to_bytes() for mol in molecules]
    df["SMILES"] = [mol.to_smiles(canonical=True) for mol in molecules]

    return df[["SMILES", "Molecule"]]


def convert_smi_to_parquet(
    input_filepath: Path,
    output_filepath: Path,
    factory: MoleculeFactory,
    chunk_size: int = 1_000,
) -> None:
    """Convert SMI file to parquet file with binary representation of molecules.

    Args:
        input_filepath (Path): Path to SMI file.
        output_filepath (Path): Directory to write files.
        factory (MoleculeFactory): Factory to create molecules.
        chunk_size (int): Molecules per partition in parallelisation of work.
    """
    output_filepath.parent.mkdir(parents=True, exist_ok=True)

    ddf = dd.from_pandas(
        pd.DataFrame({"SMILES": read_smi_file(input_filepath)}), chunksize=chunk_size
    )

    df = ddf.map_partitions(
        _convert_molecules_from_smiles,
        factory=factory,
        meta={"SMILES": str, "Molecule": np.dtype("O")},
    ).compute()

    df.to_parquet(output_filepath)


def _convert_molecules_from_smiles(
    df: pd.DataFrame, factory: MoleculeFactory
) -> pd.DataFrame:
    molecules = df["SMILES"].apply(factory.from_smiles).to_list()

    df["Molecule"] = [mol.to_bytes() for mol in molecules]

    return df[["SMILES", "Molecule"]]


def download_from_s3(uri: str, filepath: Path) -> None:
    """Download file from s3.

    Args:
        uri (str): URI for s3 object.
        filepath (Path): Path to write object to.
    """
    url_components = urlparse(uri)

    bucket = url_components.netloc
    key = url_components.path.lstrip("/")

    filepath.parent.mkdir(exist_ok=True, parents=True)

    with open(filepath, "wb") as f:
        _S3_CLIENT.download_fileobj(bucket, key, f)


def is_s3_uri(uri: str) -> bool:
    """Determine if path is an s3 URI.

    Args:
        uri (str): Path to check.

    Returns:
        bool: s3 URI.
    """
    parsed_uri = urlparse(uri)
    return parsed_uri.scheme == "s3"


def read_mol_blocks(mol_blocks: list[str], factory: MoleculeFactory) -> list[Molecule]:
    """Read MOl blocks to molecules.

    Args:
        list[str]: MOL blocks.
        factory (MoleculeFactory): Factory to create molecules.

    Returns:
        list[Molecule]: Molecules.
    """
    return [
        factory.from_mol_block(block)
        for block in tqdm(mol_blocks, "Creating molecules from MOL blocks")
    ]


def read_sdf_file(filepath: Path) -> list[str]:
    """Read all MOL blocks from SDF file.

    Args:
        filepath (Path): Path to SDF file.

    Returns:
        list[str]: MOL blocks.
    """
    iterator = SDMolIter(filepath)

    return [mol_block for mol_block in iterator]


def read_smiles_strings(smiles: list[str], factory: MoleculeFactory) -> list[Molecule]:
    """Read SMILES strings to molecules.

    Args:
        smiles (list[str]): SMILES strings.
        factory (MoleculeFactory): Factory to create molecules.

    Returns:
        list[Molecule]: Molecules.
    """
    return [
        factory.from_smiles(i)
        for i in tqdm(smiles, desc="Creating molecules from SMILES strings")
    ]


def read_smi_file(filepath: Path) -> list[str]:
    """Read all SMILES strings from SMI file.

    Args:
        filepath (Path): Path to SMI file.

    Returns:
        list[str]: SMILES strings.
    """
    iterator = SMIMolIter(filepath)

    return [smiles for smiles in iterator]


def serialise_as_smiles(mols: list[Molecule]) -> list[str]:
    """Serialise molecules as canonical SMILES strings.

    Args:
        mols (list[Molecule]): Molecules.

    Returns:
        list[str]: SMILES strings.
    """
    return [
        mol.to_smiles(canonical=True)
        for mol in tqdm(mols, "Creating SMILES strings from molecules")
    ]


def upload_smi_file(uri: str, smiles: list[str]) -> None:
    """Upload molecules to SMI file in s3 bucket.

    Args:
        uri (str): URI to put file in s3 bucket.
        smiles (list[str]): Molecules to write.
    """
    url_components = urlparse(uri)

    bucket = url_components.netloc
    key = url_components.path.lstrip("/")
    body = "\n".join(smiles).encode()

    _ = _S3_CLIENT.put_object(Bucket=bucket, Key=key, Body=body)


def write_smi_file(filepath: Path, smiles: list[str]) -> None:
    """Write molecules to SMI file.

    Args:
        filepath (Path): Path to write file to.
        smiles (list[str]): Molecules to write.
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)

    with open(filepath, "w") as fh:
        fh.write("\n".join(smiles))


class SDMolIter:
    """SDFile iterator.

    Returns one SDFile record at a time.
    """

    def __init__(self, path):
        self._path = path
        self._size = self._get_dataset_size()

    def __len__(self):
        return self._size

    def __iter__(self):
        return self._get_generator()

    def _get_dataset_size(self):
        size = 0
        with open(self._path) as f:
            for line in f:
                if line.startswith("$$$$"):
                    size += 1
        return size

    def _get_generator(self):
        text_buffer = []
        with open(self._path) as f:
            for line in f:
                text_buffer.append(line)
                if line.startswith("$$$$"):
                    mol_file = "".join(text_buffer)
                    text_buffer = []
                    yield mol_file


class SMIMolIter:
    """SMI File iterator.

    Returns one SMILES string at a time.
    """

    def __init__(self, path):
        self._path = path
        self._size = self._get_dataset_size()

    def __len__(self):
        return self._size

    def __iter__(self):
        return self._get_generator()

    def _get_dataset_size(self):
        size = 0
        with open(self._path) as f:
            for _ in f:
                size += 1
        return size

    def _get_generator(self):
        with open(self._path) as f:
            for line in f:
                smiles = line.split("\t")[0].strip("\n")
                yield smiles
