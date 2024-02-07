import numpy as np

from . import Molecule


def compare_by_canonical_smiles(molecule_1: Molecule, molecule_2: Molecule):
    """Compare molecules by their canonical SMILES string.

    Args:
        molecule_1 (Molecule): Molecule to compare.
        molecule_2 (Molecule): Molecule to compare.
    """
    smiles_1 = molecule_1.to_smiles(canonical=True)
    smiles_2 = molecule_2.to_smiles(canonical=True)

    return smiles_1 == smiles_2


def convert_fingerprint_bit_indices_to_vector(
    indices: list[int], length: int
) -> np.ndarray:
    """Convert fingerprint bit indices to dense 1-dimensional array.

    Args:
        indices (list[int]): Fingerprint bit indices.
        length (int): Maximum bit index.

    Returns:
        np.array: Fingerprint.
    """
    if length <= 0:
        raise ValueError("Length should be a positive integer.")

    fp = np.zeros(length, dtype="int8")

    for i in indices:
        fp[i] = 1

    return fp


def pack_fingerprint_to_64_bit_integers(fp: np.ndarray) -> np.ndarray:
    """Pack int8 fingerprint to uint64 datatype.

    The fingerprint length must be a multiple of 64.

    Args:
        fp (np.ndarray): Fingerprint with int8 datatype.

    Returns:
        np.ndarray: Packed fingerprint.
    """
    fp = np.packbits(fp, axis=-1).view("uint64")

    return fp


def pad_fingerprint_to_multiple_of_64(fp: np.ndarray) -> np.ndarray:
    """Pad fingerprint with zeros to a length that is a multiple of 64.

    Args:
        fp (np.ndarray): Fingerprint.

    Returns:
        np.ndarray: Padded fingerprint.
    """
    remainder = fp.shape[-1] % 64

    if remainder:
        fp = np.pad(fp, (0, 64 - remainder))

    return fp
