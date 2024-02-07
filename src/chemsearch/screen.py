import numpy as np
import structlog

from chemsearch.constants import Events as events
from chemsearch.database import Database
from chemsearch.fingerprints import Fingerprints
from chemsearch.molecule import Molecule
from chemsearch.utils import time_func


def execute_direct_graph_screen(
    query: Molecule,
    molecules: Database,
    mask: np.ndarray | None = None,
    logger=structlog.get_logger(),
) -> list[str]:
    """Find all molecules matching substructure query.

    Args:
        query (Molecule): Substructure query.
        molecules (Database): Molecules to search.
        mask (np.ndarray | None, optional): Boolean array of molecules to screen.
            Defaults to None.

    Returns:
        list[str]: SMILES strings of hits.
    """
    duration, substructure_hits = time_func(molecules.get_substructure_hits)(
        query, mask=mask
    )
    logger.debug(
        events.completed_dg,
        duration=duration,
        screened=int(mask.sum()) if mask is not None else len(molecules.smiles),
        hits=len(substructure_hits),
    )

    return substructure_hits


def execute_fingerprint_screen(
    query: Molecule,
    fingerprints: Fingerprints,
    bit_length: int,
    logger=structlog.get_logger(),
) -> np.ndarray:
    """Get boolean mask for all fingerprints matching substructure query.

    The fingerprints must have the same bit length as given.

    Args:
        query (Molecule): Substructure query.
        molecules (Database): Molecules to search.
        fingerprints (Fingerprints): Molecular fingerprints for first pass screen.
        bit_length (int): Bit length for generating query fingerprint.

    Returns:
        np.ndarray: Boolean mask.
    """
    duration, fp_query = time_func(query.get_substructure_fingerprint)(bit_length)
    logger.debug(events.completed_fp_gen, duration=duration)

    duration, fp_hits = time_func(fingerprints.screen)(fp_query)
    logger.debug(
        events.completed_fp,
        duration=duration,
        screened=fp_hits.shape[0],
        hits=int(fp_hits.sum()),
    )

    return fp_hits


def execute_multistage_screen(
    query: Molecule,
    molecules: Database,
    fingerprints: Fingerprints,
    bit_length: int,
    logger=structlog.get_logger(),
) -> list[str]:
    """Find all molecules matching substructure query.

    A two-stage screen is used:
        1. Fingerprint comparison.
        2. Direct graph search.

    The fingerprints must share the same index as the molecules database
    and have the same bit length as given.

    Args:
        query (Molecule): Substructure query.
        molecules (Database): Molecules to search.
        fingerprints (Fingerprints): Molecular fingerprints for first pass screen.
        bit_length (int): Bit length for generating query fingerprint.

    Returns:
        list[str]: SMILES strings of hits.
    """
    fp_hits = execute_fingerprint_screen(query, fingerprints, bit_length, logger)

    substructure_hits = execute_direct_graph_screen(query, molecules, fp_hits, logger)

    return substructure_hits
