import os
from pathlib import Path

from fastapi import HTTPException
from structlog._config import BoundLoggerLazyProxy

from chemsearch import io
from chemsearch.constants import Events as events
from chemsearch.exceptions import MoleculeParsingException
from chemsearch.molecule import Molecule, MoleculeFactory
from chemsearch.utils import time_func

from .config import Settings


def parse_query(
    factory: MoleculeFactory,
    logger: BoundLoggerLazyProxy,
    smiles: str | None = None,
    smarts: str | None = None,
) -> Molecule:
    """Parse either SMILES or SMARTS string query as molecule.

    Args:
        factory (MoleculeFactory, optional): Molecule factory.
        logger (BoundLoggerLazyProxy, optional): Logger.
        smiles (str | None, optional): SMILES string query. Defaults to None.
        smarts (str | None, optional): SMARTS string query. Defaults to None.

    Raises:
        HTTPException: If no valid SMILES or SMARTS string.

    Returns:
        Molecule: Query molecule.
    """
    if (smarts is None) and (smiles is None):
        raise HTTPException(status_code=400, detail=events.failed_no_query)

    query = smarts if (smarts is not None) else smiles
    logger = logger.bind(query=query)

    try:
        method = factory.from_smarts if (smarts is not None) else factory.from_smiles
        duration, query = time_func(method)(query)
        logger.info(events.completed_mol_gen, duration=duration)

    except MoleculeParsingException:
        logger.warning(events.failed_mol_gen)
        raise HTTPException(status_code=400, detail=events.failed_mol_gen)

    return query, logger


def persist_hits(output_dir: str, query: Molecule, hits: list[str]) -> str:
    """Persist hits in an SMI file.

    The output directory can be a local path, or in an s3 bucket.
    The filename is constructed from the hash of the query.

    Args:
        output_dir (str): Directory to write file.
        query (Molecule): Query molecule.
        hits (list[str]): Hits to write.

    Returns:
        str: Location of file.
    """
    filename = f"{query.sha1_hash}.smi"

    if io.is_s3_uri(output_dir):
        filepath = os.path.join(output_dir, filename).replace(os.sep, "/")
        io.upload_smi_file(filepath, hits)

    else:
        filepath = Path(output_dir) / filename
        io.write_smi_file(filepath, hits)

    return str(filepath)


def process_hits(
    query: Molecule,
    hits: list[str],
    persist: bool,
    logger: BoundLoggerLazyProxy,
    settings: Settings,
) -> dict:
    """Construct JSON response and persist hits if desired.

    If there are no hits then no file is uploaded.

    Args:
        query (Molecule): Query molecule.
        hits (list[str]): Hits to process.
        persist (bool): Persist hits in SMI file.
        logger (BoundLoggerLazyProxy): Logger.
        settings (Settings, optional): App settings.

    Returns:
        dict: JSON response.
    """
    count = len(hits)
    result = {"count": count}

    if persist and (count > 0):
        duration, filepath = time_func(persist_hits)(settings.output_dir, query, hits)
        result["filepath"] = filepath

        logger.info(events.completed_persist, duration=duration, output=filepath)

    elif persist and (count == 0):
        result["filepath"] = None

    else:
        result["hits"] = hits

    return result
