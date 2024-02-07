from fastapi import Depends, FastAPI
from structlog._config import BoundLoggerLazyProxy

from chemsearch import screen
from chemsearch.app.response import HitsResponse, PersistedResponse
from chemsearch.app.utils import parse_query, process_hits
from chemsearch.constants import URLPaths
from chemsearch.database import Database
from chemsearch.fingerprints import Fingerprints
from chemsearch.molecule import MoleculeFactory

from . import dependencies
from .config import Settings

app = FastAPI()


@app.get(URLPaths.DIRECT_GRAPH_SCREEN)
def direct_graph_screen(
    smiles: str | None = None,
    smarts: str | None = None,
    persist: bool = False,
    factory: MoleculeFactory = Depends(dependencies.get_molecule_factory),
    logger: BoundLoggerLazyProxy = Depends(dependencies.get_logger),
    molecules: Database = Depends(dependencies.get_molecules),
    settings: Settings = Depends(dependencies.get_settings),
) -> HitsResponse | PersistedResponse:
    """Search database for molecules matching substructure query.

    No fingerprint screen is carried out, so all molecules are compared.

    Either a SMILES or SMARTS string query must be provided, not both.

    Args:
        smiles (str | None, optional): SMILES string query. Defaults to None.
        smarts (str | None, optional): SMARTS string query. Defaults to None.
        persist (bool, optional): Persist hits in SMI file. Defaults to False.
        factory (MoleculeFactory, optional): Molecule factory.
        logger (BoundLoggerLazyProxy, optional): Logger.
        molecules (Database, optional): Molecules.
        settings (Settings, optional): App settings.

    Raises:
        HTTPException: If no valid SMILES or SMARTS string.

    Returns:
        HitsResponse | PersistedResponse: Matching molecules.
    """
    query, logger = parse_query(factory, logger, smiles=smiles, smarts=smarts)

    hits = screen.execute_direct_graph_screen(query, molecules, logger=logger)

    result = process_hits(query, hits, persist, logger, settings)

    return result


@app.get(URLPaths.FINGERPRINT_SCREEN)
def fingerprint_screen(
    smiles: str | None = None,
    smarts: str | None = None,
    factory: MoleculeFactory = Depends(dependencies.get_molecule_factory),
    fingerprints: Fingerprints = Depends(dependencies.get_fingerprints),
    logger: BoundLoggerLazyProxy = Depends(dependencies.get_logger),
    settings: Settings = Depends(dependencies.get_settings),
) -> int:
    """Get count of fingerprints that are compatible with substructure query.

    Either a SMILES or SMARTS string query must be provided, not both.

    Args:
        smiles (str | None, optional): SMILES string query. Defaults to None.
        smarts (str | None, optional): SMARTS string query. Defaults to None.
        factory (MoleculeFactory, optional): Molecule factory.
        fingerprints (Fingerprints, optional): Fingerprints.
        logger (BoundLoggerLazyProxy, optional): Logger.
        settings (Settings, optional): App settings.

    Raises:
        HTTPException: If no valid SMILES or SMARTS string.

    Returns:
        int: Fingerprint count.
    """
    query, logger = parse_query(factory, logger, smiles=smiles, smarts=smarts)

    hits = screen.execute_fingerprint_screen(
        query, fingerprints, settings.bit_length, logger
    )

    return hits.sum()


@app.get(URLPaths.SUBSTRUCTURE_SEARCH)
def substructure_search(
    smiles: str | None = None,
    smarts: str | None = None,
    persist: bool = False,
    factory: MoleculeFactory = Depends(dependencies.get_molecule_factory),
    fingerprints: Fingerprints = Depends(dependencies.get_fingerprints),
    logger: BoundLoggerLazyProxy = Depends(dependencies.get_logger),
    molecules: Database = Depends(dependencies.get_molecules),
    settings: Settings = Depends(dependencies.get_settings),
) -> HitsResponse | PersistedResponse:
    """Search database for molecules matching substructure query.

    Either a SMILES or SMARTS string query must be provided, not both.

    Args:
        smiles (str | None, optional): SMILES string query. Defaults to None.
        smarts (str | None, optional): SMARTS string query. Defaults to None.
        persist (bool, optional): Persist hits in SMI file. Defaults to False.
        factory (MoleculeFactory, optional): Molecule factory.
        fingerprints (Fingerprints, optional): Fingerprints.
        logger (BoundLoggerLazyProxy, optional): Logger.
        molecules (Database, optional): Molecules.
        settings (Settings, optional): App settings.

    Raises:
        HTTPException: If no valid SMILES or SMARTS string.

    Returns:
        HitsResponse | PersistedResponse: Matching molecules.
    """
    query, logger = parse_query(factory, logger, smiles=smiles, smarts=smarts)

    hits = screen.execute_multistage_screen(
        query, molecules, fingerprints, settings.bit_length, logger
    )

    result = process_hits(query, hits, persist, logger, settings)

    return result
