import logging
from functools import cache
from pathlib import Path
from time import perf_counter

import cupy as cp
import structlog
from dask.distributed import Client, LocalCluster, performance_report
from fastapi import Depends
from structlog._config import BoundLoggerLazyProxy

from chemsearch.constants import INTERIM_DIR, Databases
from chemsearch.constants import Events as events
from chemsearch.database import Database, DatabaseFactory
from chemsearch.fingerprints import Fingerprints, FingerprintsFactory
from chemsearch.io import download_from_s3, is_s3_uri
from chemsearch.molecule import MoleculeFactory
from chemsearch.utils import time_func

from .config import Settings


@cache
def get_settings() -> Settings:
    """Get settings for app.

    Returns:
        Settings: App settings.
    """
    return Settings()


@cache
def get_logger(settings: Settings = Depends(get_settings)) -> BoundLoggerLazyProxy:
    """Get logger for app.

    Args:
        settings (Settings, optional): App settings.

    Returns:
        BoundLoggerLazyProxy: Logger.
    """
    logger_name = str(settings.log_file)
    logger = logging.getLogger(logger_name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        settings.log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(settings.log_file)
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger(
        logger_name,
        cartridge=settings.cartridge,
        database=settings.database,
        molecules=(
            str(Path(settings.molecules).as_posix())
            if not is_s3_uri(settings.molecules)
            else settings.molecules
        ),
        fingerprints=(
            str(Path(settings.fingerprints).as_posix())
            if not is_s3_uri(settings.fingerprints)
            else settings.fingerprints
        ),
        fingerprint_method=settings.fingerprint_method,
        bit_length=settings.bit_length,
        processes=settings.processes,
        threads_per_process=settings.threads_per_process,
        partitions_per_thread=settings.partitions_per_thread,
    )

    return logger


@cache
def get_molecule_factory(settings: Settings = Depends(get_settings)) -> MoleculeFactory:
    """Get molecule factory for app.

    Args:
        settings (Settings, optional): App settings.

    Returns:
        MoleculeFactory: Molecule factory.
    """
    molecule_factory = MoleculeFactory(settings.cartridge)

    return molecule_factory


@cache
def get_dask_client(
    logger: BoundLoggerLazyProxy = Depends(get_logger),
    settings: Settings = Depends(get_settings),
) -> Client:
    """Start local cluster with Dask client.

    Args:
        logger (BoundLoggerLazyProxy, optional): Logger.
        settings (Settings, optional): App settings.

    Returns:
        Client: Dask client.
    """
    if settings.database != Databases.DASK:
        yield None

    start = perf_counter()

    local_cluster = LocalCluster(
        n_workers=settings.processes,
        threads_per_worker=settings.threads_per_process,
        memory_limit=0,
    )
    client = Client(local_cluster)

    try:
        logger.info(
            events.completed_cluster,
            duration=perf_counter() - start,
            link=client.dashboard_link,
        )

        with performance_report(filename=get_dask_report_filepath(settings)):
            yield client

    finally:
        local_cluster.close()
        client.close()


def get_dask_report_filepath(settings: Settings) -> Path:
    """Derive filepath to Dask performance report from path to log file.

    A numeric suffix is appended to the filepath until a unique filepath is found.

    Args:
        settings (Settings): App settings.

    Returns:
        Path: Path to report.
    """
    fp = settings.log_file.with_suffix(".html")

    i = 0
    while fp.exists():
        i += 1
        fp = fp.with_stem(f"{settings.log_file.stem}_{i}")

    return fp


@cache
def get_molecules(
    factory: MoleculeFactory = Depends(get_molecule_factory),
    dask_client: Client = Depends(get_dask_client),
    logger: BoundLoggerLazyProxy = Depends(get_logger),
    settings: Settings = Depends(get_settings),
) -> Database:
    """Get molecules for app.

    If the input file location is an s3 URI,
    it is first downloaded to the interim data directory.

    Args:
        factory (MoleculeFactory, optional): Molecule factory.
        logger (BoundLoggerLazyProxy, optional): Logger.
        settings (Settings, optional): App settings. Defaults to Depends(get_settings).

    Returns:
        Database: Molecules.
    """
    _ = next(dask_client)
    db_factory = DatabaseFactory(settings.database)

    if s3_uri := is_s3_uri(settings.molecules):
        filepath = (INTERIM_DIR / "molecules").with_suffix(
            Path(settings.molecules).suffix
        )
        download_from_s3(settings.molecules, filepath)

    duration, molecules = time_func(db_factory.from_file)(
        Path(settings.molecules) if not s3_uri else filepath,
        factory,
        partitions_per_thread=settings.partitions_per_thread,
    )
    logger.info(events.completed_db, duration=duration)

    return molecules


@cache
def get_fingerprints(
    logger: BoundLoggerLazyProxy = Depends(get_logger),
    settings: Settings = Depends(get_settings),
) -> Fingerprints:
    """Get fingerprints for app.

    If the input file location is an s3 URI,
    it is first downloaded to the interim data directory.

    Args:
        logger (BoundLoggerLazyProxy, optional): Logger.
        settings (Settings, optional): App settings. Defaults to Depends(get_settings).

    Returns:
        Fingerprints: Fingerprints.
    """
    factory = FingerprintsFactory(settings.fingerprint_method)

    if s3_uri := is_s3_uri(settings.fingerprints):
        filepath = (INTERIM_DIR / "fingerprints").with_suffix(
            Path(settings.fingerprints).suffix
        )
        download_from_s3(settings.fingerprints, filepath)

    duration, fingerprints = time_func(factory.from_file)(
        Path(settings.fingerprints) if not s3_uri else filepath
    )
    logger.info(events.completed_matrix, duration=duration)

    return fingerprints


def clear_all_caches() -> None:
    """Clear memory cache for all app dependencies."""
    for func in (
        get_settings,
        get_logger,
        get_dask_client,
        get_molecule_factory,
        get_molecules,
        get_fingerprints,
    ):
        func.cache_clear()

    clear_memory_pool()


def clear_memory_pool() -> None:
    """Clear CuPy memory pool on GPU."""
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
