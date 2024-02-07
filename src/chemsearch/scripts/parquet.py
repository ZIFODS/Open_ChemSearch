import argparse
import logging
from pathlib import Path

from dask.distributed import Client, LocalCluster

from chemsearch import io
from chemsearch.constants import Cartridges
from chemsearch.molecule.factory import MoleculeFactory

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--molecules",
        help="File to read molecules from.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="File to write molecules to.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-c",
        "--cartridge",
        help="Chemistry cartridge to use.",
        default=Cartridges.RDKit,
        type=Cartridges,
    )
    parser.add_argument(
        "-p",
        "--processes",
        help="Processes for Dask local cluster.",
        default=11,
        type=int,
    )

    args = parser.parse_args()

    return args


def main():
    """Entry point for script."""
    args = parse_args()

    factory = MoleculeFactory(args.cartridge)

    logging.info("Writing molecules database to parquet file.")

    match args.molecules.suffix.lower():
        case ".sdf":
            func = io.convert_sdf_to_parquet

        case ".smi":
            func = io.convert_smi_to_parquet

        case _:
            raise ValueError(
                "Unrecognised file extension. Only SDF and SMI files can be read."
            )

    with LocalCluster(
        n_workers=args.processes, threads_per_worker=1
    ) as local_cluster, Client(local_cluster) as client:
        logging.info(f"Link to Dask dashboard: {client.dashboard_link}")
        func(args.molecules, args.output, factory)

    logging.info(f"Data written to: '{args.output}'")


if __name__ == "__main__":
    main()
