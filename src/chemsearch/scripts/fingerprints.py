import argparse
import logging
from pathlib import Path

from dask.distributed import Client, LocalCluster

from chemsearch import io
from chemsearch.constants import Cartridges, FingerprintMethods
from chemsearch.fingerprints.factory import FingerprintsFactory
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
        help="File to write fingerprints to.",
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
        "-b", "--bit-length", help="Length of fingerprints.", required=True, type=int
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

    molecule_factory = MoleculeFactory(args.cartridge)
    fingerprint_factory = FingerprintsFactory(FingerprintMethods.CPU)

    with LocalCluster(
        n_workers=args.processes, threads_per_worker=1
    ) as local_cluster, Client(local_cluster) as client:
        logging.info(f"Link to Dask dashboard: {client.dashboard_link}")

        match args.molecules.suffix.lower():
            case ".sdf":
                mol_blocks = io.read_sdf_file(args.molecules)
                fingerprints = fingerprint_factory.from_mol_blocks(
                    mol_blocks, length=args.bit_length, factory=molecule_factory
                )

            case ".smi":
                smiles = io.read_smi_file(args.molecules)
                fingerprints = fingerprint_factory.from_smiles(
                    smiles, length=args.bit_length, factory=molecule_factory
                )

            case _:
                raise ValueError(
                    "Unrecognised file extension. Only SDF and SMI files can be read."
                )

    logging.info("Writing fingerprints matrix to numpy binary file.")
    fingerprints.to_file(args.output)
    logging.info(f"Data written to: '{args.output}'")


if __name__ == "__main__":
    main()
