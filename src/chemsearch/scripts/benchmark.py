import argparse
from pathlib import Path
from random import seed, shuffle

import numpy as np
from fastapi.testclient import TestClient
from tqdm import tqdm

from chemsearch import io
from chemsearch.app import app, dependencies, read_settings_from_json
from chemsearch.constants import Events as events
from chemsearch.constants import Queries, URLPaths


def parse_args() -> argparse.Namespace:
    """Parse arguments from command line.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--queries",
        help="SMI file to read query molecules from.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-s",
        "--settings",
        help="JSON file to read settings from.",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "-f",
        "--format",
        help="Format of query strings.",
        default=Queries.SMILES,
        type=Queries,
    )
    parser.add_argument(
        "--fp-only",
        help="Only carry out fingerprint screen.",
        action="store_const",
        dest="fp_only",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--dg-only",
        help="Only carry out direct graph screen.",
        action="store_const",
        dest="dg_only",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--runs",
        help="Number of times to run queries.",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--cuts",
        help="Number of cuts for splitting queries.",
        default=10,
        type=int,
    )
    parser.add_argument(
        "--seed",
        help="Seed for shuffling queries and runs.",
        default=None,
        type=int,
    )

    args = parser.parse_args()

    return args


def main():
    """Entry point for script."""
    args = parse_args()

    all_settings = read_settings_from_json(args.settings)
    settings = all_settings[0]

    def get_settings_override():
        return settings

    app.dependency_overrides[dependencies.get_settings] = get_settings_override

    if args.fp_only:
        url = URLPaths.FINGERPRINT_SCREEN

    elif args.dg_only:
        url = URLPaths.DIRECT_GRAPH_SCREEN

    else:
        url = URLPaths.SUBSTRUCTURE_SEARCH

    if args.seed is not None:
        seed(args.seed)

    queries = io.read_smi_file(args.queries)
    shuffle(queries)
    queries = np.array_split(queries, args.cuts)

    runs = [
        (settings, cut)
        for settings in all_settings
        for cut in range(args.cuts)
        for _ in range(args.runs)
    ]
    shuffle(runs)

    for i, (settings, cut) in enumerate(runs):
        with TestClient(app) as client:
            logger = dependencies.get_logger(settings)

            for query in tqdm(queries[cut], desc=f"Run {i + 1} / {len(runs)}"):
                response = client.get(url, params={args.format: query})
                hits = response.json()

                logger.info(
                    events.completed_get,
                    query=query,
                    duration=response.elapsed.microseconds * 10**-6,
                    hits=hits["count"],
                )

        dependencies.clear_all_caches()


if __name__ == "__main__":
    main()
