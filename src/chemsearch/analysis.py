from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from rdkit import Chem

from chemsearch.constants import Columns as cols
from chemsearch.constants import Events as events

_MULTI_INDEX = [
    "molecules",
    "fingerprints",
    "cartridge",
    "fingerprint_method",
    "bit_length",
    "query",
    "database",
    "processes",
    "threads_per_process",
    "partitions_per_thread",
    "experiment",
]


class ExperimentCounter:
    def __init__(self):
        self.count = 1

    def assign(self, row: pd.Series) -> int:
        """Assign experiment based on event of previous row.

        Assumes that logs for an experiment follow immediately after each other.

        Args:
            row (pd.Series): Single JSON log.

        Returns:
            int: Experiment number.
        """
        if row["query"] is np.nan:
            return 0

        if row["previous_event"] == events.completed_get:
            self.count += 1

        return self.count


def add_molecules(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for molecules.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{cols.molecule: df[cols.query].apply(lambda x: Chem.MolFromSmiles(x))}
    )


def add_atom_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for atom counts.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{cols.atom_count: df[cols.molecule].apply(lambda x: x.GetNumHeavyAtoms())}
    )


def add_total_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for total duration across all steps.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{
            cols.duration_all: df.loc[
                :,
                df.columns.isin(
                    [
                        cols.duration_mol_gen,
                        cols.duration_fp_gen,
                        cols.duration_fp,
                        cols.duration_dg,
                        cols.duration_persist,
                    ]
                ),
            ].sum(axis=1)
        }
    )


def add_percent_fp_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for percent of total duration spent on fingerprint screen.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{cols.percent_fp: 100 * (df[cols.duration_fp] / df[cols.duration_all])}
    )


def add_percent_dg_screen(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for percent of total duration spent on direct graph screen.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{cols.percent_dg: 100 * (df[cols.duration_dg] / df[cols.duration_all])}
    )


def add_specificity(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for specificity using hits from fingerprint and direct graph screens.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(
        **{
            cols.specificity: 100
            * (df[cols.screened] - df[cols.hits_fp])
            / (df[cols.screened] - df[cols.hits_dg])
        }
    )


def add_precision(df: pd.DataFrame) -> pd.DataFrame:
    """Add column for precision using hits from fingerprint and direct graph screens.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with added column.
    """
    return df.assign(**{cols.precision: (100 * df[cols.hits_dg] / df[cols.hits_fp])})


def flatten_logs(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten logs to one row per query.

    Flattening is performed by pivoting on event column and maintaining hit counts.
    The following columns are used for indexing:
        - cartridge
        - fingerprint_method
        - bit_length
        - query

    Other changes include:
        - Filtering of events from startup of app dependencies.
        - Conversion of duration from seconds to milliseconds.
        - Renaming of columns for analysis in notebooks.

    Args:
        df (pd.DataFrame): Long form logs.

    Returns:
        pd.DataFrame: Wide form logs.
    """
    df = df.assign(duration=df["duration"] * 1000)

    df["previous_event"] = df["event"].shift()
    df["experiment"] = df.apply(ExperimentCounter().assign, axis=1)

    df_pivot = df.pivot_table(index=_MULTI_INDEX, columns="event", values="duration")
    df_pivot = df_pivot.loc[
        :,
        df_pivot.columns.isin(
            [
                events.completed_mol_gen,
                events.completed_fp_gen,
                events.completed_fp,
                events.completed_dg,
                events.completed_get,
                events.completed_persist,
            ]
        ),
    ]
    df_pivot = rename_event_columns(df_pivot)

    if cols.duration_fp in df_pivot:
        df_fp_hits = (
            df.loc[
                df["event"] == events.completed_fp,
                _MULTI_INDEX + ["screened", "hits"],
            ]
            .set_index(_MULTI_INDEX)
            .rename(columns={"hits": cols.hits_fp, "screened": cols.screened})
        )
        df_pivot = pd.merge(
            left=df_pivot, right=df_fp_hits, left_index=True, right_index=True
        )

    if cols.duration_dg in df_pivot:
        df_dg_hits = (
            df.loc[df["event"] == events.completed_dg, _MULTI_INDEX + ["hits"]]
            .set_index(_MULTI_INDEX)
            .rename(columns={"hits": cols.hits_dg})
        )
        df_pivot = pd.merge(
            left=df_pivot, right=df_dg_hits, left_index=True, right_index=True
        )

    for col in [cols.screened, cols.hits_fp, cols.hits_dg]:
        if col in df_pivot.columns:
            df_pivot[col] = df_pivot[col].astype("int64")

    df_pivot = df_pivot.reset_index().drop(columns=["experiment"])
    df_pivot = rename_context_columns(df_pivot)

    return df_pivot


def filter_by_fp_method(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Filter dataframe by fingerprint method.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Filtered dataframe.
    """
    return df.loc[df[cols.fp_method] == method]


def read_logs(filepath: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read logs from JSON files.

    Wildcard characters in the filepath can be used to match multiple files.

    Args:
        filepath (Path): Path to file(s).

    Returns:
        pd.DataFrame: Logs.
    """
    df = dd.read_json(filepath)

    if columns is not None:
        df = df[columns]

    return df.compute()


def rename_context_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns derived from context variables of logged events.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with renamed columns.
    """
    return df.rename(
        columns={
            "molecules": cols.file_mols,
            "fingerprints": cols.file_fp,
            "cartridge": cols.cartridge,
            "database": cols.database,
            "fingerprint_method": cols.fp_method,
            "bit_length": cols.bit_length,
            "processes": cols.processes,
            "threads_per_process": cols.threads_per_process,
            "partitions_per_thread": cols.partitions_per_thread,
            "query": cols.query,
        }
    )


def rename_event_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns derived from logged events.

    Args:
        df (pd.DataFrame): Flattened logs.

    Returns:
        pd.DataFrame: Dataframe with renamed columns.
    """
    return df.rename(
        columns={
            events.completed_mol_gen: cols.duration_mol_gen,
            events.completed_fp_gen: cols.duration_fp_gen,
            events.completed_fp: cols.duration_fp,
            events.completed_dg: cols.duration_dg,
            events.completed_get: cols.duration_get,
            events.completed_persist: cols.duration_persist,
        }
    )
