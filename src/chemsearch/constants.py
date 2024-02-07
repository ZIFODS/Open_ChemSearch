from enum import StrEnum

from pyprojroot import here


class Cartridges(StrEnum):
    RDKit = "rdkit"


class Databases(StrEnum):
    PANDAS = "pandas"
    DASK = "dask"


class Columns(StrEnum):
    atom_count = "Atom Count"
    bit_length = "Bit Length"
    cartridge = "Cartridge"
    database = "Database"
    duration_all = "Duration (All steps, ms)"
    duration_fp = "Duration (FP screen, ms)"
    duration_fp_gen = "Duration (FP generation, ms)"
    duration_dg = "Duration (DG screen, ms)"
    duration_get = "Duration (GET request, ms)"
    duration_mol_gen = "Duration (Query parsing, ms)"
    duration_persist = "Duration (Persist hits, ms)"
    file_mols = "File (Molecules)"
    file_fp = "File (Fingerprints)"
    fp_method = "Fingerprint Method"
    hits_fp = "Hits (FP screen)"
    hits_dg = "Hits (DG screen)"
    molecule = "Molecule"
    percent_fp = "Percent (FP screen)"
    percent_fp_dg = "Percent (FP + DG screen)"
    percent_dg = "Percent (DG screen)"
    precision = "Precision (FP screen)"
    processes = "Processes"
    query = "Query"
    screened = "Screened"
    threads_per_process = "Threads per Process"
    partitions_per_thread = "Partitions per Thread"
    specificity = "Specificity (FP screen)"


class Events(StrEnum):
    completed_cluster = "Started Dask cluster."
    completed_db = "Created molecules database."
    completed_matrix = "Read fingerprints matrix."
    completed_mol_gen = "Parsed query SMILES string to molecule."
    completed_fp_gen = "Created query fingerprint."
    completed_fp = "Completed fingerprint screen."
    completed_dg = "Completed direct graph screen."
    completed_get = "Received hits from GET request."
    completed_persist = "Persisted hits in SMI file."
    failed_mol_gen = "Query string could not be parsed."
    failed_no_query = "Either a SMARTS or SMILES string query must be provided."


class FingerprintMethods(StrEnum):
    CPU = "cpu"
    GPU = "gpu"


class Queries(StrEnum):
    SMARTS = "smarts"
    SMILES = "smiles"


class URLPaths(StrEnum):
    DIRECT_GRAPH_SCREEN = "/direct-graph-screen"
    FINGERPRINT_SCREEN = "/fingerprint-screen"
    SUBSTRUCTURE_SEARCH = "/substructure-search"


DATA_DIR = here() / "data"
INTERIM_DIR = DATA_DIR / "interim"
RESULTS_DIR = DATA_DIR / "results"
LOG_DIR = here() / "logs"
