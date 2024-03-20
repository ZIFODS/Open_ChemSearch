![image](https://github.com/ZIFODS/Open_ChemSearch/assets/122999957/40650bec-4be2-4e86-9677-8d82be4dfb0c)

# Chemsearch

Substructure search with initial fingerprint screen accessed through REST API.

The purpose of this project is to benchmark substructure search with different chemistry cartridges, fingerprints and parallelisation methods.

## Environment Setup

The Python environment can be recreated through [Mamba](https://mamba.readthedocs.io/en/latest/):

```
mamba env create -f environment.yml
mamba activate chemsearchEnv
```

Once the Python environment is activated, ensure everything is setup correctly through running the test suite:

```
python -m pytest tests
```

## Usage

With the environment setup complete, it will be possible to run the app.

The project can be run through files in the [scripts directory](src/chemsearch/scripts/):

1. [Fingerprints](#fingerprints), for generating and saving fingerprints in a binary file.
1. [Parquet](#parquet), for saving molecules in a binary file (optional).
1. [App](#app), for running the chemsearch application.
1. [Benchmark](#benchmark), for testing the performance of the running application.

### Fingerprints

The first stage of the substructure search screening cascade involves a fingerprint screen.
The generation of fingerprints from molecules is slow,
so it is better to generate them once for the application and then reuse.

The [fingerprints script](src/chemsearch/scripts/fingerprints.py) reads molecules from SDF or SMI files, calculates fingerprints and writes them to a NumPy binary file.
The file stores a matrix of fingerprints with dimensions (molecules, bit length).

NumPy binary files of the **Original Datasets** used for the publication can be found in [**Zenodo**](https://zenodo.org/records/10842827)

**Command Line**

```
fingerprints --molecules path/to/read/molecules.smi -o path/to/write/fingerprints.npy -b 2048
```

| Flag | Type | Description | Required? | Allowed Values | Default Value |
| --- | --- | --- | --- | --- | --- |
| -m / --molecules | str | File to read molecules from. | Yes | | |
| -o / --output | str | File to write fingerprints to. | Yes | | |
| -b / --bit-length | int | Length of fingerprints. | Yes | | |
| -c / --cartridge | str | Chemistry cartridge to use. | No | rdkit | rdkit |
| -p / --processes | int | Processes for Dask local cluster. | No | | 11 |

### Parquet

Molecules can be read by the app from either SDF or SMI files.
This is fine for small numbers of molecules, however it can be slow on a large scale.
The solution is to use a file format where the molecules are already parsed by the cartridge to Python objects.
Pickle files would be a natural choice but are not supported by the Dask library, which is required for CPU parallelisation.

Parquet files are a good alternative.
The [parquet script](src/chemsearch/scripts/parquet.py) reads molecules from SDF or SMI files, and writes them to a parquet file that can be read faster by the app.
The molecule objects are serialised in the parquet file as a binary representation, to get quick loading whilst staying within the constraints of the allowed types.

**Command Line**

```
parquet --molecules path/to/read/molecules.smi -o path/to/write/molecules
```

| Flag | Type | Description | Required? | Allowed Values | Default Value |
| --- | --- | --- | --- | --- | --- |
| -m / --molecules | str | File to read molecules from. | Yes | | |
| -o / --output | str | File to write molecules to. | Yes | | |
| -c / --cartridge | str | Chemistry cartridge to use. | No | rdkit | rdkit |
| -p / --processes | int | Processes for Dask local cluster. | No | | 11 |

### App

The chemsearch app is built using FastAPI, and runs on uvicorn.
The [app script](src/chemsearch/scripts/app.py) requires environment variables to be set before running.

| Variable | Type | Description | Required? | Allowed Values | Example Value | Default Value |
| --- | --- | --- | --- | --- | --- | --- |
| MOLECULES | str | File to read molecules from. | Yes | | path/to/read/molecules.parquet | |
| FINGERPRINTS | str | File to read fingerprints from. | Yes | | path/to/read/fingerprints.npy | |
| BIT_LENGTH | int | Length of fingerprints. | Yes | | 6144 | |
| CARTRIDGE | str | Chemistry cartridge to use. | No | rdkit | | rdkit |
| DATABASE | str | Library for storage of molecules. | No |  dask/ pandas | | dask |
| PROCESSES |  int | Number of processes, if using dask for database. | No |  | 4 | 8 |
| THREADS_PER_PROCESS |  int | Number of threads per process, if using dask for database. | No |  | 1 | 1 |
| PARTITIONS_PER_THREAD |  int | Number of partitions per thread, if using dask for database. | No |  | 2 | 1 |
| FINGERPRINT_METHOD | str | How to compare fingerprints. | No | cpu/ gpu | | gpu |
| LOG_FILE | str | File to write logs to. | No | | path/to/write/logs.log | log/app_yyyy_mm_dd_hh_mm_ss.log |
| OUTPUT_DIR | str | Directory to persist hits to. | No | | path/to/dir/ | data/results/ |

**Command Line**

```
app -p 8080
```

| Flag | Type | Description | Required? | Default Value |
| --- | --- | --- | --- | --- |
| --host | str | Host of running server. | No | 127.0.0.1 |
| -p / --port | int | Port of running server. | No | 8000 |

**Expected Output**

The following console logs are typical for successful startup of the application:

```
INFO:     Started server process [8716]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

**Client**

Requests can be sent to the running app through the browser:

```
http://<IP address>:<port number>/substructure-search?smarts=<SMARTS string>
http://<IP address>:<port number>/substructure-search?smiles=<SMILES string>
```

The substructure search query strings must parse to valid molecules.

The SMILES strings of molecules from the database matching the substructure query can either be returned directly in the JSON response or persisted to a SMI file.
This is controlled through the **persist** boolean parameter in the URL.
If `persist=true`, the hits will be written to a file in the directory specified in the **OUTPUT_DIR** environment variable.
The filepath to this file will be returned in the JSON response.

#### Fingerprint Screen

The fingerprint screen can also be run in isolation,
for fast benchmarking without waiting for the direct graph screen:

```
http://<IP address>:<port number>/fingerprint-screen?smarts=<SMARTS string>
http://<IP address>:<port number>/fingerprint-screen?smiles=<SMILES string>
```

This will only return the counts of molecules that pass the screen to minimise data transfer delays.

#### Direct Graph Screen

The direct graph screen can also be run in isolation,
for benchmarking without the influence of the fingerprint screen:

```
http://<IP address>:<port number>/direct-graph-screen?smarts=<SMARTS string>
http://<IP address>:<port number>/direct-graph-screen?smiles=<SMILES string>
```

### Benchmark

The [benchmark script](src/chemsearch/scripts/benchmark.py) allows performance testing of the application,
running through a Starlette/ httpx test client.
It does this by executing queries against all configurations of the application defined in a [settings JSON file](tests/data/settings.json).
The key-value pairs in the file must correspond to the environment variables required by the [app](#app).

The duration of processes during query execution can be analysed by inspecting the application logs.

**Command Line**

```
benchmark -q path/to/read/queries.smi -s path/to/read/settings.json
```

| Flag | Type | Description | Required? | Allowed Values | Default Value |
| --- | --- | --- | --- | --- | --- |
| -q / --queries | str | SMI file to read query molecules from. | Yes | | |
| -s / --settings | str | JSON file to read settings from. | Yes | | |
| -f / --format | str | Format of query strings. | No | smarts/ smiles | smiles |
| --fp-only | bool | Only carry out fingerprint screen. | No | | False |
| --dg-only | bool | Only carry out direct graph screen. | No | | False |
| --runs | int | Number of times to run queries. | No | | 3 |
| --cuts | int | Number of cuts for splitting queries. | No | | 10 |
| --seed | int | Seed for shuffling queries and runs. | No | | |

**Expected Output**

The following console logs are typical for successful completion of the benchmark script:

```
Executing SMILES string queries: 100%|██████████████| 1850/1850 [00:16<00:00, 112.84it/s]
```

**Kubernetes Cluster**

The app can be containerised and run in a Kubernetes cluster.
This allows the processing of more molecules than can fit in memory on a local workstation,
and has been tested on tens of millions of molecules.

A [docker file](Dockerfile) has been prepared to setup the Docker image,
using the micromamba parent image to setup both the Python environment and the CUDA installation.

The Docker image can be used with the Kubernetes manifest files in the [k8s directory](k8s).
The manifests have been structured as a helm project,
to allow configuration of environment variables through a single [values.yaml file](k8s/values.yaml).
Once the "insertValueHere" values in this file are filled in, they are used to complete the [yaml templates](k8s/templates/) and create valid Kubernetes manifests.

The Kubernetes architecture has been structured around AWS.
The templates creating this architecture are:

1. A [stateful set](/k8s/templates/stateful-set.yaml) of application containers.
1. AWS Elastic File System (EFS) [storage class](/k8s/templates/storage-class.yaml) for logs.
1. [Headless network](/k8s/templates/svc.yaml) for communicating with containers.
1. [Service account](/k8s/templates/service-account.yaml) for enabling connection to AWS S3 storage for persisting hits.

To setup a Kubernetes cluster:

1. Install helm, kubectl, eksctl and docker on your local computer.

1. Build the docker image and push it to the AWS Elastic Container Registry (ECR).

1. Create an AWS Elastic Kubernetes Service (EKS) cluster with GPU nodes using eksctl.

1. Install the [AWS EFS CSI driver](https://docs.aws.amazon.com/eks/latest/userguide/efs-csi.html) on the cluster to enable containers to use AWS EFS storage.

1. Create an AWS S3 bucket, and configure the AWS IAM OIDC provider, role and policy for the cluster to use the S3 bucket.

1. Fill in the [values.yaml file](k8s/values.yaml), and use helm to install the application on the cluster:

   ```
    `helm install chemsearch k8s`
   ```

Once the cluster is setup, it can be benchmarked through the [chemsearch-benchmark](https://github.com/ZIFODS/Open_ChemSerarch-benchmark) project.

![image](https://github.com/ZIFODS/Open_ChemSearch/assets/122999957/11105bbc-2373-406c-bc6a-8a043650e4f4)

