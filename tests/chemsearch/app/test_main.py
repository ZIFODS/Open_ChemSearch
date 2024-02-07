from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pyprojroot import here

from chemsearch import constants, io
from chemsearch.app.config import Settings
from chemsearch.app.dependencies import get_settings
from chemsearch.app.main import app
from chemsearch.constants import Cartridges, Databases, FingerprintMethods, URLPaths

client = TestClient(app)


def get_settings_override():
    test_dir = here() / "tests" / "data"

    return Settings(
        molecules=str(test_dir / "sample.smi"),
        fingerprints=str(test_dir / "sample_fingerprints.npy"),
        cartridge=Cartridges.RDKit,
        database=Databases.PANDAS,
        bit_length=2048,
        fingerprint_method=FingerprintMethods.CPU,
        log_file=test_dir / "test.log",
        output_dir=str(constants.RESULTS_DIR),
    )


app.dependency_overrides[get_settings] = get_settings_override


@pytest.fixture
def hits(query):
    hits = {
        "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1": [
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1": [
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1": [
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1": [
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1": [
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1": [
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
        ],
        "[#6]-[#6](=[#8])-[*]-[#6]": [
            "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
        ],
        "C1NCCCC1": [
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "C1NCCNC1": [
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
        ],
        "c1ccccc1": [
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "Clc1ccc(N2CCN(C/C=C/c3ccccc3)CC2)nn1",
            "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
            "Fc1ccc(Cn2c(NC3CCNCC3)nc3ccccc32)cc1",
            "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
            "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
            "Cn1c2c(c3ccccc31)CCN1C[C@@H]3CCCC[C@H]3C[C@@H]21",
        ],
        "C=O": [
            "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
            "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
            "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
        ],
        "COC": [
            "O=C[C@H](O)[C@@H](O)[C@H](O)[C@H](O)CO[C@H]1O[C@H](CO[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@H](O)[C@H]1O",
            "CCC(C)(C)c1ccc(OCCCCCCN2CCN(c3ccc(C)c(Cl)c3)CC2)cc1",
            "CCOC(=O)c1ccc(N(CC(C)O)CC(C)O)cc1",
            "CCOc1ccccc1C(=O)N/N=C(\\C)C(=O)O",
            "O=C(CC1C2C=CC=CC2C(=O)N1c1ccc2ccc(Cl)nc2n1)N1CCC2(CC1)OCCO2",
            "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
        ],
        "S": [
            "Cc1cnc(NC(=O)C2=C(O)c3ccccc3S(=O)(=O)N2C)s1",
            "COc1ccc(-c2nc(C(F)(F)F)sc2-c2ccc(OC)cc2)cc1",
        ],
    }

    return hits[query]


@pytest.fixture
def count(query):
    counts = {
        "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1": 3,
        "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1": 2,
        "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1": 2,
        "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1": 1,
        "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1": 5,
        "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1": 2,
        "[#6]-[#6](=[#8])-[*]-[#6]": 3,
        "C1NCCCC1": 3,
        "C1NCCNC1": 2,
        "c1ccccc1": 8,
        "C=O": 5,
        "COC": 6,
        "S": 2,
    }

    return counts[query]


class TestDirectGraphScreen:
    @pytest.mark.parametrize("persist", [True, False])
    def test_returns_successful_response_given_valid_smarts(
        self, mocker, tmp_path, persist
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN,
            params={
                "smarts": "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
                "persist": persist,
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_expected_hits_given_valid_smarts(self, query, count, hits):
        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smarts": query, "persist": False}
        )

        assert response.json() == {"count": count, "hits": hits}

    @pytest.mark.parametrize(
        "query, filename",
        [
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
                "515e05b051d4ed50266c244215d558a021d8359e.smi",
            ),
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
                "89dfa361c41b75f550ef97413eb60698e309676f.smi",
            ),
            (
                "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
                "35a0a7737e981b571462b3697eabd060abc4a166.smi",
            ),
            (
                "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
                "af2777b0386b81a63c8c902349d2b1d256a1d466.smi",
            ),
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
                "465927e3fc8add302f4c25b734776c3baa06ae57.smi",
            ),
            (
                "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
                "4c1affd65c9f9e8ea08a4224e683cf6e3ddc8e9d.smi",
            ),
            (
                "[#6]-[#6](=[#8])-[*]-[#6]",
                "13f16c95201d2ef1e58ae59238fe2b7a433fd332.smi",
            ),
        ],
    )
    def test_returns_expected_response_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query, count, filename
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)
        filepath = str(tmp_path / filename)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smarts": query, "persist": True}
        )

        assert response.json() == {"count": count, "filepath": filepath}

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_path_to_file_that_exists_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smarts": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])

        assert filepath.exists()

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_path_to_file_with_expected_hits_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query, hits
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smarts": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])
        actual = io.read_smi_file(filepath)

        assert actual == hits

    @pytest.mark.parametrize("persist", [True, False])
    def test_returns_successful_response_given_valid_smiles(
        self, mocker, tmp_path, persist
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN,
            params={"smiles": "c1ccccc1", "persist": persist},
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query",
        ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"],
    )
    def test_returns_expected_hits_given_valid_smiles(self, query, count, hits):
        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smiles": query, "persist": False}
        )

        assert response.json() == {"count": count, "hits": hits}

    @pytest.mark.parametrize(
        "query, filename",
        [
            ("C1NCCCC1", "18bc6610235fbbf025ec7c3c04ccd2c1c69c8872.smi"),
            ("C1NCCNC1", "5e99be7a17532f31e10e37d29ed51364a39e9610.smi"),
            ("c1ccccc1", "41e34a1b4adca57862763e2b51310d17412e90d8.smi"),
            ("C=O", "68ebf88c1bed4c4376db03a840a25623498bec31.smi"),
            ("COC", "5e2b10711364df45bf4127a9ede0610ac9168822.smi"),
            ("S", "c1291014e0a7cffcf96b8de1181fe61db70641c7.smi"),
        ],
    )
    def test_returns_expected_response_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query, count, filename
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)
        filepath = str(tmp_path / filename)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smiles": query, "persist": True}
        )

        assert response.json() == {"count": count, "filepath": filepath}

    @pytest.mark.parametrize(
        "query", ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"]
    )
    def test_returns_path_to_file_that_exists_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smiles": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])

        assert filepath.exists()

    @pytest.mark.parametrize(
        "query", ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"]
    )
    def test_returns_path_to_file_with_expected_hits_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query, hits
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.DIRECT_GRAPH_SCREEN, params={"smiles": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])
        actual = io.read_smi_file(filepath)

        assert actual == hits

    def test_returns_unsuccessful_response_given_no_query(self):
        response = client.get(URLPaths.DIRECT_GRAPH_SCREEN)

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Either a SMARTS or SMILES string query must be provided."
        }

    @pytest.mark.parametrize("query", ["CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smarts(self, query):
        response = client.get(URLPaths.DIRECT_GRAPH_SCREEN, params={"smarts": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}

    @pytest.mark.parametrize("query", ["c1cccc1", "CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smiles(self, query):
        response = client.get(URLPaths.DIRECT_GRAPH_SCREEN, params={"smiles": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}


class TestFingerprintScreen:
    def test_returns_successful_response_given_valid_smarts(self):
        response = client.get(
            URLPaths.FINGERPRINT_SCREEN,
            params={"smarts": "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1"},
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1", 3),
            ("[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1", 2),
            ("[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1", 2),
            ("[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1", 1),
            ("[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1", 5),
            ("[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1", 2),
            ("[#6]-[#6](=[#8])-[*]-[#6]", 3),
        ],
    )
    def test_returns_expected_hits_given_valid_smarts(self, query, expected):
        response = client.get(URLPaths.FINGERPRINT_SCREEN, params={"smarts": query})

        assert expected <= response.json() <= 10

    def test_returns_successful_response_given_valid_smiles(self):
        response = client.get(
            URLPaths.FINGERPRINT_SCREEN, params={"smiles": "c1ccccc1"}
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("C1NCCCC1", 3),
            ("C1NCCNC1", 2),
            ("c1ccccc1", 8),
            ("C=O", 5),
            ("COC", 6),
            ("S", 2),
        ],
    )
    def test_returns_expected_counts_given_valid_smiles(self, query, expected):
        response = client.get(URLPaths.FINGERPRINT_SCREEN, params={"smiles": query})

        assert expected <= response.json() <= 10

    def test_returns_unsuccessful_response_given_no_query(self):
        response = client.get(URLPaths.FINGERPRINT_SCREEN)

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Either a SMARTS or SMILES string query must be provided."
        }

    @pytest.mark.parametrize("query", ["CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smarts(self, query):
        response = client.get(URLPaths.FINGERPRINT_SCREEN, params={"smarts": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}

    @pytest.mark.parametrize("query", ["c1cccc1", "CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smiles(self, query):
        response = client.get(URLPaths.FINGERPRINT_SCREEN, params={"smiles": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}


class TestSubstructureSearch:
    @pytest.mark.parametrize("persist", [True, False])
    def test_returns_successful_response_given_valid_smarts(
        self, mocker, tmp_path, persist
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH,
            params={
                "smarts": "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
                "persist": persist,
            },
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_expected_hits_given_valid_smarts(self, query, count, hits):
        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smarts": query, "persist": False}
        )

        assert response.json() == {"count": count, "hits": hits}

    @pytest.mark.parametrize(
        "query, filename",
        [
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
                "515e05b051d4ed50266c244215d558a021d8359e.smi",
            ),
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
                "89dfa361c41b75f550ef97413eb60698e309676f.smi",
            ),
            (
                "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
                "35a0a7737e981b571462b3697eabd060abc4a166.smi",
            ),
            (
                "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
                "af2777b0386b81a63c8c902349d2b1d256a1d466.smi",
            ),
            (
                "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
                "465927e3fc8add302f4c25b734776c3baa06ae57.smi",
            ),
            (
                "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
                "4c1affd65c9f9e8ea08a4224e683cf6e3ddc8e9d.smi",
            ),
            (
                "[#6]-[#6](=[#8])-[*]-[#6]",
                "13f16c95201d2ef1e58ae59238fe2b7a433fd332.smi",
            ),
        ],
    )
    def test_returns_expected_response_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query, count, filename
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)
        filepath = str(tmp_path / filename)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smarts": query, "persist": True}
        )

        assert response.json() == {"count": count, "filepath": filepath}

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_path_to_file_that_exists_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smarts": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])

        assert filepath.exists()

    @pytest.mark.parametrize(
        "query",
        [
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[*])-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6](-[#6,#7])-[#6]-1",
            "[#6]1-[#7](-[*])-[#6]-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6](-[*])-[#6]-[#6]-[#6]-1",
            "[#6]1-[#7]-[#6]-[#6]-[#6,#7]-[#6]-1",
            "[#6]1:[#6]:[#7]:[#6,#7]:[#6]:[#6]:1",
            "[#6]-[#6](=[#8])-[*]-[#6]",
        ],
    )
    def test_returns_path_to_file_with_expected_hits_given_valid_smarts_and_persist_results(
        self, mocker, tmp_path, query, hits
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smarts": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])
        actual = io.read_smi_file(filepath)

        assert actual == hits

    @pytest.mark.parametrize("persist", [True, False])
    def test_returns_successful_response_given_valid_smiles(
        self, mocker, tmp_path, persist
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH,
            params={"smiles": "c1ccccc1", "persist": persist},
        )

        assert response.status_code == 200

    @pytest.mark.parametrize(
        "query",
        ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"],
    )
    def test_returns_expected_hits_given_valid_smiles(self, query, count, hits):
        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smiles": query, "persist": False}
        )

        assert response.json() == {"count": count, "hits": hits}

    @pytest.mark.parametrize(
        "query, filename",
        [
            ("C1NCCCC1", "18bc6610235fbbf025ec7c3c04ccd2c1c69c8872.smi"),
            ("C1NCCNC1", "5e99be7a17532f31e10e37d29ed51364a39e9610.smi"),
            ("c1ccccc1", "41e34a1b4adca57862763e2b51310d17412e90d8.smi"),
            ("C=O", "68ebf88c1bed4c4376db03a840a25623498bec31.smi"),
            ("COC", "5e2b10711364df45bf4127a9ede0610ac9168822.smi"),
            ("S", "c1291014e0a7cffcf96b8de1181fe61db70641c7.smi"),
        ],
    )
    def test_returns_expected_response_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query, count, filename
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)
        filepath = str(tmp_path / filename)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smiles": query, "persist": True}
        )

        assert response.json() == {"count": count, "filepath": filepath}

    @pytest.mark.parametrize(
        "query", ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"]
    )
    def test_returns_path_to_file_that_exists_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smiles": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])

        assert filepath.exists()

    @pytest.mark.parametrize(
        "query", ["C1NCCCC1", "C1NCCNC1", "c1ccccc1", "C=O", "COC", "S"]
    )
    def test_returns_path_to_file_with_expected_hits_given_valid_smiles_and_persist_results(
        self, mocker, tmp_path, query, hits
    ):
        mocker.patch("chemsearch.constants.RESULTS_DIR", tmp_path)

        response = client.get(
            URLPaths.SUBSTRUCTURE_SEARCH, params={"smiles": query, "persist": True}
        )
        filepath = Path(response.json()["filepath"])
        actual = io.read_smi_file(filepath)

        assert actual == hits

    def test_returns_unsuccessful_response_given_no_query(self):
        response = client.get(URLPaths.SUBSTRUCTURE_SEARCH)

        assert response.status_code == 400
        assert response.json() == {
            "detail": "Either a SMARTS or SMILES string query must be provided."
        }

    @pytest.mark.parametrize("query", ["CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smarts(self, query):
        response = client.get(URLPaths.SUBSTRUCTURE_SEARCH, params={"smarts": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}

    @pytest.mark.parametrize("query", ["c1cccc1", "CCC(C", "CCX"])
    def test_returns_unsuccessful_response_given_invalid_smiles(self, query):
        response = client.get(URLPaths.SUBSTRUCTURE_SEARCH, params={"smiles": query})

        assert response.status_code == 400
        assert response.json() == {"detail": "Query string could not be parsed."}
