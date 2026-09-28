import gzip
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from cider.lincs import (
    decompress_matrix,
    download_file,
    read_matrix,
    read_metadata,
    release_files,
)


class DataTests(unittest.TestCase):
    def test_geo_download_urls(self):
        for accession in ["GSE92742", "GSE70138"]:
            files = release_files(accession)
            self.assertEqual(len(files), 5)
            for path, url in files:
                self.assertFalse(path.is_absolute())
                self.assertTrue(
                    url.startswith("https://ftp.ncbi.nlm.nih.gov/geo/series/")
                )
                self.assertTrue(url.endswith(path.name))
                self.assertIn(accession, path.name)

    def test_download_and_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "metadata.txt.gz"
            response = io.BytesIO(b"data")
            response.headers = {"Content-Length": "4"}
            with patch("urllib.request.urlopen", return_value=response) as request:
                download_file("https://ftp.ncbi.nlm.nih.gov/geo/", output)
                download_file("https://ftp.ncbi.nlm.nih.gov/geo/", output)
                self.assertEqual(request.call_count, 1)
            self.assertEqual(output.read_bytes(), b"data")
            self.assertFalse(output.with_name(output.name + ".part").exists())

    def test_incomplete_download_is_not_reused(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "metadata.txt.gz"
            response = io.BytesIO(b"data")
            response.headers = {"Content-Length": "8"}
            with patch("urllib.request.urlopen", return_value=response):
                with self.assertRaises(IOError):
                    download_file("https://ftp.ncbi.nlm.nih.gov/geo/", output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_name(output.name + ".part").exists())

    def test_matrix_decompression(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.gctx.gz"
            path.write_bytes(gzip.compress(b"matrix"))
            output = decompress_matrix(path)
            self.assertEqual(output.read_bytes(), b"matrix")
            path.unlink()
            self.assertEqual(decompress_matrix(path), output)

    def test_signature_and_gene_alignment(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "matrix.gctx"
            matrix = np.arange(12, dtype=np.float32).reshape(3, 4)
            with h5py.File(path, "w") as handle:
                handle.create_dataset(
                    "0/META/COL/id", data=np.array([b"s2", b"s1", b"s3"])
                )
                handle.create_dataset(
                    "0/META/ROW/id", data=np.array([b"g3", b"g2", b"g4", b"g1"])
                )
                handle.create_dataset("0/DATA/0/matrix", data=matrix)
            result, genes = read_matrix(path, ["s3", "s1"], ["g1", "g2"], chunk_size=2)
            self.assertEqual(genes.tolist(), ["g2", "g1"])
            np.testing.assert_array_equal(result, matrix[np.ix_([2, 1], [1, 3])])
            with self.assertRaises(ValueError):
                read_matrix(path, ["missing"], ["g1"])

    def test_quality_filter_and_context(self):
        with tempfile.TemporaryDirectory() as folder:
            raw = Path(folder)
            signatures = pd.DataFrame(
                {
                    "sig_id": ["s1", "s2", "s3"],
                    "pert_id": ["p1"] * 3,
                    "pert_iname": ["compound"] * 3,
                    "pert_type": ["trt_cp", "trt_cp", "trt_sh"],
                    "cell_id": ["A375"] * 3,
                    "pert_idose": ["10.0 um"] * 3,
                    "pert_itime": ["24 h"] * 3,
                }
            )
            quality = pd.DataFrame(
                {"sig_id": ["s1", "s2", "s3"], "tas": [0.2, 0.19, 0.8]}
            )
            compounds = pd.DataFrame({"pert_id": ["p1"], "canonical_smiles": ["CCO"]})
            genes = pd.DataFrame(
                {
                    "pr_gene_id": list(range(978)),
                    "pr_gene_symbol": [f"G{i}" for i in range(978)],
                    "pr_is_lm": [1] * 978,
                }
            )
            paths = [p for p, _ in release_files("GSE92742")][1:]
            for path, frame in zip(paths, [signatures, quality, compounds, genes]):
                (raw / path).parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(raw / path, sep="\t", index=False)
            obs, var = read_metadata(raw, "GSE92742", min_tas=0.2)
            self.assertEqual(obs.index.tolist(), ["s1"])
            self.assertEqual(obs.loc["s1", "canonical_smiles"], "CCO")
            self.assertEqual(obs.loc["s1", "dose_value"], 10)
            self.assertEqual(obs.loc["s1", "dose_unit"], "um")
            self.assertEqual(obs.loc["s1", "time_value"], 24)
            self.assertEqual(len(var), 978)


if __name__ == "__main__":
    unittest.main()
