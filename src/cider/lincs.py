from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from cider.paths import resolve_path


RELEASES = {
    "GSE92742": {
        "series": "GSE92nnn",
        "matrix": "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz",
        "metadata_suffix": "",
    },
    "GSE70138": {
        "series": "GSE70nnn",
        "matrix": "GSE70138_Broad_LINCS_Level5_COMPZ_n118050x12328_2017-03-06.gctx.gz",
        "metadata_suffix": "_2017-03-06",
    },
}
METADATA = ("sig_info", "sig_metrics", "pert_info", "gene_info")
NUMBER = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def release_files(accession):
    release = RELEASES[accession]
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{release['series']}/{accession}/suppl/"
    names = [Path("level5") / release["matrix"]]
    names.extend(
        Path("metadata")
        / f"{accession}_Broad_LINCS_{name}{release['metadata_suffix']}.txt.gz"
        for name in METADATA
    )
    return [(path, base + path.name) for path in names]


def download_file(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            size = response.headers.get("Content-Length")
            size = int(size) if size is not None else None
            count = 0
            with partial.open("wb") as stream, tqdm(
                total=size, unit="B", unit_scale=True, desc=path.name
            ) as progress:
                while True:
                    block = response.read(8 * 1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
                    count += len(block)
                    progress.update(len(block))
            if size is not None and count != size:
                raise IOError(f"Incomplete download: {path.name}")
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def decompress_matrix(path):
    output = path.with_suffix("")
    if output.is_file():
        return output
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run with --download first.")
    partial = output.with_name(output.name + ".part")
    try:
        print(f"Decompressing {path.name}", flush=True)
        with gzip.open(path, "rb") as source, partial.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        partial.replace(output)
    finally:
        if partial.exists():
            partial.unlink()
    return output


def read_metadata(raw_dir, accession, min_tas):
    paths = dict(zip(METADATA, [p for p, _ in release_files(accession)][1:]))
    tables = {
        name: pd.read_csv(raw_dir / path, sep="\t", low_memory=False)
        for name, path in paths.items()
    }
    obs = tables["sig_info"]
    obs = obs.loc[obs["pert_type"].eq("trt_cp")].copy()
    for column in [
        "sig_id",
        "pert_id",
        "pert_iname",
        "cell_id",
        "pert_idose",
        "pert_itime",
    ]:
        obs[column] = obs[column].astype(str)
    for column in ["pert_id", "pert_iname", "cell_id"]:
        obs = obs.loc[~obs[column].isin(["nan", "None", ""])].copy()
    quality = tables["sig_metrics"]
    columns = [
        column
        for column in [
            "sig_id",
            "distil_cc_q75",
            "distil_ss",
            "distil_nsample",
            "tas",
            "is_gold",
            "pct_self_rank_q25",
            "wt",
        ]
        if column in quality.columns
    ]
    obs = obs.merge(
        quality[columns].drop_duplicates("sig_id"),
        on="sig_id",
        how="left",
        validate="many_to_one",
    )
    obs = obs.loc[pd.to_numeric(obs["tas"], errors="coerce").ge(min_tas)].copy()
    compounds = tables["pert_info"]
    columns = ["pert_id"] + [c for c in compounds.columns if c not in obs.columns]
    obs = obs.merge(
        compounds[columns].drop_duplicates("pert_id"),
        on="pert_id",
        how="left",
        validate="many_to_one",
    )
    for source, prefix in [("pert_idose", "dose"), ("pert_itime", "time")]:
        obs[f"{prefix}_value"] = pd.to_numeric(
            obs[source].str.extract(f"({NUMBER})", expand=False), errors="coerce"
        )
        obs[f"{prefix}_unit"] = (
            obs[source].str.replace(NUMBER, "", regex=True).str.strip()
        )
    obs = obs.set_index("sig_id")
    genes = tables["gene_info"]
    genes = genes.loc[pd.to_numeric(genes["pr_is_lm"], errors="coerce").eq(1)].copy()
    genes["pr_gene_id"] = genes["pr_gene_id"].astype(str)
    genes = genes.drop_duplicates("pr_gene_id").set_index("pr_gene_id")
    if len(genes) != 978:
        raise ValueError(f"Expected 978 landmark genes, found {len(genes)}")
    if not obs.index.is_unique or obs.empty:
        raise ValueError("Signature identifiers must be unique and non-empty.")
    return obs, genes


def read_matrix(path, signatures, landmark_ids, chunk_size=2048):
    with h5py.File(path, "r") as handle:
        rows = pd.Index(handle["0/META/COL/id"].asstr()[:])
        columns = pd.Index(handle["0/META/ROW/id"].asstr()[:])
        gene_indices = np.flatnonzero(columns.isin(landmark_ids))
        row_indices = rows.get_indexer(signatures)
        if np.any(row_indices < 0):
            raise ValueError("Some signatures are missing from the Level 5 matrix.")
        if len(gene_indices) != len(landmark_ids):
            raise ValueError("Some landmark genes are missing from the Level 5 matrix.")
        matrix = handle["0/DATA/0/matrix"]
        if matrix.shape != (len(rows), len(columns)):
            raise ValueError(f"Unexpected GCTX matrix shape: {matrix.shape}")
        result = np.empty((len(signatures), len(gene_indices)), dtype=np.float32)
        starts = np.unique((row_indices // chunk_size) * chunk_size)
        for start in tqdm(starts, desc=path.stem):
            stop = min(int(start) + chunk_size, len(rows))
            selected = np.flatnonzero((row_indices >= start) & (row_indices < stop))
            block = matrix[int(start) : stop, :]
            result[selected] = block[
                np.ix_(row_indices[selected] - start, gene_indices)
            ]
    result[result == -666] = np.nan
    if not np.isfinite(result).all():
        raise ValueError("Non-finite values in the selected landmark responses.")
    return result, columns[gene_indices]


def prepare_lincs(raw_dir, out, min_tas=0.2, chunk_size=2048):
    phases = []
    for index, accession in enumerate(RELEASES, start=1):
        obs, genes = read_metadata(raw_dir, accession, min_tas)
        matrix_path = decompress_matrix(
            raw_dir / "level5" / RELEASES[accession]["matrix"]
        )
        values, gene_ids = read_matrix(matrix_path, obs.index, genes.index, chunk_size)
        var = genes.loc[gene_ids].copy()
        var["gene_id"] = var.index
        var["gene_symbol"] = var["pr_gene_symbol"].astype(str)
        obs["phase"] = f"phase{index}_{accession}"
        obs.index = f"phase{index}_" + obs.index
        for frame in [obs, var]:
            for column in frame.select_dtypes(include="object").columns:
                frame[column] = frame[column].astype(str)
        phases.append(ad.AnnData(X=values, obs=obs, var=var))
        print(f"{accession}: {len(obs):,} signatures", flush=True)
    order = phases[0].var_names
    aligned = [phase[:, order].copy() for phase in phases]
    merged = ad.concat(
        aligned,
        join="inner",
        label="batch_phase",
        keys=["phase1", "phase2"],
        index_unique=None,
    )
    merged.var = aligned[0].var.copy()
    merged.uns["source"] = "LINCS L1000 Level 5; GSE92742 + GSE70138"
    merged.uns["feature_space"] = "978 landmark genes"
    merged.uns["perturbation_type"] = "trt_cp"
    merged.uns["min_tas"] = min_tas
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(out, compression="gzip")
    print(f"Saved {out}: {merged.n_obs:,} signatures, {merged.n_vars} genes")


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare LINCS L1000 Level 5 small-molecule signatures."
    )
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--out", default="data/processed/lincs_cp_landmark_all.h5ad")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--min-tas", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=2048)
    args = parser.parse_args()
    if args.chunk_size <= 0 or not np.isfinite(args.min_tas):
        parser.error("--chunk-size must be positive and --min-tas must be finite.")
    raw_dir = resolve_path(args.raw_dir)
    out = resolve_path(args.out)
    if not args.download_only and out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    if args.download or args.download_only:
        for accession in RELEASES:
            for path, url in release_files(accession):
                if (
                    path.parent.name == "level5"
                    and (raw_dir / path.with_suffix("")).is_file()
                ):
                    continue
                download_file(url, raw_dir / path)
    if not args.download_only:
        prepare_lincs(raw_dir, out, args.min_tas, args.chunk_size)


if __name__ == "__main__":
    main()
