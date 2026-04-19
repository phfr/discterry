"""Download validation resources into ``resources/``. Safe to re-run (skips if present)."""
from __future__ import annotations

import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "resources"

URLS = {
    "G-HumanEssential.tsv.gz": "https://snap.stanford.edu/biodata/datasets/10033/files/G-HumanEssential.tsv.gz",
    "Homo_sapiens.gene_info.gz": "https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz",
}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dest = RES / name
        if dest.is_file():
            print("skip exists:", dest)
            continue
        print("download", url)
        urllib.request.urlretrieve(url, dest)
        print("wrote", dest, f"({dest.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
