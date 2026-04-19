# Human essentiality & ID mapping (validation inputs)

Files in this folder are **not** authored in this repo; download with `../fetch_resources.py` (or URLs below).

| File | Source | Notes |
|------|--------|--------|
| `G-HumanEssential.tsv.gz` | [SNAP BioSNAP — Human gene essentiality](https://snap.stanford.edu/biodata/datasets/10033/10033-G-HumanEssential.html) | Entrez **Gene ID** + experimentally derived **Essential / Non-essential** labels (aggregated across studies). See dataset page for citations (OGEE / Chen *et al.*, NAR). |
| `Homo_sapiens.gene_info.gz` | [NCBI Gene — `GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz`](https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz) | Maps **GeneID → Symbol** for joining SNAP labels to PPI `Vertex` gene symbols. |

**License / use:** follow each provider’s terms (NCBI Gene FTP; SNAP academic datasets). Cite the original publications if you publish results.
