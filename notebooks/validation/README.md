# Validation notebooks (human PPI)

## Layout

| Path | Role |
|------|------|
| [`01_embedding_vs_human_essentiality.ipynb`](01_embedding_vs_human_essentiality.ipynb) | Join **SNAP human essentiality** (via NCBI Gene ID → symbol) to **`d-mercator-run/d2/`** and **`d3/`** embeddings; compare **essential vs non-essential** for `Inf.Hyp.Rad`, chart radii, and a logistic control for **degree**. |
| [`embedding_essentiality.py`](embedding_essentiality.py) | Load / join helpers used by the notebook. |
| [`fetch_resources.py`](fetch_resources.py) | Download inputs into [`resources/`](resources/). |
| [`resources/SOURCES.md`](resources/SOURCES.md) | URLs and citations for downloaded files. |

## Setup

1. From repo root (or this folder), download data:

   ```bash
   python notebooks/validation/fetch_resources.py
   ```

2. Open Jupyter with **kernel cwd** = `notebooks/validation/` (or repo root—see `_find_repo()` in the first notebook cell).

3. Run **`01_embedding_vs_human_essentiality.ipynb`** top to bottom.

## Dependencies

`pandas`, `numpy`, `matplotlib`, `scipy`, `scikit-learn`, `networkx` (already used elsewhere in the repo).

## Notes

- **Labels** are aggregated experimental calls from SNAP (OGEE-style); they are **context-dependent** and not gold truth for every cell type.
- **Gene symbol matching** is case-insensitive uppercasing; unmatched vertices remain unlabeled in tests.
- **Logistic regression** uses standardized `Inf.Hyp.Rad` and `degree` to compare coefficient scales.
