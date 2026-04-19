# Validation notebooks (human PPI)

## Layout

| Path | Role |
|------|------|
| [`01_embedding_vs_human_essentiality.ipynb`](01_embedding_vs_human_essentiality.ipynb) | SNAP essentiality joined to **`d2/`** and **`d3/`** runs; see subsection below. |
| [`embedding_essentiality.py`](embedding_essentiality.py) | Load / join helpers used by the notebook. |
| [`fetch_resources.py`](fetch_resources.py) | Download inputs into [`resources/`](resources/). |
| [`resources/SOURCES.md`](resources/SOURCES.md) | URLs and citations for downloaded files. |

### `01_embedding_vs_human_essentiality.ipynb`

- **Per dimension (D=2, then D=3):** Mann–Whitney / point-biserial on `Inf.Hyp.Rad`; violins for essential vs non-essential; chart radii (2D orthographic disk, 3D stereographic ball radius); logistic regression on standardized `Inf.Hyp.Rad` + **degree**.
- **Matched D=2 vs D=3:** inner-join on `Vertex`, then scatter (`hyp_d2` vs `hyp_d3`) with identity and OLS, Bland–Altman, Δ-radius violins, Pearson heatmap across `hyp` / disk / ball / degree, ROC curves and AUC bars for single-feature scores, and block-standardized logistic regression on `[hyp_d2, hyp_d3, degree]`.

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
- **Logistic regression** (per-dimension cells) uses standardized `Inf.Hyp.Rad` and `degree` to compare coefficient scales; the **D=2 vs D=3** block additionally fits a model on block-standardized `[hyp_d2, hyp_d3, degree]` for matched genes.
