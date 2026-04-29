# Matched D=2 vs D=3 comparison (run after setup cell with `ess`)
import dmercator_io as dm2
import dmercator3d_io as dm3
from ball_projection import stereographic_s3_to_r3
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import StandardScaler

paths2 = dm2.paths_for_run("d2")
paths3 = dm3.paths_for_run("d3")
_, df2 = dm2.parse_inf_coord(paths2["inf_coord"])
_, df3 = dm3.parse_inf_coord(paths3["inf_coord"])
G = dm2.load_edges_graph(paths2["edge"])
deg_map = {str(a): int(b) for a, b in G.degree()}


def prep_d2(df):
    x, y = dm2.ortho_xy_disk(df)
    r_disk = np.sqrt(np.asarray(x) ** 2 + np.asarray(y) ** 2)
    d = df.assign(degree=df["Vertex"].astype(str).map(deg_map))
    d["degree"] = d["degree"].fillna(0).astype(int)
    return pd.DataFrame(
        {
            "Vertex": df["Vertex"],
            "hyp_d2": df["Inf.Hyp.Rad"].astype(float),
            "r_disk_d2": r_disk,
            "degree": d["degree"],
        }
    )


def prep_d3(df):
    U = dm3.normalize_direction_nd(df)
    x1, x2, x3, x4 = U[:, 0], U[:, 1], U[:, 2], U[:, 3]
    Xb, Yb, Zb = stereographic_s3_to_r3(x1, x2, x3, x4, pole="north")
    r_ball = np.sqrt(Xb * Xb + Yb * Yb + Zb * Zb)
    return pd.DataFrame(
        {
            "Vertex": df["Vertex"],
            "hyp_d3": df["Inf.Hyp.Rad"].astype(float),
            "r_ball_d3": r_ball,
        }
    )


A = prep_d2(df2)
B = prep_d3(df3)
M = A[["Vertex", "hyp_d2", "r_disk_d2", "degree"]].merge(
    B[["Vertex", "hyp_d3", "r_ball_d3"]], on="Vertex", how="inner"
)

E = ess.assign(_k=ess["gene_symbol"].astype(str).str.strip().str.upper()).drop_duplicates(subset=["_k"])[["_k", "essential"]]
comb = M.assign(_k=M["Vertex"].astype(str).str.strip().str.upper()).merge(E, on="_k", how="inner").drop(columns=["_k"])
lab = comb[comb["essential"].notna()].copy()
lab["essential"] = lab["essential"].astype(int)
print("matched vertices (inner join):", len(M))
print("with SNAP label:", len(lab))
print("Spearman hyp_d2 vs hyp_d3 (all matched):", *stats.spearmanr(M["hyp_d2"], M["hyp_d3"]))

# --- 1) Scatter hyp_d2 vs hyp_d3 + identity + OLS ---
fig1, ax = plt.subplots(figsize=(6.2, 6), dpi=150)
s0 = lab["essential"] == 0
s1 = lab["essential"] == 1
ax.scatter(lab.loc[s0, "hyp_d2"], lab.loc[s0, "hyp_d3"], s=8, alpha=0.25, c="0.45", label="non-essential")
ax.scatter(lab.loc[s1, "hyp_d2"], lab.loc[s1, "hyp_d3"], s=10, alpha=0.55, c="crimson", label="essential")
lo = float(min(lab["hyp_d2"].min(), lab["hyp_d3"].min()))
hi = float(max(lab["hyp_d2"].max(), lab["hyp_d3"].max()))
ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="identity")
slope, intercept, r_val, p_val, _ = stats.linregress(lab["hyp_d2"], lab["hyp_d3"])
xs = np.linspace(lo, hi, 80)
ax.plot(xs, slope * xs + intercept, color="darkorange", lw=2, label=f"OLS (r={r_val:.3f}, p={p_val:.1e})")
ax.set_xlabel("Inf.Hyp.Rad (D=2)")
ax.set_ylabel("Inf.Hyp.Rad (D=3)")
ax.set_title("Hyperbolic radius: D=2 vs D=3 (labeled genes)")
ax.legend(loc="upper left", fontsize=8)
ax.set_aspect("equal", adjustable="box")
fig1.tight_layout()
plt.show()

# --- 2) Bland–Altman (hyp) ---
mean_h = (lab["hyp_d2"] + lab["hyp_d3"]) / 2.0
diff_h = lab["hyp_d3"] - lab["hyp_d2"]
md = float(np.median(diff_h))
mad = 1.4826 * float(np.median(np.abs(diff_h - md)))
fig2, ax = plt.subplots(figsize=(6.5, 4), dpi=150)
ax.scatter(mean_h[s0], diff_h[s0], s=8, alpha=0.25, c="C0", label="non-essential")
ax.scatter(mean_h[s1], diff_h[s1], s=10, alpha=0.45, c="C3", label="essential")
ax.axhline(md, color="k", lw=1)
ax.axhline(md + 1.96 * mad, color="k", ls="--", lw=0.9)
ax.axhline(md - 1.96 * mad, color="k", ls="--", lw=0.9)
ax.set_xlabel("mean(hyp_d2, hyp_d3)")
ax.set_ylabel("hyp_d3 − hyp_d2")
ax.set_title("Bland–Altman; robust median ± 1.96·MAD")
ax.legend(loc="upper right", fontsize=8)
fig2.tight_layout()
plt.show()

# --- 3) Violin Δhyp by essential ---
fig3, ax = plt.subplots(figsize=(5, 4), dpi=150)
parts = [
    lab.loc[s0, "hyp_d3"].to_numpy() - lab.loc[s0, "hyp_d2"].to_numpy(),
    lab.loc[s1, "hyp_d3"].to_numpy() - lab.loc[s1, "hyp_d2"].to_numpy(),
]
ax.violinplot(parts, positions=[0, 1], showmeans=True, showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(["non-essential", "essential"])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("Δ Inf.Hyp.Rad (D3 − D2)")
ax.set_title("Embedding shift by essentiality")
fig3.tight_layout()
plt.show()

# --- 4) Correlation heatmap (labeled subset) ---
cols = ["hyp_d2", "hyp_d3", "r_disk_d2", "r_ball_d3", "degree"]
C = np.corrcoef(lab[cols].to_numpy().T)
fig4, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(cols)))
ax.set_yticks(range(len(cols)))
short = ["hyp2", "hyp3", "disk2", "ball3", "deg"]
ax.set_xticklabels(short, rotation=35, ha="right")
ax.set_yticklabels(short)
fig4.colorbar(im, ax=ax, fraction=0.046)
ax.set_title("Pearson correlation (labeled genes)")
fig4.tight_layout()
plt.show()

# --- 5–6) ROC + AUC bars ---
y = lab["essential"].to_numpy()
scores = {
    "−hyp_d2": -lab["hyp_d2"].to_numpy(),
    "−hyp_d3": -lab["hyp_d3"].to_numpy(),
    "−r_disk_d2": -lab["r_disk_d2"].to_numpy(),
    "−r_ball_d3": -lab["r_ball_d3"].to_numpy(),
    "degree": lab["degree"].to_numpy(),
}
fig5, ax = plt.subplots(figsize=(6, 5), dpi=150)
aucs = {}
for name, s in scores.items():
    fpr, tpr, _ = roc_curve(y, s)
    aucs[name] = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=1.5, label=f"{name} AUC={aucs[name]:.3f}")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("FPR")
ax.set_ylabel("TPR")
ax.set_title("ROC — single-feature (higher score ⇒ more essential)")
ax.legend(loc="lower right", fontsize=7)
fig5.tight_layout()
plt.show()

fig6, ax = plt.subplots(figsize=(6.2, 3.2), dpi=150)
names = list(aucs.keys())
vals = [aucs[n] for n in names]
colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
ax.barh(names, vals, color=colors)
ax.axvline(0.5, color="k", ls=":", lw=0.9)
ax.set_xlim(0.4, max(0.55, max(vals) + 0.03))
ax.set_xlabel("ROC AUC")
ax.set_title("Single-feature AUC (same labeled set)")
for i, v in enumerate(vals):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8)
fig6.tight_layout()
plt.show()

Xb = np.column_stack(
    [
        StandardScaler().fit_transform(lab[["hyp_d2", "hyp_d3"]].to_numpy()),
        StandardScaler().fit_transform(lab[["degree"]].to_numpy()),
    ]
)
clf_b = LogisticRegression(max_iter=400)
clf_b.fit(Xb, y)
print("Logistic coef [hyp_d2, hyp_d3, degree] (std blocks):", clf_b.coef_[0])
