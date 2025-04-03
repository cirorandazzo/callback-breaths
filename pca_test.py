# %%
# pca_test.py
# 
# use df all_breaths from umap-all_breaths.py

from sklearn.decomposition import PCA

mat = np.vstack(all_breaths.breath_interpolated)

mat.shape

# %%

pca = PCA()

pca.fit_transform(mat)

# %%

# dir(pca)
fig, ax = plt.subplots()

p_indiv, = ax.plot(pca.explained_variance_ratio_, label="by component", c="tab:blue")

ax1 = ax.twinx()
p_cumsum, = ax1.plot(np.cumsum(pca.explained_variance_ratio_), label="cumulative", c="tab:orange")

ax.legend(handles=[p_indiv, p_cumsum], labelcolor="linecolor", loc="center right")

ax.set(xlabel="PC", ylabel="explained variance", xlim=[-.9, 25])
ax1.set(ylabel="cumulative explained variance")


# %%

n_comp = 15
n_cols = 3

fig, axs = plt.subplots(ncols=n_cols, nrows=np.ceil(n_comp / n_cols).astype(int))

for i_comp, ax in enumerate(axs.ravel()):
    y = pca.components_[i_comp, :]
    x = np.linspace(0, 1, len(y))
    ax.plot(x, y)

    ax.set(title=f"PC{i_comp}")

fig.tight_layout()
