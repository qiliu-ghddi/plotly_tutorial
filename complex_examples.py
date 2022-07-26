# %% [markdown]
# # More complex examples
# 
# Let's go beyond scatter plots and explore a few other graphs that might be relevant for cheminformatics, hopefully letting you see how `molplotly` could be useful for you when looking through (messy) data!

# %% [markdown]
# ## Imports and Data Loading

# %% [markdown]
# Import pandas for data manipulation, plotly for plotting, and molplot for visualising structures!

# %%
import pandas as pd
import plotly.express as px
import molplotly


# %% [markdown]
# Let's load the ESOL dataset from [ESOL: Estimating Aqueous Solubility Directly from Molecular Structure](https://doi.org/10.1021/ci034243x) - helpfully hosted by the [deepchem](https://github.com/deepchem/deepchem) team but also included as `example.csv` in the repo.

# %%
df_esol = pd.read_csv('example.csv')
# df_esol = pd.read_csv(
#     'https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv')
df_esol['y_pred'] = df_esol['ESOL predicted log solubility in mols per litre']
df_esol['y_true'] = df_esol['measured log solubility in mols per litre']


# %% [markdown]
# ## 3D Scatter Plots

# %% [markdown]
# The default coordinates settings don't work with 3D scatter plots so you will need to set `show_coords=False` and manually put in the coordinates into `caption_cols`.
# 
# 

# %%
fig_3d = px.scatter_3d(df_esol, 
                       x="y_pred", 
                       y="y_true", 
                       z="Molecular Weight", 
                       width=1000,
                       height=800)

app_3d = molplotly.add_molecules(
    fig=fig_3d,
    df=df_esol,
    smiles_col="smiles",
    caption_cols=["y_pred", "y_true", "Molecular Weight"],
    show_coords=False,
)

app_3d.run_server(mode="inline", port=8704, height=850)


# %% [markdown]
# ## Strip plots
# 
# Strip plots are useful for visualising how the same property is distributed between data from different groups. Here I plot how the measured solubility changes with the number of rings on a molecule (it goes down, surprising I know).
# 
# Violin plots can also useful for this purpose but it's not compatible with `plotly` (see section ["violin plots"](#violin)) 

# %%
fig_strip = px.strip(df_esol.sort_values('Number of Rings'), # sorting so that the colorbar is sorted!
                     x='Number of Rings',
                     y='y_true',
                     color='Number of Rings',
                     labels={'y_true': 'Measured Solubility'},
                     width=1000,
                     height=800)

app_strip = molplotly.add_molecules(fig=fig_strip,
                          df=df_esol,
                          smiles_col='smiles',
                          title_col='Compound ID',
                          color_col='Number of Rings',
                          caption_transform={'Measured Solubility': lambda x: f"{x:.2f}"},
                          wrap=True,
                          wraplen=25,
                          width=150,
                          show_coords=True)

app_strip.run_server(mode='inline', port=8705, height=850)


# %% [markdown]
# ## Scatter Matrices
# 
# For visualising the relationship between multiple variables at once, use a matrix of scatter plots!
# 
# Here I've increased the width of the hover box using the `width` parameter because the caption titles were getting long; also I've used `show_coords=False` because $(x, y)$ coordinates for non-trivial scatter plots become messy.

# %%
features = ['Number of H-Bond Donors',
            'Number of Rings',
            'Number of Rotatable Bonds',
            'Polar Surface Area']
fig_matrix = px.scatter_matrix(df_esol,
                               dimensions=features,
                               width=1200,
                               height=800,
                               title='Scatter matrix of molecular properties')

app_matrix = molplotly.add_molecules(fig=fig_matrix,
                                     df=df_esol,
                                     smiles_col='smiles',
                                     title_col='Compound ID',
                                     caption_cols=features,
                                     width=200,
                                     show_coords=False)

# Only show informative lower triangle
fig_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
app_matrix.run_server(mode='inline', port=8706, height=1000)


# %% [markdown]
# ## Visualising MorganFP PCA components
# 
# A common way to visualise a molecular dataset is to calculate the morgan fingerprints of the molecules and visualise them in a 2D embedding (eg PCA/t-SNE). In this example I'm going to plot the 2 largest PCA components for ESOL and inspect the data.

# %% [markdown]
# Let's calculate the PCA components first!

# %%
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.decomposition import PCA


def smi_to_fp(smi):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(smi), 2, nBits=1024)
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

esol_fps = np.array([smi_to_fp(smi) for smi in df_esol['smiles']])
pca = PCA(n_components=2)
components = pca.fit_transform(esol_fps.reshape(-1, 1024))
df_esol['PCA-1'] = components[:, 0]
df_esol['PCA-2'] = components[:, 1]


# %% [markdown]
# and now let's look at them!
# 
# with `molplotly`, it's super easy to see which molecules are where - steroid molecules at the top, alcohols in the bottom left, chlorinated aromatic compounds in the bottom right.

# %%
fig_pca = px.scatter(df_esol,
                     x="PCA-1",
                     y="PCA-2",
                     color='y_true',
                     title='ESOL PCA of morgan fingerprints',
                     labels={'y_true': 'Measured Solubility'},
                     width=1200,
                     height=800)

app_pca = molplotly.add_molecules(fig=fig_pca,
                                  df=df_esol.rename(columns={'y_true': 'Measured Solubility'}),
                                  smiles_col='smiles',
                                  title_col='Compound ID',
                                  caption_cols=['Measured Solubility'],
                                  caption_transform={'Measured Solubility': lambda x: f"{x:.2f}"},
                                  color_col='Measured Solubility',
                                  show_coords=False)

app_pca.run_server(mode='inline', port=8707, height=850)


# %% [markdown]
# ## Clustering
# 
# Let's do some clustering of the ESOL molecules, borrowing useful functions from Pat Walters' excellent blog post on [clustering](http://practicalcheminformatics.blogspot.com/2021/11/picking-highest-scoring-molecules-from.html).

# %%
from rdkit.ML.Cluster import Butina

def smi2fp(smi):
    fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2)
    return fp


def taylor_butina_clustering(fp_list, cutoff=0.35):
    dists = []
    nfps = len(fp_list)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list[:i])
        dists.extend([1-x for x in sims])
    mol_clusters = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return mol_clusters


cluster_res = taylor_butina_clustering(
    [smi2fp(smi) for smi in df_esol['smiles']])
cluster_id_list = np.zeros(len(df_esol), dtype=int)
for cluster_num, cluster in enumerate(cluster_res):
    for member in cluster:
        cluster_id_list[member] = cluster_num
df_esol['cluster'] = cluster_id_list


# %% [markdown]
# Now let's make a strip plot of the top-10 clusters, see what they look like and how soluable they are!

# %%
df_cluster = df_esol.query('cluster < 10').copy().reset_index()
# sorting is needed to make the legend appear in order!
df_cluster = df_cluster.sort_values('cluster')

fig_cluster = px.strip(df_cluster,
                      y='y_true',
                      color='cluster',
                      labels={'y_true': 'Measured Solubility'},
                      width=1000,
                      height=800)

app_cluster = molplotly.add_molecules(fig=fig_cluster,
                           df=df_cluster,
                           smiles_col='smiles',
                           title_col='Compound ID',
                           color_col='cluster'
                           )

app_cluster.run_server(mode='inline', port=8708, height=850)



